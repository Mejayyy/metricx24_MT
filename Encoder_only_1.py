import os
import pandas as pd
import torch
from torch import nn
import copy
import dataclasses
from typing import Optional, Tuple, Union
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    MT5PreTrainedModel,
    MT5Config,
    Trainer,
    TrainingArguments
)
from transformers.models.mt5.modeling_mt5 import MT5Stack
import transformers.modeling_outputs
from datasets import Dataset

# Types
BaseModelOutput = transformers.modeling_outputs.BaseModelOutput
ModelOutput = transformers.modeling_outputs.ModelOutput
MT5Config = transformers.models.mt5.modeling_mt5.MT5Config
MT5PreTrainedModel = transformers.models.mt5.modeling_mt5.MT5PreTrainedModel
MT5Stack = transformers.models.mt5.modeling_mt5.MT5Stack


@dataclasses.dataclass
class MT5ForRegressionOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    predictions: torch.FloatTensor = None


class MT5EncoderForRegression(MT5PreTrainedModel):
    """MT5 Encoder-only model for regression."""

    def __init__(self, config: MT5Config):
        super().__init__(config)
        self.model_dim = config.d_model

        # 1. Shared Embeddings
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        # 2. Encoder Only (We copy config and ensure it's set to encoder mode)
        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = MT5Stack(encoder_config, self.shared)

        # 3. NO DECODER (We removed it to save memory and mimicking MetricX-25)

        # 4. Regression Head
        # Takes the pooled encoder output and projects to scalar
        self.lm_head=nn.Linear(config.d_model, 1)
        """
        self.lm_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),  # Optional: Dense layer for feature extraction
            nn.Tanh(),  # Non-linearity
            nn.Linear(config.d_model, 1),  # Project to scalar
            nn.Sigmoid()  # Squash to [0, 1]
        )
        """

        # Initialize weights
        self.post_init()

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.FloatTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            **kwargs  # Catch unused args (like decoder_* vars) that Trainer might pass
    ) -> Union[Tuple[torch.FloatTensor], MT5ForRegressionOutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 1. Run Encoder
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # hidden_states shape: [Batch, Seq_Len, Dim]
        hidden_states = encoder_outputs[0]

        # 2. Mean Pooling
        # We must mask out padding tokens so they don't drag down the average.
        if attention_mask is not None:
            # Expand mask: [Batch, Seq] -> [Batch, Seq, Dim]
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()

            # Sum embeddings of non-padding tokens
            sum_embeddings = torch.sum(hidden_states * input_mask_expanded, 1)

            # Count non-padding tokens
            sum_mask = input_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)  # Avoid div by zero

            # Average
            pooled_output = sum_embeddings / sum_mask  # Shape: [Batch, Dim]
        else:
            # Fallback if no mask (rare)
            pooled_output = torch.mean(hidden_states, dim=1)

        # 3. Predict Score
        logits = self.lm_head(pooled_output)  # Shape: [Batch, 1]
        predictions = logits.squeeze(-1)  # Shape: [Batch]

        # 4. Scale Sigmoid Output [0, 1] -> [0, 25]
        # This matches your previous logic for MQM scores
        #predictions = predictions * 25.0

        loss = None
        if labels is not None:
            loss_fct = nn.MSELoss()
            # Move labels to correct device
            labels = labels.to(predictions.device)
            loss = loss_fct(predictions, labels.view(-1))

        return MT5ForRegressionOutput(
            loss=loss,
            predictions=predictions,
        )


def load_preformatted_data(filepath):
    print(f"Loading pre-formatted data from {filepath}...")
    df = pd.read_json(filepath, lines=True)
    df = df.rename(columns={"score": "labels"})
    df["labels"] = pd.to_numeric(df["labels"], errors='coerce')
    df = df.dropna(subset=["labels", "input_text"])
    return df[["input_text", "labels"]]


def check_gradients(trainer, model):
    print("--- Verifying Weight Updates ---")

    # Grab a weight from the encoder (which still exists)
    param_before = model.encoder.block[0].layer[0].SelfAttention.q.weight.clone()

    # Run one training step
    train_dataloader = trainer.get_train_dataloader()
    batch = next(iter(train_dataloader))
    batch = {k: v.to(model.device) for k, v in batch.items()}

    model.train()
    outputs = model(**batch)
    loss = outputs.loss
    loss.backward()

    optimizer = trainer.create_optimizer()
    optimizer.step()
    optimizer.zero_grad()

    param_after = model.encoder.block[0].layer[0].SelfAttention.q.weight
    diff = torch.sum(torch.abs(param_after - param_before)).item()

    if diff > 0:
        print(f"SUCCESS: Weights updated! Difference: {diff}")
    else:
        print("FAILURE: Weights did not change. Check gradients.")
    print("--------------------------------")


def train():
    MODEL_NAME = "google/mt5-base"
    JSONL_FILE = "wmt21_mqm_ende.jsonl"
    MAX_LENGTH = 256

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    df = load_preformatted_data(JSONL_FILE)
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)

    # DEBUG: Cap sizes for testing speed
    train_df = train_df.iloc[:250].reset_index(drop=True)
    val_df = val_df.iloc[:100].reset_index(drop=True)

    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, model_max_length=MAX_LENGTH)

    def tokenize_function(examples):
        return tokenizer(
            examples["input_text"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH
        )

    print("Tokenizing data...")
    cols_to_remove = [c for c in train_dataset.column_names if c != "labels"]

    tokenized_train = train_dataset.map(tokenize_function, batched=True, remove_columns=cols_to_remove,
                                        load_from_cache_file=False)
    tokenized_val = val_dataset.map(tokenize_function, batched=True, remove_columns=cols_to_remove,
                                    load_from_cache_file=False)

    print("Initializing Encoder-Only Model...")

    # We use strict=False implicitly by ignoring the "Some weights... were not used" warning.
    # The loading function will successfully load the encoder weights and ignore the decoder weights from the checkpoint.
    model = MT5EncoderForRegression.from_pretrained(MODEL_NAME,
                                                    ignore_mismatched_sizes=True)

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = predictions.flatten()
        labels = labels.flatten()
        mse = ((predictions - labels) ** 2).mean()
        return {"mse": mse}

    training_args = TrainingArguments(
        output_dir="./mt5-encoder-metric-output",
        learning_rate=1e-5,

        # CRITICAL: Keep this False to prevent the shared tensor crash
        save_safetensors=False,

        # --- LOGGING SETTINGS ---
        logging_strategy="steps",
        logging_steps=2,  # <--- Prints loss every step
        # ------------------------

        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=4,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="mse",
        greater_is_better=False,
        dataloader_num_workers=0,
        remove_unused_columns=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    print("Starting Training...")
    # Verify gradients before full loop
    check_gradients(trainer, model)

    trainer.train()

    trainer.save_model("./mt5-encoder-metric-final")
    tokenizer.save_pretrained("./mt5-encoder-metric-final")
    print("Done.")

if __name__ == "__main__":
    print()
    # train()