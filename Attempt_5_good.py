import os


import pandas as pd

import torch
from torch import nn
import copy
import dataclasses

from typing import Optional, Tuple, Union
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr, kendalltau
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
import transformers

BaseModelOutput = transformers.modeling_outputs.BaseModelOutput
ModelOutput = transformers.modeling_outputs.ModelOutput
MT5Config = transformers.models.mt5.modeling_mt5.MT5Config
MT5PreTrainedModel = transformers.models.mt5.modeling_mt5.MT5PreTrainedModel
MT5Stack = transformers.models.mt5.modeling_mt5.MT5Stack

__HEAD_MASK_WARNING_MSG = (
    #transformers.models.mt5.modeling_mt5.__HEAD_MASK_WARNING_MSG  # pylint: disable=protected-access
    "Warning HEAD_MASK_WARNING_MSG"
)

@dataclasses.dataclass
class MT5ForRegressionOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    predictions: torch.FloatTensor = None


from transformers import MT5Model


class MT5ForRegression(MT5PreTrainedModel):
  """MT5 model for regression."""

  def __init__(self, config: MT5Config):
    super().__init__(config)

    self.model_dim = config.d_model

    self.shared = nn.Embedding(config.vocab_size, config.d_model)

    encoder_config = copy.deepcopy(config)
    encoder_config.is_decoder = False
    encoder_config.use_cache = False
    encoder_config.is_encoder_decoder = False
    self.encoder = MT5Stack(encoder_config, self.shared)

    decoder_config = copy.deepcopy(config)
    decoder_config.is_decoder = True
    decoder_config.is_encoder_decoder = False
    decoder_config.num_layers = config.num_decoder_layers
    self.decoder = MT5Stack(decoder_config, self.shared)

    self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
    #'train_runtime': 159.4742
    #self.lm_head = nn.Linear(config.d_model,1,bias=False)

    # Initialize weights and apply final processing
    self.post_init()

    # Model parallel
    self.model_parallel = False
    self.device_map = None

  def forward(
      self,
      input_ids: Optional[torch.LongTensor] = None,
      attention_mask: Optional[torch.FloatTensor] = None,
      decoder_attention_mask: Optional[torch.BoolTensor] = None,
      head_mask: Optional[torch.FloatTensor] = None,
      decoder_head_mask: Optional[torch.FloatTensor] = None,
      cross_attn_head_mask: Optional[torch.Tensor] = None,
      encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
      past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
      inputs_embeds: Optional[torch.FloatTensor] = None,
      decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
      labels: Optional[torch.FloatTensor] = None,
      use_cache: Optional[bool] = None,
      output_attentions: Optional[bool] = None,
      output_hidden_states: Optional[bool] = None,
      return_dict: Optional[bool] = None,
  ) -> Union[Tuple[torch.FloatTensor], MT5ForRegressionOutput]:
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = (
        return_dict if return_dict is not None else self.config.use_return_dict
    )
    # --- DEBUG PRINTS ---

    # FutureWarning: head_mask was separated into two input args - head_mask,
    # decoder_head_mask
    if head_mask is not None and decoder_head_mask is None:
      if self.config.num_layers == self.config.num_decoder_layers:
        print("\nHEAD_MASK_MSG WARNING \n")
        #warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
        decoder_head_mask = head_mask

    # Encode if needed (training, first prediction pass)
    if encoder_outputs is None:
      # Convert encoder inputs in embeddings if needed
      encoder_outputs = self.encoder(
          input_ids=input_ids,
          attention_mask=attention_mask,
          inputs_embeds=inputs_embeds,
          head_mask=head_mask,
          output_attentions=output_attentions,
          output_hidden_states=output_hidden_states,
          return_dict=return_dict,
      )

    elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
      encoder_outputs = BaseModelOutput(
          last_hidden_state=encoder_outputs[0],
          hidden_states=encoder_outputs[1]
          if len(encoder_outputs) > 1
          else None,
          attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
      )

    hidden_states = encoder_outputs[0]


    if self.model_parallel:
      torch.cuda.set_device(self.decoder.first_device)

    # Create 1 step of dummy input for the decoder.

    batch_size = input_ids.size(0)



    decoder_input_ids = torch.LongTensor([0]).repeat(batch_size).reshape(-1, 1)


    if torch.cuda.is_available():
      decoder_input_ids = decoder_input_ids.to(torch.device("cuda"))


    if self.model_parallel:
      torch.cuda.set_device(self.decoder.first_device)
      hidden_states = hidden_states.to(self.decoder.first_device)
      if decoder_input_ids is not None:
        decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
      if attention_mask is not None:
        attention_mask = attention_mask.to(self.decoder.first_device)
      if decoder_attention_mask is not None:
        decoder_attention_mask = decoder_attention_mask.to(
            self.decoder.first_device
        )

    # Decode

    decoder_outputs = self.decoder(
        input_ids=decoder_input_ids,
        attention_mask=decoder_attention_mask,
        inputs_embeds=decoder_inputs_embeds,
        past_key_values=past_key_values,
        encoder_hidden_states=hidden_states,
        encoder_attention_mask=attention_mask,
        head_mask=decoder_head_mask,
        cross_attn_head_mask=cross_attn_head_mask,
        use_cache=False,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    sequence_output = decoder_outputs[0]

    # Set device for model parallelism
    if self.model_parallel:
      torch.cuda.set_device(self.encoder.first_device)
      self.lm_head = self.lm_head.to(self.encoder.first_device)
      sequence_output = sequence_output.to(self.lm_head.weight.device)

    if self.config.tie_word_embeddings:
      # Rescale output before projecting on vocab
      # See
      # https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
      sequence_output = sequence_output * (self.model_dim**-0.5)

    #print("\n Sequence_output Shape \n",sequence_output.shape)
    lm_logits = self.lm_head(sequence_output)

    # 250089 = <extra_id_10>
    #print("\n lm_logits shape : \n",lm_logits.shape)
    predictions = lm_logits[:, 0, 250089]
    #predictions = lm_logits[:, 0, 0]


    # Clip to 0 to 25
    #predictions = torch.clamp(predictions, 0, 25)

    loss = None
    if labels is not None:
      loss_fct = nn.MSELoss()
      # move labels to correct device to enable PP
      labels = labels.to(predictions.device)
      loss = loss_fct(predictions.view(-1), labels.view(-1))

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

    # 1. Grab a specific weight (e.g., from the encoder)
    param_before = model.encoder.block[0].layer[0].SelfAttention.q.weight.clone()

    # 2. Run one training step
    train_dataloader = trainer.get_train_dataloader()
    batch = next(iter(train_dataloader))

    # Move batch to device
    batch = {k: v.to(model.device) for k, v in batch.items()}

    model.train()
    outputs = model(**batch)
    loss = outputs.loss
    loss.backward()

    # Optimizer step
    optimizer = trainer.create_optimizer()
    optimizer.step()
    optimizer.zero_grad()

    # 3. Check if changed
    param_after = model.encoder.block[0].layer[0].SelfAttention.q.weight
    diff = torch.sum(torch.abs(param_after - param_before)).item()

    if diff > 0:
        print(f"SUCCESS: Weights updated! Difference: {diff}")
    else:
        print("FAILURE: Weights did not change. Check gradients (likely clamp issue).")
    print("--------------------------------")



def train():
    MODEL_NAME = "google/mt5-base"
    JSONL_FILE = "wmt21_mqm_ende.jsonl"

    # Back to standard 512
    MAX_LENGTH = 256

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    df = load_preformatted_data(JSONL_FILE)
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
    #DEBUGGING----------------
    # Then cap sizes for debugging
    train_df = train_df.iloc[:500].reset_index(drop=True)
    val_df = val_df.iloc[:100].reset_index(drop=True)
    #-------------------------


    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        model_max_length=MAX_LENGTH,
        use_fast=True
    )

    def tokenize_function(examples):
        return tokenizer(
            examples["input_text"],
            padding="max_length",
            #padding=False,
            truncation=True,
            max_length=MAX_LENGTH
        )

    print("Tokenizing data...")
    cols_to_remove = [c for c in train_dataset.column_names if c != "labels"]

    # Force fresh cache
    tokenized_train = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=cols_to_remove,
        load_from_cache_file=False
    )

    tokenized_val = val_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=cols_to_remove,
        load_from_cache_file=False
    )

    print("Initializing Model...")

    model = MT5ForRegression.from_pretrained(MODEL_NAME,
                                             #use_safetensors=True,
                                             torch_dtype="auto"
                                               )



    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = predictions.flatten()
        labels = labels.flatten()
        mse = ((predictions - labels) ** 2).mean()
        #pearson_corr, _ = pearsonr(predictions, labels)
        pearson_corr=0.5
        #kendall_corr, _ = kendalltau(predictions, labels)
        kendall_corr=0.5
        return {"mse": mse,
                "pearson": pearson_corr,
                "kendall": kendall_corr}

    # --- 6GB GPU SETTINGS ---
    training_args = TrainingArguments(
        output_dir="./mt5-custom-metric-output",
        learning_rate=1e-5,
        save_safetensors=False,


        #per_device_train_batch_size=8,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=1,
        per_device_eval_batch_size=8,
        fp16=False,
        num_train_epochs=6,
        weight_decay=1e-6,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="mse",
        greater_is_better=False,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=False
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

    # Add this right before trainer.train()
    check_gradients(trainer, model)
    trainer.train()

    trainer.save_model("./mt5-custom-metric-final")
    tokenizer.save_pretrained("./mt5-custom-metric-final")
    print("Done.")


if __name__ == "__main__":
    train()