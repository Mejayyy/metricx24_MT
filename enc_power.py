import os
import argparse
import json
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import pandas as pd
import torch
from torch import nn
import copy
import dataclasses

from typing import Optional, Tuple, Union
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr, kendalltau
from mt_metrics_eval import stats as mt_stats
from transformers import (
    AutoTokenizer,
    MT5PreTrainedModel,
    MT5Config,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from transformers.models.mt5.modeling_mt5 import MT5Stack
import transformers.modeling_outputs
from datasets import Dataset
import transformers
import warnings
import numpy as np

from transformers import Adafactor
from transformers.optimization import get_inverse_sqrt_schedule

BaseModelOutput = transformers.modeling_outputs.BaseModelOutput
ModelOutput = transformers.modeling_outputs.ModelOutput
MT5Config = transformers.models.mt5.modeling_mt5.MT5Config
MT5PreTrainedModel = transformers.models.mt5.modeling_mt5.MT5PreTrainedModel
MT5Stack = transformers.models.mt5.modeling_mt5.MT5Stack

__HEAD_MASK_WARNING_MSG = (
    transformers.models.mt5.modeling_mt5.__HEAD_MASK_WARNING_MSG  # pylint: disable=protected-access
    # "Warning HEAD_MASK_WARNING_MSG"
)

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
        # Projects pooled encoder output -> single scalar score
        self.regression_head = nn.Linear(config.d_model, 1)

        # Optional: start very small to avoid huge initial outputs
        # nn.init.normal_(self.regression_head.weight, mean=0.0, std=1e-3)
        # nn.init.zeros_(self.regression_head.bias)
        """
        self.lm_head = nn.Sequential(
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

        # Safety: avoid propagating NaNs/Infs from mixed precision / bad batches
        pooled_output = torch.nan_to_num(pooled_output, nan=0.0, posinf=1e4, neginf=-1e4)

        # 3. Predict Score
        logits = self.regression_head(pooled_output)  # [Batch, 1]
        predictions = logits.squeeze(-1)  # [Batch]
        predictions = torch.nan_to_num(predictions, nan=0.0, posinf=1e4, neginf=-1e4)
        #predictions = 25 * logits.squeeze(-1)
        # 4. Scale Sigmoid Output [0, 1] -> [0, 25]
        # This matches your previous logic for MQM scores
        #predictions = predictions * 25.0

        loss = None
        if labels is not None:
            loss_fct = nn.MSELoss()
            # Move labels to correct device and compute loss in fp32 for stability
            labels = labels.to(predictions.device)
            pred_fp32 = predictions.float()
            lab_fp32 = labels.view(-1).float()
            # Safety: strip NaNs/Infs if they appear
            pred_fp32 = torch.nan_to_num(pred_fp32, nan=0.0, posinf=1e4, neginf=-1e4)
            lab_fp32 = torch.nan_to_num(lab_fp32, nan=0.0, posinf=1e4, neginf=-1e4)
            loss = loss_fct(pred_fp32, lab_fp32)

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


class MetricXTrainer(Trainer):
    def create_optimizer(self):
        if self.optimizer is None:
            # MetricX-style Adafactor: external LR, no relative_step
            self.optimizer = Adafactor(
                self.model.parameters(),
                lr=self.args.learning_rate,
                scale_parameter=False,
                relative_step=False,
                warmup_init=False,
            )
        return self.optimizer

    def create_scheduler(self, num_training_steps: int, optimizer=None):
        if self.lr_scheduler is None:
            opt = optimizer if optimizer is not None else self.optimizer
            # inverse sqrt schedule with linear warmup
            self.lr_scheduler = get_inverse_sqrt_schedule(
                opt,
                num_warmup_steps=self.args.warmup_steps,
                # optional: timescale defaults in HF; you can pass it if you want
                # timescale=10000,
            )
        return self.lr_scheduler

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./mt5-custom-metric-small-output",
        help="Base directory for Trainer checkpoints/logs; final model is saved to <output_dir>/final",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="all_data.jsonl",
        help="Path to the input JSONL file",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="google/mt5-small",
        help="Base model name or path",
    )
    args_cli = parser.parse_args()

    final_model_dir = os.path.join(args_cli.output_dir, "final")

    MODEL_NAME = args_cli.model_name
    JSONL_FILE = args_cli.data_path

    # Back to standard 512
    MAX_LENGTH = 512

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    df = load_preformatted_data(JSONL_FILE)
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
    #DEBUGGING----------------
    # Then cap sizes for debugging
    # train_df = train_df.iloc[:500].reset_index(drop=True)
    # val_df = val_df.iloc[:100].reset_index(drop=True)
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

    model = MT5EncoderForRegression.from_pretrained(
        MODEL_NAME,
        torch_dtype="auto",
        ignore_mismatched_sizes=True,
    )

    # --- Force encoder-only behavior (extra safety) ---
    # If a decoder exists (e.g., if you swap to an encoder-decoder class later), drop it.
    if hasattr(model, "decoder"):
        model.decoder = None

    # Make sure HF utilities/Trainer don't assume an encoder-decoder model.
    model.config.is_encoder_decoder = False
    model.config.is_decoder = False
    model.config.use_cache = False
    # -----------------------------------------------

    # Performance options (helpful on RTX 5090)
    model.gradient_checkpointing_enable()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


    def compute_metrics(eval_pred):
      predictions, labels = eval_pred
      preds = np.asarray(predictions).reshape(-1)
      labs = np.asarray(labels).reshape(-1)

      # Filter out NaNs/Infs to avoid scipy/numpy crashes
      finite = np.isfinite(preds) & np.isfinite(labs)
      preds = preds[finite]
      labs = labs[finite]

      if preds.size == 0 or labs.size == 0:
        return {
            "mse": float("nan"),
            "rmse": float("nan"),
            "pearson": 0.0,
            "kendall_tau": 0.0,
            "pairwise_acc": 0.0,
            "num_finite": 0,
        }

      # MSE / RMSE
      mse = float(np.mean((preds - labs) ** 2))
      rmse = float(np.sqrt(mse))

      # Pearson / Kendall (need at least 2 points)
      if preds.size > 1:
        pearson = float(pearsonr(preds, labs)[0])
        kendall = float(kendalltau(preds, labs).correlation)
      else:
        pearson = 0.0
        kendall = 0.0

      # Pairwise accuracy (MetricX/MTME-style). Negate so higher-is-better.
      agree, num_pairs = mt_stats.Agreement(-labs, -preds)
      pairwise_acc = float(agree / num_pairs) if num_pairs > 0 else 0.0

      return {
          "mse": mse,
          "rmse": rmse,
          "pearson": pearson,
          "kendall_tau": kendall,
          "pairwise_acc": pairwise_acc,
          "num_finite": int(preds.size),
      }

    training_args = TrainingArguments(
      output_dir=args_cli.output_dir,
      learning_rate=1e-4,
      save_safetensors=False,

      max_steps=3200,
      warmup_steps=250,

      per_device_train_batch_size=32,
      gradient_accumulation_steps=1,
      per_device_eval_batch_size=64,

      bf16=torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8,
      fp16=torch.cuda.is_available() and not (torch.cuda.get_device_capability(0)[0] >= 8),

      weight_decay=0.0,
      max_grad_norm=1.0,

      eval_strategy="steps",
      eval_steps=100,

      save_strategy="steps",
      save_steps=100,
      save_total_limit=3,

      logging_strategy="steps",
      logging_steps=50,

      load_best_model_at_end=True,
      metric_for_best_model="mse",
      greater_is_better=False,

      remove_unused_columns=False,
      dataloader_num_workers=8,
      dataloader_pin_memory=True,
    )

    callbacks=[EarlyStoppingCallback(early_stopping_patience=6, early_stopping_threshold=1e-4)]

    trainer = MetricXTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=callbacks
    )

    print("Starting Training...")

    # Add this right before trainer.train()
    # check_gradients(trainer, model)
    trainer.train()

    # At this point, Trainer has loaded the best checkpoint into trainer.model
    # (because load_best_model_at_end=True).
    final_metrics = trainer.evaluate(eval_dataset=tokenized_val)

    os.makedirs(final_model_dir, exist_ok=True)
    trainer.save_model(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)

    # Save final evaluation metrics for the best model
    metrics_path = os.path.join(final_model_dir, "final_eval_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(final_metrics, f, indent=2)

    print(f"Done. Final model saved to: {final_model_dir}")
    print(f"Final eval metrics saved to: {metrics_path}")
    print(final_metrics)


if __name__ == "__main__":
    train()