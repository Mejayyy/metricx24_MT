from operator import call, truediv
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


# ============================================================================
# Device Detection Utility
# ============================================================================
def get_device():
    """
    Detect and return the best available device for training.
    Priority: MPS (Apple Silicon) > CUDA (NVIDIA) > CPU
    
    Returns:
        Tuple of (device_string, device_object, device_name)
    """
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        # macOS with Apple Silicon
        device = torch.device("mps")
        device_name = "Metal Performance Shaders (MPS) - Apple Silicon"
        return "mps", device, device_name
    elif torch.cuda.is_available():
        # NVIDIA GPU
        device = torch.device("cuda")
        device_name = f"CUDA - NVIDIA GPU (Device 0: {torch.cuda.get_device_name(0)})"
        return "cuda", device, device_name
    else:
        # CPU fallback
        device = torch.device("cpu")
        device_name = "CPU"
        return "cpu", device, device_name


def clear_device_cache(device_type):
    """
    Clear device-specific cache.
    
    Args:
        device_type: String indicating device type ("mps", "cuda", or "cpu")
    """
    try:
        if device_type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif device_type == "mps" and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        # CPU doesn't need cache clearing
    except Exception as e:
        print(f"Warning: Could not clear {device_type} cache: {e}")


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
        # print("\nHEAD_MASK_MSG WARNING \n")
        warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
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

    # Create 1 step of dummy input for the decoder.
    batch_size = input_ids.size(0)
    decoder_input_ids = torch.LongTensor([0]).repeat(batch_size).reshape(-1, 1)

    # Move decoder_input_ids to the same device as the input
    decoder_input_ids = decoder_input_ids.to(input_ids.device)

    # Note: Model parallelism with MPS is not supported
    # MPS only supports single-GPU execution
    if self.model_parallel and input_ids.device.type == "cuda":
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

    # Set device for model parallelism (CUDA only, MPS doesn't support multi-GPU)
    if self.model_parallel and sequence_output.device.type == "cuda":
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
    MODEL_NAME = "google/mt5-small"
    JSONL_FILE = "all_data.jsonl"

    # Back to standard 512
    MAX_LENGTH = 512

    # Detect and log device
    device_type, device, device_name = get_device()
    
    # Comprehensive device logging
    print(f"\n{'='*80}")
    print(f"PyTorch Training Configuration")
    print(f"{'='*80}")
    print(f"Device Type: {device_type.upper()}")
    print(f"Device: {device_name}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Model: {MODEL_NAME}")
    print(f"Max Sequence Length: {MAX_LENGTH}")
    
    # Device-specific information
    if device_type == "mps":
        print(f"\nMPS Configuration:")
        print(f"  - Metal Performance Shaders Available: {torch.backends.mps.is_available()}")
        print(f"  - MPS Built: {torch.backends.mps.is_built()}")
        print(f"  - Note: Batch sizes and num_workers adjusted for MPS optimization")
    elif device_type == "cuda":
        print(f"\nCUDA Configuration:")
        print(f"  - CUDA Available: {torch.cuda.is_available()}")
        print(f"  - GPU Count: {torch.cuda.device_count()}")
        print(f"  - Compute Capability: {torch.cuda.get_device_capability(0)}")
        print(f"  - Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print(f"\nCPU Configuration:")
        print(f"  - Running on CPU (this will be slow for training)")
        print(f"  - Consider using a GPU for better performance")
    
    print(f"{'='*80}\n")

    # Clear device cache
    clear_device_cache(device_type)

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

    model = MT5ForRegression.from_pretrained(MODEL_NAME, torch_dtype="auto")
    model = model.to(device)

    # Performance options - device-specific
    model.gradient_checkpointing_enable()
    
    # TF32 optimizations are CUDA-only; skip for MPS/CPU
    if device_type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("Enabled CUDA TF32 optimizations")
    elif device_type == "mps":
        print("Using MPS (Apple Silicon) - TF32 optimizations not available")


    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = predictions.flatten()
        labels = labels.flatten()
        mse = ((predictions - labels) ** 2).mean()
        rmse = np.sqrt(mse)
        return {"mse": mse, "rmse": rmse}

  #   training_args = TrainingArguments(
  #     output_dir="./mt5-custom-metric-output",
  #     learning_rate=2e-4,
  #     save_safetensors=False,

  #     warmup_steps=2000

  #     per_device_train_batch_size=32,
  #     gradient_accumulation_steps=1,
  #     per_device_eval_batch_size=64,

  #     # RTX 5090: prefer bf16 if available; fallback to fp16
  #     bf16=torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8,
  #     fp16=torch.cuda.is_available() and not (torch.cuda.get_device_capability(0)[0] >= 8),

  #     num_train_epochs=None,
  #     max_steps=20000,
  #     weight_decay=0.0,

  #     eval_strategy="steps",
  #     eval_steps=500,

  #     save_strategy="steps",
  #     save_steps=1000,
  #     save_total_limit=5,

  #     logging_strategy="steps",
  #     logging_steps=100,

  #     load_best_model_at_end=True,
  #     metric_for_best_model="mse",
  #     greater_is_better=False,

  #     remove_unused_columns=False,

  #     dataloader_num_workers=8,
  #     dataloader_pin_memory=True,
  # )

    # Determine mixed precision settings based on device
    if device_type == "mps":
        # MPS prefers bf16 and doesn't support pin_memory
        use_bf16 = True
        use_fp16 = False
        use_pin_memory = False
        num_workers = 0  # MPS works better with 0 workers
    elif device_type == "cuda":
        # CUDA: bf16 for compute capability >= 8, fp16 otherwise
        compute_capability = torch.cuda.get_device_capability(0)[0]
        use_bf16 = compute_capability >= 8
        use_fp16 = not use_bf16
        use_pin_memory = True
        num_workers = 4
    else:
        # CPU: no mixed precision
        use_bf16 = False
        use_fp16 = False
        use_pin_memory = False
        num_workers = 0

    training_args = TrainingArguments(
      output_dir="./mt5-custom-metric-output",
      learning_rate=1e-4,
      save_safetensors=False,

      max_steps=2500,
      warmup_steps=200,

      per_device_train_batch_size=32,
      gradient_accumulation_steps=1,
      per_device_eval_batch_size=64,

      bf16=use_bf16,
      fp16=use_fp16,

      weight_decay=0.0,
      max_grad_norm=1.0,

      eval_strategy="steps",
      eval_steps=100,

      save_strategy="steps",
      save_steps=100,
      save_total_limit=5,

      logging_strategy="steps",
      logging_steps=50,

      load_best_model_at_end=True,
      metric_for_best_model="mse",
      greater_is_better=False,

      remove_unused_columns=False,
      dataloader_num_workers=num_workers,
      dataloader_pin_memory=use_pin_memory,
    )

    callbacks=[EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=1e-4)]

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

    trainer.save_model("./mt5-custom-metric-final")
    tokenizer.save_pretrained("./mt5-custom-metric-final")
    print("Done.")


if __name__ == "__main__":
    train()