
#from Encoder_only_1 import MT5EncoderForRegression

import torch
from torch import nn
import copy
from transformers import (
    AutoTokenizer,
    MT5PreTrainedModel,
    MT5Config,
)

from transformers.models.mt5.modeling_mt5 import MT5Stack
from transformers.modeling_outputs import BaseModelOutput, ModelOutput
from typing import Optional, Tuple, Union
import dataclasses

"""
# --- 1. DEFINE THE CLASS EXACTLY AS TRAINED ---
@dataclasses.dataclass
class MT5ForRegressionOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    predictions: torch.FloatTensor = None


class MT5EncoderForRegression(MT5PreTrainedModel):
    def __init__(self, config: MT5Config):
        super().__init__(config)
        self.model_dim = config.d_model

        # 1. Shared Embeddings
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        # 2. Encoder Only
        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = MT5Stack(encoder_config, self.shared)

        # 3. NO DECODER

        # 4. Regression Head
        # IMPORTANT: This must match what you trained with!
        # If you used the simple Linear head in your last run:
        self.lm_head = nn.Linear(config.d_model, 1)

        # If you used the Sequential/Sigmoid head, uncomment this instead:
        # self.lm_head = nn.Sequential(
        #     nn.Linear(config.d_model, config.d_model),
        #     nn.Tanh(),
        #     nn.Linear(config.d_model, 1),
        #     nn.Sigmoid()
        # )

        self.post_init()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            labels=None,
            **kwargs
    ):
        # 1. Run Encoder
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        hidden_states = outputs.last_hidden_state

        # 2. Mean Pooling
        if attention_mask is not None:
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            pooled_output = sum_embeddings / sum_mask
        else:
            pooled_output = torch.mean(hidden_states, dim=1)

        # 3. Predict
        logits = self.lm_head(pooled_output)
        predictions = logits.squeeze(-1)

        # Scale output if you used Sigmoid during training
        # predictions = predictions * 25.0

        return MT5ForRegressionOutput(predictions=predictions)

"""
# --- 2. LOAD THE MODEL ---
def predict():
    MODEL_PATH = "./mt5-encoder-metric-final"

    print(f"Loading model from {MODEL_PATH}...")

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    # Load Model (Strict=False helps if there are minor config discrepancies)
    model = MT5EncoderForRegression.from_pretrained(MODEL_PATH)

    model.eval()
    if torch.cuda.is_available():
        model.cuda()

    # --- 3. RUN PREDICTION ---
    print("\n--- Testing Inference ---")

    # Example Sentence Pair (Source + Translation)
    input_text = "This is a perfect translation."

    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=256,
        padding="max_length"
    )

    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        score = outputs.predictions.item()

    print(f"Input: {input_text}")
    print(f"Predicted Score: {score:.4f}")


if __name__ == "__main__":
    predict()