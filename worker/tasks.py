import os

import gdown
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from celery import Celery
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from worker import LLM_NAME, PROMPT, SCORER_NAME

url = "https://drive.google.com/file/d/17SZ-XeQuKb8D4U40PhmexOBEgvQoNFZ0/view?usp=drive_link"
output = "models/best_second_model.pth"
if not os.path.exists(output):
    gdown.download(url=url, output=output, fuzzy=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

celery = Celery(
    "inference", broker=os.getenv("REDIS_BROKER"), backend=os.getenv("REDIS_BACKEND")
)


class DebertaNHeads(nn.Module):
    """
    DeBERTa-based neural network model with multiple regression heads.

    This model uses a pretrained DeBERTa transformer as a feature extractor,
    followed by multiple linear heads (one per aspect) to predict regression
    scores for different aspects of text evaluation.

    Args:
        model_name (str): Name or path of the pretrained model to load.
        num_aspects (int): Number of regression heads / aspects to predict.

    Attributes:
        deberta (AutoModel): Pretrained DeBERTa transformer model.
        regression_heads (nn.ModuleList): List of linear layers for regression.
        sigmoid (nn.Sigmoid): Sigmoid activation to bound outputs between 0 and 1.
        dropout (nn.Dropout): Dropout layer to reduce overfitting.
    """

    def __init__(self, model_name=SCORER_NAME, num_aspects=7):
        super().__init__()
        self.deberta = AutoModel.from_pretrained(model_name)
        hidden_size = self.deberta.config.hidden_size

        self.num_aspects = num_aspects

        self.regression_heads = nn.ModuleList(
            [nn.Linear(hidden_size, 1) for _ in range(num_aspects)]
        )
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.2)

        self._init_weights()

    def _init_weights(self):
        """
        Initialize weights of regression heads using Xavier uniform initialization,
        and biases to zeros.
        """
        for head in self.regression_heads:
            init.xavier_uniform_(head.weight)
            if head.bias is not None:
                init.zeros_(head.bias)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        """
        Forward pass of the model.

        Args:
            input_ids (torch.Tensor): Input token IDs.
            attention_mask (torch.Tensor, optional): Attention mask for inputs.
            token_type_ids (torch.Tensor, optional): Token type IDs (unused).

        Returns:
            torch.Tensor: Concatenated regression outputs for each aspect,
                          with shape (batch_size, num_aspects).
        """
        outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        regression_outputs = []

        for i in range(self.num_aspects):
            regression_outputs.append(
                self.sigmoid(self.regression_heads[i](pooled_output))
            )

        regression_outputs = torch.cat(regression_outputs, dim=1)

        return regression_outputs


def init_models():
    """
    Initialize and load the models and tokenizers required for evaluation.

    - Loads the tokenizer for the DeBERTa scorer model, adding special tokens.
    - Loads the DebertaNHeads model and its pretrained weights.
    - Loads the large language model (LLM) and tokenizer for generating feedback.

    Returns:
        tuple:
            - model (DebertaNHeads): The loaded DeBERTa regression model.
            - tokenizer (AutoTokenizer): Tokenizer for the scorer model.
            - model_llm (AutoModelForCausalLM): Pretrained causal language model.
            - tokenizer_llm (AutoTokenizer): Tokenizer for the LLM.
    """
    tokenizer = AutoTokenizer.from_pretrained(SCORER_NAME, use_fast=True)
    special_token = "\n\n"
    tokenizer.add_special_tokens({"additional_special_tokens": [special_token]})

    model = DebertaNHeads()
    model.deberta.resize_token_embeddings(len(tokenizer))
    with open(output, "rb") as f:
        model.load_state_dict(torch.load(f, weights_only=True))
    model.to(DEVICE)
    model.eval()

    model_llm_name = LLM_NAME

    model_llm = AutoModelForCausalLM.from_pretrained(
        model_llm_name, torch_dtype="auto", device_map="auto"
    )
    tokenizer_llm = AutoTokenizer.from_pretrained(model_llm_name)

    return model, tokenizer, model_llm, tokenizer_llm


model, tokenizer, model_llm, tokenizer_llm = init_models()


@celery.task(name="evaluate_qwk")
def predict_aspects(essay):
    """
    Predict quality aspects of an essay using the DeBERTa regression model.

    Args:
        essay (str): The input essay text to evaluate.

    Returns:
        dict: A dictionary with aspect names as keys and predicted scores as values.
              Scores are scaled to [1, 5] range with 0.5 increments.
    """
    inputs = tokenizer(
        essay,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
        max_length=512,
    )
    with torch.no_grad():
        input_ids = inputs["input_ids"].to(DEVICE)
        attention_mask = inputs["attention_mask"].to(DEVICE)
        outputs = model(input_ids, attention_mask)
    outputs = outputs.detach().cpu().numpy()[0]
    outputs = outputs * 4 + 1
    outputs = np.round(outputs * 2) / 2
    outputs = outputs.astype(float).tolist()
    return {
        "qwk": {
            "Overall": outputs[0],
            "Cohesion": outputs[1],
            "Syntax": outputs[2],
            "Vocabulary": outputs[3],
            "Phraseology": outputs[4],
            "Grammar": outputs[5],
            "Conventions": outputs[6],
        }
    }


@celery.task(name="evaluate_feedback")
def give_feedback(essay):
    """
    Generate feedback on an essay using a causal language model.

    Args:
        essay (str): The input essay text to provide feedback on.

    Returns:
        str: Generated feedback text from the language model.
    """
    prompt = PROMPT + essay
    messages = [
        {
            "role": "system",
            "content": "You are a helpful English language teaching assistant.",
        },
        {"role": "user", "content": prompt},
    ]
    text = tokenizer_llm.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    model_inputs = tokenizer_llm([text], return_tensors="pt").to(model_llm.device)

    generated_ids = model_llm.generate(**model_inputs, max_new_tokens=512)
    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer_llm.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response
