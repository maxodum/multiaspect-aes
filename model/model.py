import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
from celery import Celery, signals
import os


# celery = Celery(
#     "inference",
#     broker="redis://localhost:6379/0",
#     backend="redis://localhost:6379/0"
# )

device = torch.device('cuda')

BASE_DIR = Path(__file__).resolve(strict=True).parent
PROMPT = "You are given the following essay. Please provide feedback on the following aspects: Overall, Cohesion, Syntax, Vocabulary, Phraseology, Grammar and Conventions. Do not try to interpret or expand the meaning. If the writing lacks development or clarity, point it out directly.\n\n"


class DebertaNHeads(nn.Module):
    def __init__(self, model_name="microsoft/deberta-v3-base", num_aspects=7):
        super().__init__()
        self.deberta = AutoModel.from_pretrained(model_name)
        hidden_size = self.deberta.config.hidden_size

        self.num_aspects = num_aspects
        
        self.regression_heads = nn.ModuleList([
            nn.Linear(hidden_size, 1) for _ in range(num_aspects)
        ])
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.2)
        
        self._init_weights()

    def _init_weights(self):
        for head in self.regression_heads:
            init.xavier_uniform_(head.weight)
            if head.bias is not None:
                init.zeros_(head.bias)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        regression_outputs = []

        for i in range(self.num_aspects):
            regression_outputs.append(self.sigmoid(self.regression_heads[i](pooled_output)))

        regression_outputs = torch.cat(regression_outputs, dim=1)

        return regression_outputs
    
def init_models():
    model = DebertaNHeads()
    with open(f'{BASE_DIR}/best_second_model.pth', 'rb') as f:
        model.load_state_dict(torch.load(f, weights_only=True))
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base", use_fast=True)

    model_llm_name = "Qwen/Qwen2.5-1.5B-Instruct"

    model_llm = AutoModelForCausalLM.from_pretrained(
        model_llm_name,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer_llm = AutoTokenizer.from_pretrained(model_llm_name)

    return model, tokenizer, model_llm, tokenizer_llm


#model.eval()


# @celery.task(name='evaluate_qwk')
# def predict_aspects(essay):
#     inputs = tokenizer(essay,
#                        padding="max_length",
#                        truncation=True,
#                        return_tensors="pt",
#                        max_length=512)
#     with torch.no_grad():
#         input_ids = inputs["input_ids"].to(device)
#         attention_mask = inputs["attention_mask"].to(device)
#         outputs = model(input_ids, attention_mask)
#     outputs = outputs.detach().cpu().numpy()[0]
#     outputs = outputs * 4 + 1
#     outputs = np.round(outputs * 2) / 2
#     outputs = outputs.astype(float).tolist()
#     return {'qwk':{'Overall': outputs[0],
#             'Cohesion': outputs[1],
#             'Syntax': outputs[2],
#             'Vocabulary': outputs[3],
#             'Phraseology': outputs[4],
#             'Grammar': outputs[5],
#             'Conventions': outputs[6]}}


# @celery.task(name='evaluate_feedback')
# def give_feedback(essay):
#     prompt = PROMPT + essay
#     messages = [
#     {"role": "system", "content": "You are a helpful English language teaching assistant."},
#     {"role": "user", "content": prompt}
#     ]
#     text = tokenizer_llm.apply_chat_template(
#         messages,
#         tokenize=False,
#         add_generation_prompt=True
#     )
#     model_inputs = tokenizer_llm([text], return_tensors="pt").to(model_llm.device)

#     generated_ids = model_llm.generate(**model_inputs,
#                                    max_new_tokens=512
#                                    )
#     generated_ids = [
#         output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
#         ]

#     response = tokenizer_llm.batch_decode(generated_ids, skip_special_tokens=True)[0]
#     return response