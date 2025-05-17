import torch
import torch.nn as nn
import torch.nn.init as init
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from model import DEVICE, BASE_DIR, LLM_NAME, BASE_MODEL_NAME


device = torch.device(DEVICE)


class DebertaNHeads(nn.Module):
    def __init__(self, model_name=BASE_MODEL_NAME, num_aspects=7):
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

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, use_fast=True)

    model_llm = AutoModelForCausalLM.from_pretrained(
        LLM_NAME,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer_llm = AutoTokenizer.from_pretrained(LLM_NAME)

    return model, tokenizer, model_llm, tokenizer_llm
