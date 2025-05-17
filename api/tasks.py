from celery import Celery
import torch
from model.model import init_models
from pathlib import Path
import numpy as np


celery = Celery(
    "inference",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/0"
)

BASE_DIR = Path(__file__).resolve(strict=True).parent
PROMPT = "You are given the following essay. Please provide feedback on the following aspects: Overall, Cohesion, Syntax, Vocabulary, Phraseology, Grammar and Conventions. Do not try to interpret or expand the meaning. If the writing lacks development or clarity, point it out directly.\n\n"
device = torch.device('cuda')
model, tokenizer, model_llm, tokenizer_llm = init_models()

@celery.task(name='evaluate_qwk')
def predict_aspects(essay):
    inputs = tokenizer(essay,
                       padding="max_length",
                       truncation=True,
                       return_tensors="pt",
                       max_length=512)
    with torch.no_grad():
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        outputs = model(input_ids, attention_mask)
    outputs = outputs.detach().cpu().numpy()[0]
    outputs = outputs * 4 + 1
    outputs = np.round(outputs * 2) / 2
    outputs = outputs.astype(float).tolist()
    return {'qwk':{'Overall': outputs[0],
            'Cohesion': outputs[1],
            'Syntax': outputs[2],
            'Vocabulary': outputs[3],
            'Phraseology': outputs[4],
            'Grammar': outputs[5],
            'Conventions': outputs[6]}}


@celery.task(name='evaluate_feedback')
def give_feedback(essay):
    prompt = PROMPT + essay
    messages = [
    {"role": "system", "content": "You are a helpful English language teaching assistant."},
    {"role": "user", "content": prompt}
    ]
    text = tokenizer_llm.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer_llm([text], return_tensors="pt").to(model_llm.device)

    generated_ids = model_llm.generate(**model_inputs,
                                   max_new_tokens=512
                                   )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

    response = tokenizer_llm.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response