from pathlib import Path

BASE_DIR = Path(__file__).resolve(strict=True).parent
PROMPT = "You are given the following essay. Please provide feedback on the following aspects: Overall, Cohesion, Syntax, Vocabulary, Phraseology, Grammar and Conventions. Do not try to interpret or expand the meaning. If the writing lacks development or clarity, point it out directly.\n\n"
LLM_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
SCORER_NAME = "microsoft/deberta-v3-base"
