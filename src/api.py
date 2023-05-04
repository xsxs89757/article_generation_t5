from fastapi import FastAPI
from pydantic import BaseModel
from generate import generate_article
from model import load_model
import torch

app = FastAPI()

class Prompt(BaseModel):
    title: str

saved_model_path = "../saved_models/t5_article_gen"
tokenizer_name = "t5-small"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, tokenizer = load_model(saved_model_path, tokenizer_name)

@app.post("/generate_article")
def generate_article_api(prompt: Prompt):
    generated_text = generate_article(model, tokenizer, prompt.title, device)
    return {"generated_text": generated_text}
