import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from model import load_model

def generate_article(model, tokenizer, prompt, device):
    model.eval()
    model.to(device)

    inputs = tokenizer("生成文章: " + prompt, return_tensors="pt").to(device)
    outputs = model.generate(inputs["input_ids"], max_length=1024, num_return_sequences=1)

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

def main():
    saved_model_path = "../saved_models/t5_article_gen"
    model_name = "t5-small"
    tokenizer_name = "t5-small"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, tokenizer = load_model(saved_model_path, tokenizer_name)

    prompt = "示例文章标题"
    generated_article = generate_article(model, tokenizer, prompt, device)

    print("Generated Article:")
    print(generated_article)

if __name__ == "__main__":
    main()
