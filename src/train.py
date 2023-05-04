import torch
from torch.utils.data import DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer, AdamW
from dataset import ArticleDataset
from model import load_model

def train_model(model, tokenizer, train_dataset, epochs, batch_size, device, saved_model_path):
    model.train()
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=5e-5)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        for batch in train_loader:
            optimizer.zero_grad()
            inputs = tokenizer(batch["source"], padding=True, return_tensors="pt", truncation=True).to(device)
            labels = tokenizer(batch["target"], padding=True, return_tensors="pt", truncation=True).to(device)
            outputs = model(**inputs, labels=labels["input_ids"])
            loss = outputs.loss
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss.item()}")

    model.save_pretrained(saved_model_path)

def main():
    train_data_file = "../data/train.csv"
    model_name = "t5-small"
    tokenizer_name = "t5-small"
    saved_model_path = "../saved_models/t5_article_gen"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = ArticleDataset(train_data_file)
    model, tokenizer = load_model(model_name, tokenizer_name)

    train_model(model, tokenizer, train_dataset, epochs=2, batch_size=8, device=device, saved_model_path=saved_model_path)

if __name__ == "__main__":
    main()
