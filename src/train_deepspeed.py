import torch
import pandas as pd
import argparse
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.utils.data import DataLoader
from src.dataset import ArticleDataset
import deepspeed

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--deepspeed_config', type=str, default='ds_config.json', help='Path to DeepSpeed config file')
args = parser.parse_args()

# Load data
data = pd.read_csv("data/train.csv")
train_dataset = ArticleDataset(data)
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# Initialize the model and tokenizer
tokenizer = T5Tokenizer.from_pretrained("t5-base")
model = T5ForConditionalGeneration.from_pretrained("t5-base")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Initialize DeepSpeed
model_engine, optimizer, _, _ = deepspeed.initialize(args=args, model=model, model_parameters=model.parameters())

# Training loop
for epoch in range(2):
    for batch in train_dataloader:
        input_ids, attention_mask, labels = batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model_engine(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        model_engine.backward(loss)
        model_engine.step()

# Save the fine-tuned model
model_engine.save_checkpoint("saved_models/t5_article_gen_deepspeed")
