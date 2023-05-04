from transformers import T5ForConditionalGeneration, T5Tokenizer

def load_model(model_name, tokenizer_name):
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    tokenizer = T5Tokenizer.from_pretrained(tokenizer_name)
    return model, tokenizer
