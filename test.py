from transformers import AutoModel, AutoTokenizer

model_name = "SpanBERT/spanbert-base-cased"  # Or "SpanBERT/spanbert-large-cased"
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)