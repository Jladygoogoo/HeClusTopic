import torch
from transformers import BertModel, BertTokenizer


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert = BertModel.from_pretrained("bert-base-uncased")

max_len = 512
texts = [
    "Masked Language Modeling works by inserting a mask token at the desired position where you want to predict the best candidate word that would go in that position. You can simply insert the mask token by concatenating it at the desired position in your input like I did above. ",
    "Masked Language Modeling works by inserting a mask token at the desired position where you want to predict the best candidate word that would go in that position. You can simply insert the mask token by concatenating it at the desired position in your input like I did above. ",
    "Next Sentence prediction is the task of predicting how good a sentence is a next sentence for a given sentence. In this case, “The child came home from school.” is the given sentence and we are trying to predict whether “He played soccer after school.” is the next sentence. "
    ]

device = torch.device("cuda:0")

encoded_dict = tokenizer.batch_encode_plus(texts, max_length=max_len, padding='max_length', add_special_tokens=True, return_attention_mask=True, truncation=True, return_tensors='pt')
input_ids = encoded_dict['input_ids'].to(device)
attention_masks = encoded_dict['attention_mask'].to(device)
bert.to(device)

bert_outputs = bert(input_ids, attention_mask=attention_masks)
