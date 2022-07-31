import torch

from transformers import GPT2LMHeadModel, GPT2Tokenizer


### Step 1: instanciate pretrained tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')


### Step 2: use tokenizer to convert text to inputs vector 
text = "Run Forest"
inputs = tokenizer(text, return_tensors='pt') 
#  text_vect = tokenizer.encode(text)
# inputs =  {'input_ids': tensor([[10987,  9115]]), 'attention_mask': tensor([[1, 1]])}

### Step 3: use model to predict logits using inputs vector
with torch.no_grad():
    logits = model(**inputs).logits[:, -1, :]

print(logits)
print(logits.shape) # torch.Size([1, 50257])

### Step 4: get predicted word from logits
pred_id = torch.argmax(logits).item()
pred_word = tokenizer.decode(pred_id)

print(pred_word)

