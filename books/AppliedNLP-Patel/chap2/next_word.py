import torch
import pandas as pd

from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import BertLMHeadModel, BertTokenizer


def predict_next_word(model, tokenizer, text):
    """ 
    Predict next word

    Parameters
    ----------


    Returns
    -------
    word_prediction: str
        next word predicted

    Examples
    --------
    
    """
    # use tokenizer to convert text to input vector
    inputs = tokenizer(text, return_tensors='pt') 

    # use model to predict logits from input vector
    with torch.no_grad():
        logits = model(**inputs).logits[:, -1, :]

    # predict next word
    pred_id = torch.argmax(logits).item()
    pred_word = tokenizer.decode(pred_id)

    return pred_word


### Step 1: instanciate pretrained tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')


text = "Run Forest"
word_pred = predict_next_word(model, tokenizer, text)
print(word_pred)


### MORE: Compare model

models_tokenizers = [
    (GPT2Tokenizer.from_pretrained('gpt2'), 
        GPT2LMHeadModel.from_pretrained('gpt2')), 
    (BertTokenizer.from_pretrained('bert-base-uncased'), 
        BertLMHeadModel.from_pretrained('bert-base-uncased')), 
]

next_words_dict = [
    ("Run Forest", "run"),
    ("With great power comes great", "responsability"),
]


def compare_model_prediction(models_tokenizers, texts): # TODO
    """ 
    Compare models/tokenizer performance

    Parameters
    ----------
    models_tokenizers: list of tuples
        list of (model, tokenizer)
    texts: list of tuples
        list of (text, next_word)

    Returns 
    -------
    df: pandas dataframe
        - model_name
        - text
        - next_word
        - prediction
    """
    pass



    


