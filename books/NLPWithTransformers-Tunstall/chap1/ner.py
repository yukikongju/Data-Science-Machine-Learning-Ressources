import pandas as pd

from transformers import pipeline

# text generated from https://app.inferkit.com/demo
texts = [
        "I went to the restaurant called Sona's that is set back in the woods near Provence.  It was a very nice place.  Kelton made friends quickly with the girl who waited on us, Aimee.  She seemed to really like him which made him"
]

tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
ner_tagger = pipeline("ner", model=model, tokenizer=tokenizer)
#  ner_tagger = pipeline('ner', aggregation_strategy='simple')
outputs = ner_tagger(texts)
df = pd.DataFrame(outputs)
print(df)


