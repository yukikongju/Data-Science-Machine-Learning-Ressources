import pandas as pd

from transformers import pipeline

# text generated from https://app.inferkit.com/demo
texts = [
        "I went to the restaurant called Sona's that is set back in the woods near Provence.  It was a very nice place.  Kelton made friends quickly with the girl who waited on us, Aimee.  She seemed to really like him which made him"
]

ner_tagger = pipeline('ner', aggregation_strategy='simple')
outputs = ner_tagger(texts)
df = pd.DataFrame(texts)
print(df)


