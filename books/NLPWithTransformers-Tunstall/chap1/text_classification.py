import pandas as pd

from transformers import pipeline


texts = [
    "I am kind of happy with this order",
    "I am disgusted by this customer", 
    "I am working today",
    "I am neutral",
    "I need to cleanup this mess"
]

classifier = pipeline("text-classification")

outputs = classifier(texts)
df = pd.DataFrame(outputs)
print(df)

