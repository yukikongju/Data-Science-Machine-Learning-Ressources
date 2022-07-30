import torch
import pandas as pd
import matplotlib.pyplot as plt
import sklearn

from random import randrange
from torch import optim, nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from model import AdaptNet, BatchNormNet, CNN, ConvNet, DepthNet, DropoutNet


### Step 0: Create dummy data

num_img = 100
label_classes = ['yes', 'no']
num_classes = len(label_classes)
labels = [randrange(0, num_classes) for _ in range(num_img)]
imgs = [torch.rand((1,3,32,32)) for _ in range(num_img)]

x_train, x_test, y_train, y_test = train_test_split(imgs, labels, test_size=0.3)

### Step 1: create model, optimizer, loss function and set hyperparameters

n_epochs = 100
learning_rate = 1e-4
loss_fn = nn.CrossEntropyLoss()

models = [
    ('adaptnet', AdaptNet(3, num_classes)),
    ('batchnormnet', BatchNormNet(3, num_classes, n_channels=32)),
    ('cnn', CNN(3, num_classes)),
    ('depthnet', DepthNet(3, num_classes)),
    ('dropoutnet', DropoutNet(3, num_classes)),
]

### Step 2: train the models

def training(n_epochs, model, loss_fn, optimizer, x_train, y_train):
    """ 
    Training a single model with its respective loss and optimizer function

    Returns
    -------
    history : list
        loss history
    """
    history = []
    for epoch in range(n_epochs):
        for img, label in zip(x_train, y_train):
            
            out = model(img)
            loss = loss_fn(out, torch.tensor([label]))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss {loss}")

        history.append(loss.tolist())

    return history
    

def testing(model, scoring_function, x_test, y_test):
    """ 
    Get scoring metrics when testing model on unseen data 

    Parameters
    ----------
    scoring_function: sklearn.metrics function
    """
    preds = []

    with torch.no_grad():
        for img, label in zip(x_test, y_test):
            out = model(img)
            _, pred = torch.max(out, dim=1)

            preds.append(pred.tolist()[0])

    score = scoring_function(preds, y_test)
    print(score)

    return score

#  model = AdaptNet(3, num_classes)
#  optimizer = optim.SGD(model.parameters(), lr=learning_rate)
#  history = training(n_epochs, model, loss_fn, optimizer, x_train, y_train)
#  scoring_function = sklearn.metrics.accuracy_score
#  scores = testing(model, scoring_function, x_test, y_test)

### Step 3: Compare the models

def compare_models_training(n_epochs, models, loss_fn, x_train, y_train):
    """ 
    Get loss history for models

    Parameters
    ----------
    models: list of tuples
        list of models

    Returns
    -------
    df: pandas dataframe
        loss per epoch for all models

    """
    df = pd.DataFrame()
    df['epoch'] = [epoch for epoch in range(1, n_epochs+1)]
    
    for name, model in models: 
        optimizer = optim.SGD(model.parameters(), lr=1e-4)
        df[name] = training(n_epochs, model, loss_fn, optimizer, x_train, y_train)

    return df


def compare_models_testing(models, scoring_function, x_test, y_test):
    """ 
    Get scoring for trained models

    Parameters
    ----------
    scoring_function: sklearn.metrics function

    Returns
    -------
    df: 
        scores
    """
    df = pd.DataFrame()
    for name, model in models: 
        score = testing(model, scoring_function, x_test, y_test)
        df[name] = [score]

    return df
    

history = compare_models_training(n_epochs, models, loss_fn, x_train, y_train)
print(history)

scoring_function = sklearn.metrics.accuracy_score
scores = compare_models_testing(models, scoring_function, x_test, y_test)
print(scores)

### TODO: Step 4: Plot models performance


