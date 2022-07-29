import torch

from torch import nn, optim
from torchvision import datasets
from random import randint, randrange
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from torchvision import transforms
from torch.utils.data import DataLoader


### Step 1: generate dummy dataset: 100 colored image of 800x800 [size: 100x3x800x800]
num_img = 100
label_classes = ['doggo', 'not']
label_dict = {'doggo': 0, 'not':1}
imgs = torch.rand((num_img, 3, 800, 800))
labels = [ randrange(0, len(label_classes)) for _ in range(num_img) ]

### Step 2: perform preprocessing 

# images transformation: resize and normalize on all imgs to 3x32x32
transform = transforms.Compose([
    transforms.Resize((32, 32)), 
    transforms.Normalize([0.4915, 0.4823, 0.4468], [0.247, 0.2435, 0.2616]),
])

t_imgs = []
for img in imgs: 
    t_img = transform(img)
    t_imgs.append(t_img)

### Step 3: split into training and testing data

x_train, x_test, y_train, y_test = train_test_split(t_imgs, labels, 
        train_size = 0.3)

# Step 4: train the model
in_features = 32 * 32 * 3
out_features = len(label_classes)
n_epochs = 100

# initialize param
model = nn.Sequential(
    nn.Linear(in_features, 128), 
    nn.Tanh(),
    nn.Linear(128, out_features), 
    nn.LogSoftmax(),
)
optimizer = optim.SGD(model.parameters(), lr=1e-4)
loss_fn = nn.NLLLoss()

# REMARK: img: [1, 3072] ; label: 2 (int value)
def training(n_epochs, model, optimizer, loss_fn, imgs, labels):
    for epoch in range(1, n_epochs + 1):
        for img, label in zip(imgs, labels):
            # REMARK: img.shape: [3, 32, 32]   ; label = 'doggo'
            label_pred = model(img.reshape(1, -1))
            loss = loss_fn(label_pred, torch.tensor([label]))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch}, Loss {loss}")


def flatten(t):
    t = t.reshape(1, -1)
    t = t.squeeze()
    return t
    
training(n_epochs, model, optimizer, loss_fn, x_train, y_train)

# Step 5: test the model

def testing(model, x_test, y_test):
    correct, total = 0, len(x_test)

    with torch.no_grad():
        for img, label in zip(x_test, y_test):
            t_p = model(img.reshape(1, -1))
            _, label_pred = torch.max(t_p, dim=1)
            correct += int(label_pred == label)

    print(f"Total: {total}, Correct: {correct}")
    print(f"Accuracy {correct/total}")

testing(model, x_test, y_test)
    
    


