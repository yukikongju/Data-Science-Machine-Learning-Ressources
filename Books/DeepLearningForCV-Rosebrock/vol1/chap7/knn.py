import argparse
import logging

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from pyimagesearch.preprocessing import SimplePreprocessor
from pyimagesearch.datasets import SimpleDatasetLoader


""" Runing the script: python3 knn.py --dataset ../datasets/animals """

# Step 1: Construct parser

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", required=True, help="path to datasets")
parser.add_argument("-k", "--neighbors", type=int, default=1, 
        help="num of neighbors to consider for the classification")
parser.add_argument("-j", '--jobs', default=-1, help="num of jobs to use to compute KNN distance (-1 uses all all available cores)")

args = vars(parser.parse_args())


# Step 2: Preprocess images from dataset path

imagePaths = args['dataset']


preprocessor = SimplePreprocessor(32, 32)
dataloader = SimpleDatasetLoader(preprocessorheight)
data, labels = dataloader.load(imagePaths)

#  Step 3: Encode the label as integer
encoder = LabelEncoder()
labels = encoder.fit_transform(labels)

# Step 4: split data into train test 
x_train, x_test, y_train, y_test = train_test_split(data, labels, 
        test_size=0.25, random_state=420)

# Step 5: train model knn
model = KNeighborsClassifier(n_neighbors=args['k'], n_jobs=args['jobs'])
model.fit(x_train, y_train)

# Step 6: test model
predictions = model.predict(x_test)

# Step 7: get accuracy score
results = classification_report(y_test, predictions)
print(results)


