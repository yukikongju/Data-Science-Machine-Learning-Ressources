import cv2
import os
import numpy as np
import logging

class SimpleDatasetLoader:

    def __init__(self, preprocessors=None):
        self.preprocessors = preprocessors

        if self.preprocessors is None:
            self.preprocessors = []
        
    def load(self, imagePaths, verbose=1):
        """ 
        Initialize list of features and labels
        """
        data = []
        labels = []

        for i, imagePath in enumerate(imagePaths):
            #  path: path/to/data/{class}/{image}.jpg
            image = cv2.imread(imagePath)
            label = imagePath.split(os.path.sep)[-2]

            # preprocess the data if preprocessor defined
            if self.preprocessors is not None:
                for p in self.preprocessors:
                    image = p.preprocess(image)

            # show update every 'verbose' image
            if verbose > 0 and i > 0 and (i+1) % verbose == 0:
                logging.info("preprocessed {}/{}").format(i+1, len(imagePaths))

        return (np.array(data), np.array(labels))

        


        

