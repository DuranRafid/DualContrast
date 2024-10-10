import numpy as np
import os
import matplotlib.pyplot as plt
import pickle

path = './starmen/starmen/output_random/images/'
images = os.listdir(path)

dataset = []
for image in images:
    sample = np.load(os.path.join(path, image))
    dataset.append(sample)
    
data_pickle = np.array(dataset)
with open('./data/starmen_train_dataset.pkl','wb') as f:
    pickle.dump(data_pickle, f)