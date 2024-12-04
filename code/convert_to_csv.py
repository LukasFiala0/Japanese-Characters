import numpy as np
import pandas as pd

# loading the data
train_images = np.load("kmnist-train-imgs.npz")['arr_0']
train_labels = np.load("kmnist-train-labels.npz")['arr_0']
test_images = np.load("kmnist-test-imgs.npz")['arr_0']
test_labels = np.load("kmnist-test-labels.npz")['arr_0']

def save_to_csv(images, labels, output_file):
    # Reshape to flat "array"
    images_flattened = images.reshape(images.shape[0], -1)
    
    # DataFrame
    data = pd.DataFrame(images_flattened, columns=[f"pixel{i+1}" for i in range(images_flattened.shape[1])])
    data.insert(0, "label", labels)  # add label collumn
    data.insert(0, "id", range(1, len(labels) + 1))  # add id collumn
    
    data.to_csv(output_file, index=False)

save_to_csv(train_images, train_labels, "../data/kmnist_train.csv")
save_to_csv(test_images, test_labels, "../data/kmnist_test.csv")
