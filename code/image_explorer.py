import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Loading the data
file1 = "../data/kmnist_train.csv" 
file2 = "../data/kmnist_test.csv"

data1 = pd.read_csv(file1)
data2 = pd.read_csv(file2)

# Image showing function
def plot_random_images_for_class(class_label, data1, data2, num_images=6):
    # random select for given label
    images1 = data1[data1['label'] == class_label].sample(n=num_images, random_state=42).iloc[:, 2:].values
    images2 = data2[data2['label'] == class_label].sample(n=num_images, random_state=42).iloc[:, 2:].values

    fig, axes = plt.subplots(2, num_images, figsize=(15, 5))
    fig.suptitle(f"Třída {class_label}", fontsize=16)

    # loops for showing the images
    for i, img in enumerate(images1):
        axes[0, i].imshow(img.reshape(28, 28), cmap="gray")
        axes[0, i].axis("off")
        axes[0, i].set_title(f"Soubor 1")

    for i, img in enumerate(images2):
        axes[1, i].imshow(img.reshape(28, 28), cmap="gray")
        axes[1, i].axis("off")
        axes[1, i].set_title(f"Soubor 2")

    plt.show()

# all the classes
classes = sorted(data1['label'].unique())

# calling the funtion
for class_label in classes:
    plot_random_images_for_class(class_label, data1, data2)