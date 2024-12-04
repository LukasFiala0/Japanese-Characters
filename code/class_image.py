import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


file = "../data/kmnist_test.csv"
data = pd.read_csv(file)

def plot_representative_images(data, classes, num_images=5, output_file="output.svg"):
    num_classes = len(classes)
    fig, axes = plt.subplots(num_classes, num_images, figsize=(num_images * 2, num_classes*2))  # Větší okno

    for row, class_label in enumerate(classes):
        images = data[data['label'] == class_label].sample(n=num_images, random_state=42).iloc[:, 2:].values

        for col, img in enumerate(images):
            ax = axes[row, col]
            ax.imshow(img.reshape(28, 28), cmap="gray")
            ax.axis("off")

        axes[row, 0].annotate(
            f"Class {class_label}",
            xy=(-0.5, 0.5),
            xycoords="axes fraction",
            ha="right",
            va="center",
            fontsize=15,
            weight="bold",
            color="black"
        )

    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.tight_layout(rect=[0.05, 0, 1, 0.90])
    
    plt.savefig(output_file, format="svg")
    plt.show()

classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

plot_representative_images(data, classes, output_file="../img/prezentation_img.svg")
