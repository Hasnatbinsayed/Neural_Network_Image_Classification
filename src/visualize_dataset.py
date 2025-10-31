import matplotlib.pyplot as plt
import tensorflow as tf
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# CIFAR-10 class labels
CIFAR10_LABELS = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Function to plot a grid of images
def plot_sample_images(x, y, labels, num_rows=3, num_cols=4):
    plt.figure(figsize=(num_cols*2, num_rows*2))
    for i in range(num_rows*num_cols):
        plt.subplot(num_rows, num_cols, i+1)
        plt.imshow(x[i])
        plt.title(labels[y[i][0]])
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Plot 12 sample images from training set
plot_sample_images(x_train, y_train, CIFAR10_LABELS)
