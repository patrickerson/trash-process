from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

# data_trash = tf.data.experimental.make_csv_dataset(
#     'features.csv', 
#     header=False,
#     column_names=False,
#     batch_size=5,
#     num_epochs=1,
#     ignore_errors=True,


# )
# np.set_printoptions(precision=3, suppress=True)

data_trash =  keras.datasets.fashion_mnist
print(data_trash)

(train_images, train_labels), (test_images, test_labels) = data_trash.load_data()

print(data_trash)
class_names = ["cardboard", "glass", "paper", "plastic", "trash"]


train_images = train_images / 255.0

test_images = test_images / 255.0


plt.figure(figsize=(10,10))
for i in range(5):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.savefig("teste.png")

