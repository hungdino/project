import pandas as pd
import numpy as np
import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import matplotlib.pyplot as plt

def load_test_images(sample_submission_path, test_images_directory):
    sample_submission = pd.read_csv(sample_submission_path)
    images = []
    ids = sample_submission['id'].tolist()
    for img_id in ids:
        img_path = os.path.join(test_images_directory, img_id + '.jpg')
        img = image.load_img(img_path, target_size=(224, 224))
        img = image.img_to_array(img)
        img = preprocess_input(img)
        images.append(img)
    return np.array(images), ids

def display_test_images(images, ids, num_images=5):
    plt.figure(figsize=(15, 5))
    for i in range(num_images):
        plt.subplot(1, num_images, i+1)
        plt.imshow(images[i].astype('uint8'))
        plt.title(ids[i])
        plt.axis('off')
    plt.show()
