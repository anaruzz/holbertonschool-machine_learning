#!/usr/bin/env python3
"""
A function that performs PCA color augmentation as described in the AlexNet paper
"""
import numpy as np
from tensorflow import keras

def pca_color(image, alphas):
    """
    Returns the augmented image
    """
    img = keras.preprocessing.image.img_to_array(image)
    reshaped_img = np.reshape(img, (img.shape[0] * img.shape[1], 3))
    mean = np.mean(reshaped_img, axis=0)
    img_c = reshaped_img - mean
    std_dv = np.std(img_c, axis=0)
    img_c /= std_dv
    cov = np.cov(img_c, rowvar=False)
    eig, p = np.linalg.eig(cov)
    delta = np.matmul(p, alphas*eig)
    pca_a = img_c + delta
    PCA = pca_a * std_dv + mean
    PCA = np.maximum(np.minimum(PCA, 255), 0).astype('uint8')
    PCA = PCA.reshape((img.shape[0], img.shape[1], 3))
    return PCA