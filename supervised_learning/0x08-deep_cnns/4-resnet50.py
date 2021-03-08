#!/usr/bin/env python3
"""
Script that builds the ResNet-50 architecture
"""
import tensorflow.keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """
    Returns the keras model
    """
    X = K.Input(shape=(224, 224, 3))
    kernel = K.initializers.he_normal()

    conv1 = K.layers.Conv2D(filters=64,
                            kernel_size=(7, 7),
                            strides=(2, 2),
                            padding='same')(X)

    norm1 = K.layers.BatchNormalization()(conv1)
    act1 = K.layers.Activation('relu')(norm1)
    pool1 = K.layers.MaxPool2D(pool_size=(3, 3),
                               strides=(2, 2),
                               padding='same')(act1)

    proj1 = projection_block(pool1, [64, 64, 256], 1)
    id1 = identity_block(proj1, [64, 64, 256])
    id1 = identity_block(id1, [64, 64, 256])

    proj2 = projection_block(id1, [128, 128, 512])
    id2 = identity_block(proj2, [128, 128, 512])
    id2 = identity_block(id2, [128, 128, 512])
    id2 = identity_block(id2, [128, 128, 512])

    proj3 = projection_block(id2, [256, 256, 1024])
    id3 = identity_block(proj3, [256, 256, 1024])
    id3 = identity_block(id3, [256, 256, 1024])
    id3 = identity_block(id3, [256, 256, 1024])
    id3 = identity_block(id3, [256, 256, 1024])
    id3 = identity_block(id3, [256, 256, 1024])

    proj4 = projection_block(id3, [512, 512, 2048])
    id4 = identity_block(proj4, [512, 512, 2048])
    id4 = identity_block(id4, [512, 512, 2048])

    avg_l = K.layers.AveragePooling2D(pool_size=(7, 7),
                                      strides=(1, 1))(id4)
    Y = K.layers.Dense(units=1000, activation='softmax',
                       kernel_initializer=kernel)(avg_l)
    model = K.models.Model(inputs=X, outputs=Y)
    return model
