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
                            padding='same',
                            activation='relu')(X)

    norm1 = K.layers.BatchNormalization()(conv1)
    act1 = K.layers.Activation('relu')(norm1)
    pool1 = K.layers.MaxPooling2D(pool_size=(3, 3),
                                  strides=(2, 2),
                                  padding='same')(act1)

    proj1 = projection_block(pool1, [64, 64, 256], 1)
    id1 = identity_block(proj1, [64, 64, 256])
    id2 = identity_block(id1, [64, 64, 256])

    proj2 = projection_block(id2, [128, 128, 512])
    id3 = identity_block(proj2, [128, 128, 512])
    id4 = identity_block(id3, [128, 128, 512])
    id5 = identity_block(id4, [128, 128, 512])

    proj3 = projection_block(id5, [256, 256, 1024])
    id6 = identity_block(proj3, [256, 256, 1024])
    id7 = identity_block(id6, [256, 256, 1024])
    id8 = identity_block(id7, [256, 256, 1024])
    id9 = identity_block(id8, [256, 256, 1024])
    id10 = identity_block(id9, [256, 256, 1024])

    proj4 = projection_block(id10, [512, 512, 2048])
    id11 = identity_block(proj4, [512, 512, 2048])
    id12 = identity_block(id11, [512, 512, 2048])

    avg_l = K.layers.AveragePooling2D(pool_size=(7, 7),
                                      strides=(1, 1))(id12)
    Y = K.layers.Dense(units=1000, activation='softmax',
                       kernel_initializer=kernel)(avg_l)
    model = K.models.Model(inputs=X, outputs=Y)
    return model
