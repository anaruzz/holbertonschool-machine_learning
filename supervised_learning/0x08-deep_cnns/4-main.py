#!/usr/bin/env python3

import tensorflow.keras as K
resnet50 = __import__('4-resnet50').resnet50

if __name__ == '__main__':
    model = resnet50()
    model.summary()
#!/usr/bin/env python3
#
# import tensorflow.keras as K
# resnet50 = __import__('4-resnet50').resnet50
#
# model = resnet50()
# for layer in model.layers:
#     if type(layer) is K.layers.Conv2D:
#         for k, v in sorted(layer.kernel_initializer.__dict__.items()):
#             print(k, v)
