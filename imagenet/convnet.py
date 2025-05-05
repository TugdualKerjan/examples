from typing import List
import equinox.nn as nn
import jax.numpy as np
import jax
import equinox as eqx
class ConvMax(eqx.Module):

    conv: nn.Conv2d
    act= jax.nn.relu
    pool: nn.MaxPool2d

    def __init__(self, in_chan, out_chan, kernel_size, padding, key):
        self.conv = nn.Conv2d(in_chan, out_chan, kernel_size, padding=padding, key=key)
        self.pool = nn.MaxPool2d(2)

    def __call__(self, x):
        y = self.conv(x)
        y = self.act(x)
        y = self.pool(x)
        return y

class VGG11(eqx.Module):
    layer_1: ConvMax
    layer_2: ConvMax
    layer_3: nn.Conv2d
    layer_4: ConvMax
    layer_5: nn.Conv2d
    layer_6: ConvMax
    layer_7: nn.Conv2d
    layer_8: ConvMax

    classifier: List

    def __init__ (self, key):
        k1, k2, k3, k4, k5, k6, k7, k8, k9,k10,k11 = jax.random.split(key, 11)

        self.layer_1 = ConvMax(3, 64, 3, padding=1, key=k1)
        self.layer_2 = ConvMax(64, 128, 3, padding=1, key=k2)
        self.layer_3 = nn.Conv2d(128, 256, 3, padding=1, key=k3)
        self.layer_4 = ConvMax(256, 256, 3, padding=1, key=k4)
        self.layer_5 = nn.Conv2d(256, 512, 3, padding=1, key=k5)
        self.layer_6 = ConvMax(512, 512, 3, padding=1, key=k6)
        self.layer_7 = nn.Conv2d(512, 512, 3, padding=1, key=k7)
        self.layer_8 = ConvMax(512, 512, 3, padding=1, key=k8)

        self.classifier = [
            nn.Linear(4096, 4096, key=k9),
            nn.Linear(4096, 4096, key=k10),
            nn.Linear(4096, 1000, key=k11),
        ]

    def __call__(self, x):
        y = self.layer_1(x)
        y = self.layer_2(y)
        y = self.layer_3(y)
        y = self.layer_4(y)
        y = self.layer_5(y)
        y = self.layer_6(y)
        y = self.layer_7(y)
        y = self.layer_8(y)

        y = np.reshape(y, -1)
        
        for layer in self.classifier:
            y = layer(y)
            y = jax.nn.relu(y)
        
        return y
    