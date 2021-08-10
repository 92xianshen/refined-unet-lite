# -*- coding: utf-8 -*-

import os
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (Conv2D, Dropout, MaxPooling2D,
                                     UpSampling2D, concatenate)
from tensorflow.keras.optimizers import Adam


class UNet(Model):
    def __init__(self):
        super(UNet, self).__init__()

        self.conv1_1 = Conv2D(64, 3, activation='relu',
                              padding='same', name='conv1_1')
        self.conv1_2 = Conv2D(64, 3, activation='relu',
                              padding='same', name='conv1_2')
        self.pool1 = MaxPooling2D(pool_size=[2, 2], name='pool1')

        self.conv2_1 = Conv2D(128, 3, activation='relu',
                              padding='same', name='conv2_1')
        self.conv2_2 = Conv2D(128, 3, activation='relu',
                              padding='same', name='conv2_2')
        self.pool2 = MaxPooling2D(pool_size=[2, 2], name='pool2')

        self.conv3_1 = Conv2D(256, 3, activation='relu',
                              padding='same', name='conv3_1')
        self.conv3_2 = Conv2D(256, 3, activation='relu',
                              padding='same', name='conv3_2')
        self.pool3 = MaxPooling2D(pool_size=[2, 2], name='pool3')

        self.conv4_1 = Conv2D(512, 3, activation='relu',
                              padding='same', name='conv4_1')
        self.conv4_2 = Conv2D(512, 3, activation='relu',
                              padding='same', name='conv4_2')
        self.drop4 = Dropout(.5, name='drop4')
        self.pool4 = MaxPooling2D(pool_size=[2, 2], name='pool4')

        self.conv5_1 = Conv2D(1024, 3, activation='relu',
                              padding='same', name='conv5_1')
        self.conv5_2 = Conv2D(1024, 3, activation='relu',
                              padding='same', name='conv5_2')
        self.drop5 = Dropout(.5, name='drop5')

        self.up6 = UpSampling2D(size=[2, 2], name='up6')
        self.conv6_1 = Conv2D(512, 2, activation='relu',
                              padding='same', name='conv6_1')
        self.conv6_2 = Conv2D(512, 3, activation='relu',
                              padding='same', name='conv6_2')
        self.conv6_3 = Conv2D(512, 3, activation='relu',
                              padding='same', name='conv6_3')

        self.up7 = UpSampling2D(size=[2, 2], name='up7')
        self.conv7_1 = Conv2D(256, 2, activation='relu',
                              padding='same', name='conv7_1')
        self.conv7_2 = Conv2D(256, 3, activation='relu',
                              padding='same', name='conv7_2')
        self.conv7_3 = Conv2D(256, 3, activation='relu',
                              padding='same', name='conv7_3')

        self.up8 = UpSampling2D(size=[2, 2], name='up8')
        self.conv8_1 = Conv2D(128, 2, activation='relu',
                              padding='same', name='conv8_1')
        self.conv8_2 = Conv2D(128, 3, activation='relu',
                              padding='same', name='conv8_2')
        self.conv8_3 = Conv2D(128, 3, activation='relu',
                              padding='same', name='conv8_3')

        self.up9 = UpSampling2D(size=[2, 2], name='up9')
        self.conv9_1 = Conv2D(64, 2, activation='relu',
                              padding='same', name='conv9_1')
        self.conv9_2 = Conv2D(64, 3, activation='relu',
                              padding='same', name='conv9_2')
        self.conv9_3 = Conv2D(64, 3, activation='relu',
                              padding='same', name='conv9_3')
        self.conv9_4 = Conv2D(16, 3, activation='relu',
                              padding='same', name='conv9_4')

        self.conv10 = Conv2D(4, 1, activation=None, name='conv10')

    def call(self, x, training=True):
        conv1 = self.conv1_2(self.conv1_1(x))
        pool1 = self.pool1(conv1)

        conv2 = self.conv2_2(self.conv2_1(pool1))
        pool2 = self.pool2(conv2)

        conv3 = self.conv3_2(self.conv3_1(pool2))
        pool3 = self.pool3(conv3)

        conv4 = self.conv4_2(self.conv4_1(pool3))
        drop4 = self.drop4(conv4, training=training)
        pool4 = self.pool4(drop4)

        conv5 = self.conv5_2(self.conv5_1(pool4))
        drop5 = self.drop5(conv5, training=training)

        up6 = self.conv6_1(self.up6(drop5))
        merge6 = concatenate([drop4, up6], axis=-1)
        conv6 = self.conv6_3(self.conv6_2(merge6))

        up7 = self.conv7_1(self.up7(conv6))
        merge7 = concatenate([conv3, up7], axis=-1)
        conv7 = self.conv7_3(self.conv7_2(merge7))

        up8 = self.conv8_1(self.up8(conv7))
        merge8 = concatenate([conv2, up8], axis=-1)
        conv8 = self.conv8_3(self.conv8_2(merge8))

        up9 = self.conv9_1(self.up9(conv8))
        merge9 = concatenate([conv1, up9], axis=-1)
        conv9 = self.conv9_4(self.conv9_3(self.conv9_2(merge9)))

        conv10 = self.conv10(conv9)

        return conv10
