import os

import numpy as np
import tensorflow as tf

from model.GuidedFilter import GuidedFilter
from model.UNet import UNet

def create_refined_unet_lite(input_channels, num_classes, r=60, eps=1e-4, unet_pretrained=None):
    """ Create Refined UNet lite
    """

    # Input
    inputs = tf.keras.Input(shape=[None, None, input_channels], name='inputs')

    # Create UNet
    unet = UNet()

    # Restore pretrained model
    if unet_pretrained:
        checkpoint = tf.train.Checkpoint(model=unet)
        checkpoint.restore(tf.train.latest_checkpoint(unet_pretrained))
        print('UNet restored, at {}'.format(tf.train.latest_checkpoint(unet_pretrained)))

    # Create Guided filter layer
    guided_filter = GuidedFilter()

    # RGB channels
    image = inputs[..., 4:1:-1]

    # Guidance
    guide = tf.image.rgb_to_grayscale(image)
    
    # Forward
    logits = unet(inputs)
    refined_logits = guided_filter(guide, logits, r=r, eps=eps)

    return tf.keras.Model(inputs=inputs, outputs=[logits, refined_logits])