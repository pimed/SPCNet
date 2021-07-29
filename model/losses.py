import os
import random

import numpy as np
import pandas as pd
import SimpleITK as sitk
import tensorflow as tf

from keras import backend as K
from sklearn.model_selection import KFold
from imutils import rotate

def categorical_cross_entropy_balanced(num_normal=1, num_ind=1, num_agg=1):
    
    def loss(y_true, y_pred):
        # Assume pixels to be excluded are -1 in y_true, remove these before calculating y_true
        y_true = tf.cast(y_true, tf.float32)

        y_pred = tf.reshape(y_pred, [-1, 3])
        y_true = tf.reshape(y_true, [-1])

        y_pred = tf.boolean_mask(tensor=y_pred, mask=tf.greater(y_true, -1))
        y_true = tf.boolean_mask(tensor=y_true, mask=tf.greater(y_true, -1))

        # Note: tf.nn.sigmoid_cross_entropy_with_logits expects y_pred is logits, Keras expects probabilities.

        _epsilon = _to_tensor(K.epsilon(), y_pred.dtype.base_dtype)
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True) # ensure that entries for each pixel add to 1, should happen from softmax anyway
        y_pred = tf.clip_by_value(y_pred, _epsilon, 1 - _epsilon)

        y_normal = tf.where(tf.equal(y_true, 0.0), tf.ones_like(y_true), tf.zeros_like(y_true))
        y_ind = tf.where(tf.equal(y_true, 1.0), tf.ones_like(y_true), tf.zeros_like(y_true))
        y_ind = tf.where(tf.equal(y_true, 3.0), .5 * tf.ones_like(y_true), y_ind)
        y_agg = tf.where(tf.equal(y_true, 2.0), tf.ones_like(y_true), tf.zeros_like(y_true))
        y_agg = tf.where(tf.equal(y_true, 3.0), .5 * tf.ones_like(y_true), y_agg)

        num_total = num_normal + num_ind + num_agg

        y_true = tf.stack([y_normal, y_ind, y_agg], axis=-1)
        y_true = tf.cast(y_true, tf.float32)

        weights = tf.stack([num_total/num_normal, num_total/num_ind, num_total/num_agg]) * y_true
        weights = tf.reduce_sum(input_tensor=weights, axis=-1)

        cost = tf.losses.softmax_cross_entropy(onehot_labels=y_true, logits=K.log(y_pred), weights=weights, reduction='none')
        cost = tf.reduce_mean(input_tensor=cost)

        return cost

    return loss


def _to_tensor(x, dtype):
    """Convert the input `x` to a tensor of type `dtype`.
    # Arguments
    x: An object to be converted (numpy array, list, tensors).
    dtype: The destination type.
    # Returns
    A tensor.
    """
    x = tf.convert_to_tensor(value=x)
    if x.dtype != dtype:
        x = tf.cast(x, dtype)
    return x
