# --------------------------------------------------------
# SPCNet training
# Copyright (c) 2021 PIMED@Stanford
#
# Written by Arun Seetharaman
# --------------------------------------------------------
import numpy as np

from keras.layers import Conv2D, Conv2DTranspose, Input, MaxPooling2D, BatchNormalization
from keras.layers import Concatenate, Activation
from keras.models import Model
from keras.optimizers import Adam

from .losses import categorical_cross_entropy_balanced


class SPCNet_all():
    # Thresholds selected based on optimum performance on cross-validation sets
    agg_threshold = .20
    ind_threshold = .15
    normal_threshold = .60
    agg_bias = .14

    def network(self, lr, size=224, num_channels=3):
        def side_branch(x, factor):
            # Side branch to bottleneck into one channel and then upsample by appropriate factor
            x = Conv2D(3, (1, 1), activation=None, padding='same')(x)

            kernel_size = (2 * factor, 2 * factor)
            x = Conv2DTranspose(3, kernel_size, strides=factor, padding='same', use_bias=False, activation=None)(x)
            return x

        # Input
        t2 = Input(shape=(size, size, num_channels), name='t2')
        adc = Input(shape=(size, size, num_channels), name='adc')

        # Block 1-1 (T2)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(t2)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
        x = BatchNormalization()(x)
        b11 = side_branch(x, 1)  # Upsample by 1
        x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block1_pool')(x)  # 240 240 64

        # Block 2-1 (T2)
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
        x = BatchNormalization()(x)
        b21 = side_branch(x, 2)  # Upsample by 2
        x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block2_pool')(x)  # 120 120 128

        # Block 3-1 (T2)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
        x = BatchNormalization()(x)
        b31 = side_branch(x, 4)  # Upsample by 4
        x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block3_pool')(x)  # 60 60 256

        # Block 1-2 (ADC)
        x2 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1_2')(adc)
        x2 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2_2')(x2)
        x2 = BatchNormalization()(x2)
        b12 = side_branch(x2, 1)  # Upsample by 1
        x2 = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block1_pool_2')(x2)  # 240 240 64

        # Block 2-2 (ADC)
        x2 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1_2')(x2)
        x2 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2_2')(x2)
        x2 = BatchNormalization()(x2)
        b22 = side_branch(x2, 2)  # Upsample by 2
        x2 = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block2_pool_2')(x2)  # 120 120 128

        # Block 3-2 (ADC)
        x2 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1_2')(x2)
        x2 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2_2')(x2)
        x2 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3_2')(x2)
        x2 = BatchNormalization()(x2)
        b32 = side_branch(x2, 4)  # Upsample by 4
        x2 = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block3_pool_2')(x2)  # 60 60 256

        x = Concatenate(axis=-1)([x, x2])

        # Block 4
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
        x = BatchNormalization()(x)
        b4 = side_branch(x, 8)  # Upsample by 8
        x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block4_pool')(x)  # 30 30 512

        # Block 5
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)  # 30 30 512
        x = BatchNormalization()(x)
        b5 = side_branch(x, 16)  # Upsample by 16

        # Fuse the intermediate outputs into a single output
        fuse = Concatenate(axis=-1)([b11, b21, b31, b12, b22, b32, b4, b5])
        fuse = Conv2D(3, (1, 1), padding='same', use_bias=False, activation=None)(fuse)  # 480 480 1

        # outputs
        o11 = Activation('softmax', name='o11')(b11)
        o21 = Activation('softmax', name='o21')(b21)
        o31 = Activation('softmax', name='o31')(b31)
        o12 = Activation('softmax', name='o12')(b12)
        o22 = Activation('softmax', name='o22')(b22)
        o32 = Activation('softmax', name='o32')(b32)
        o4 = Activation('softmax', name='o4')(b4)
        o5 = Activation('softmax', name='o5')(b5)
        ofuse = Activation('softmax', name='ofuse')(fuse)

        # model
        model = Model(inputs=[t2, adc], outputs=[o11, o21, o31, o12, o22, o32, o4, o5, ofuse])

        # optimizer = self.create_optimizer()
        optimizer = Adam(lr=lr)

        #-------------------------------------------------------------------------------
        # num_normal, num_indolent and num_agg pixels are the number of normal, indolent and aggressive pixels computed in the training set
        model.compile(loss={'o11': categorical_cross_entropy_balanced(num_normal=9468014, num_agg=346237, num_ind=369685),
                            'o21': categorical_cross_entropy_balanced(num_normal=9468014, num_agg=346237, num_ind=369685),
                            'o31': categorical_cross_entropy_balanced(num_normal=9468014, num_agg=346237, num_ind=369685),
                            'o12': categorical_cross_entropy_balanced(num_normal=9468014, num_agg=346237, num_ind=369685),
                            'o22': categorical_cross_entropy_balanced(num_normal=9468014, num_agg=346237, num_ind=369685),
                            'o32': categorical_cross_entropy_balanced(num_normal=9468014, num_agg=346237, num_ind=369685),
                            'o4': categorical_cross_entropy_balanced(num_normal=9468014, num_agg=346237, num_ind=369685),
                            'o5': categorical_cross_entropy_balanced(num_normal=9468014, num_agg=346237, num_ind=369685),
                            'ofuse': categorical_cross_entropy_balanced(num_normal=9468014, num_agg=346237, num_ind=369685),
                            },
                      optimizer=optimizer)

        return model

    def get_x_y(self, t2_np, adc_np, y=None):
        x = [t2_np, adc_np]
        if y is not None:
           y = [y, y, y, y, y, y, y, y, y]
        num_channels = 3
        return x, y, num_channels

    def model_prediction(self, model, x):
        return model.predict(x)[-1]

    def model_load(self, filepath, lr):
        model = self.network(lr, num_channels=3)
        model.load_weights(filepath)

        return model

    def return_mod_prediction(self, mask_np, avg_prediction_normal, avg_prediction_agg, avg_prediction_ind):
        mod_prediction = np.zeros_like(avg_prediction_normal)
        mod_prediction[avg_prediction_ind >= self.ind_threshold] = 1
        mod_prediction[avg_prediction_agg >= self.agg_threshold] = 2
        mod_prediction[avg_prediction_normal >= self.normal_threshold] = 0
        mod_prediction[mask_np == 0] = 0

        return mod_prediction
