#import keras.backend as K
#from keras.engine.topology import Layer

import tensorflow as tf
from tensorflow.keras.layers import Layer, InputSpec
from tensorflow.keras import initializers, regularizers, constraints
#from tensorflow.keras import backend as K

#from keras import initializations
#from keras import regularizers
#from keras import constraints
import numpy as np
#import theano.tensor as T

class Attention(tf.keras.layers.Layer):#두개의 입력 텐서를 받아 가중치를 입력 masking함. 
    def __init__(self,
                W_regularizer=None,
                b_regularizer=None,
                W_constraint=None,
                b_constraint=None,
                bias=True, **kwargs):
        """
        Keras Layer that implements an Content Attention mechanism.
        Supports Masking.
        """
        self.supports_masking = True
        self.init = initializers.glorot_uniform()#수정

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(Attention, self).__init__(**kwargs)
        #self.input_spec = [InputSpec(ndim=3), InputSpec(ndim=3)]#추가

    def build(self, input_shape):
        assert type(input_shape) == list
        #assert len(input_shape) == 3
        assert len(input_shape) == 2


        self.steps = input_shape[0][1]

        self.W = self.add_weight(shape=(input_shape[0][-1], input_shape[1][-1]),
                                    initializer=self.init,
                                    name='{}_W'.format(self.name),
                                    regularizer=self.W_regularizer,
                                    constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight(shape=(1,),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        self.built = True

    def compute_mask(self, input_tensor, mask=None):
        return None

    def call(self, input_tensor, mask=None):
        x = input_tensor[0]
        y = input_tensor[1]
        mask = mask[0]

        y = tf.transpose(tf.matmul(self.W, tf.transpose(y)))
        y = tf.expand_dims(y, axis=-2)#수정
        y = tf.repeat(y, self.steps, axis=1)
        eij = tf.reduce_sum(x*y, axis=-1)

        if self.bias:
            b = tf.repeat(self.b, self.steps, axis=0)
            eij += b

        eij = tf.tanh(eij)
        a = tf.exp(eij)

        if mask is not None:
            a *= tf.cast(mask, tf.keras.backend.floatx())

        a /= tf.cast(tf.reduce_sum(a, axis=1, keepdims=True) + tf.keras.backend.epsilon(), tf.keras.backend.floatx())
        return a

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1])

class WeightedSum(tf.keras.layers.Layer):#attention 레이어의 결과와 입력텐서를 가중합하여 반환. 
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(WeightedSum, self).__init__(**kwargs)

    def call(self, input_tensor, mask=None):
        assert type(input_tensor) == list
        assert type(mask) == list

        x = input_tensor[0]
        a = input_tensor[1]

        a = tf.expand_dims(a,axis=-1)#axis=-1 추가
        weighted_input = x * a

        return tf.reduce_sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][-1])

    def compute_mask(self, x, mask=None):
        return None

class WeightedAspectEmb(tf.keras.layers.Layer):#입력 벡터에 대해 가중치를 학습하여 출력벡터를 생성. 
    def __init__(self, input_dim, output_dim,
                 init='uniform', input_length=None,
                 W_regularizer=None, activity_regularizer=None,
                 W_constraint=None,
                 weights=None, dropout=0., **kwargs):
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.init = initializers.get(init)
        self.input_length = input_length
        self.dropout = dropout

        self.W_constraint = constraints.get(W_constraint)
        self.W_regularizer = regularizers.get(W_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        if 0. < self.dropout < 1.:
            self.uses_learning_phase = True
        self.initial_weights = weights
        kwargs['input_shape'] = (self.input_length,)
        #kwargs['input_dtype'] = tf.keras.backend.floatx()#여기서 오류가 났다는데 어떻게 해결하지. 일단 삭제

        super(WeightedAspectEmb, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(shape=(self.input_dim, self.output_dim),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
        self.built = True

    def compute_mask(self, x, mask=None):
        return None

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

    def call(self, x, mask=None):
        return tf.matmul(x, self.W)


class Average(tf.keras.layers.Layer):#입력텐서의 평균을 계산하여 반환한다. 
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(Average, self).__init__(**kwargs)

    def call(self, x, mask=None):
        if mask is not None:
            mask = tf.cast(mask, tf.keras.backend.floatx())
            mask = tf.expand_dims(mask,axis=-1)#수정, axis=-1
            x = x * mask
        return tf.reduce_sum(x, axis=-2) / tf.reduce_sum(mask, axis=-2)

    def compute_output_shape(self, input_shape):
        return input_shape[0:-2]+input_shape[-1:]
    
    def compute_mask(self, x, mask=None):
        return None


class MaxMargin(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(MaxMargin, self).__init__(**kwargs)

    def call(self, input_tensor, mask=None):
        z_s = input_tensor[0] 
        z_n = input_tensor[1]
        r_s = input_tensor[2]

        z_s = z_s / tf.cast(tf.keras.backend.epsilon() + tf.sqrt(tf.reduce_sum(tf.square(z_s), axis=-1, keepdims=True)), tf.keras.backend.floatx())
        z_n = z_n / tf.cast(tf.keras.backend.epsilon() + tf.sqrt(tf.reduce_sum(tf.square(z_n), axis=-1, keepdims=True)), tf.keras.backend.floatx())
        r_s = r_s / tf.cast(tf.keras.backend.epsilon() + tf.sqrt(tf.reduce_sum(tf.square(r_s), axis=-1, keepdims=True)), tf.keras.backend.floatx())

        steps = z_n.shape[1]

        pos = tf.reduce_sum(z_s*r_s, axis=-1, keepdims=True)
        pos = tf.repeat(pos, steps, axis=-1)
        r_s = tf.expand_dims(r_s, axis=-2)#수정
        r_s = tf.repeat(r_s, steps, axis=1)
        neg = tf.reduce_sum(z_n*r_s, axis=-1)

        loss = tf.cast(tf.reduce_sum(tf.maximum(0., (1. - pos + neg)), axis=-1, keepdims=True), tf.keras.backend.floatx())
        return loss

    def compute_mask(self, input_tensor, mask=None):
        return None

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], 1)




        
