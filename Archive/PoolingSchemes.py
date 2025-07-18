import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow.keras.backend as K

def AverageOfMaximums(x, max_pool_size=2, layer_num=1):
    _max = layers.MaxPooling2D(pool_size=max_pool_size,padding='same', name="Maximums_L{}".format(layer_num))(x)
    _avg = layers.GlobalAveragePooling2D(pool_size=max_pool_size, name="Average_of_Maximums_L{}".format(layer_num))(_max)
    return _avg


def ChannelAverageFlatten(x, pool_size='infer', layer_num=1):
    if pool_size == 'infer':
        c = x.shape[-1]
        d = x.shape[1]
        if(d%2 == 1): d+=1
        n_head = c//d**2
        
        assert n_head * d**2 ==  c
    else:
        assert x.shape[-1] % pool_size == 0
        
        n_head = x.shape[-1] // pool_size
    
    x = layers.Reshape((x.shape[1],x.shape[2],n_head, -1), name="Reshape-ChannelAverageFlattenPooling-L{}".format(layer_num))(x)
    try:
        _channel_average = tf.math.reduce_max(x, axis=-1, name="ChannelAverage-ChannelAverageFlattenPooling-L{}".format(layer_num)) 
    except:
        _channel_average = layers.Lambda(lambda z: keras.backend.mean(z, axis=-1), name="ChannelAverage-ChannelAverageFlattenPooling-L{}".format(layer_num))(x)
    
    _channel_average = layers.Reshape((_channel_average.shape[1], _channel_average.shape[2], -1))(_channel_average)
    return _channel_average

def ChannelMaxPooling(x, pool_size='infer', layer_num=1):
    """
      pool_size If not "infer", then the final output size will be {h*w*(c/pool_size)}., where "c" is feature size.
      "infer" works only if {c%(h*w) == 0}; if true, then the final output size is equal to "c"

    """
    if pool_size == 'infer':
        c = x.shape[-1]
        d = x.shape[1]
        if(d%2 == 1): d+=1
        n_head = c//d**2
        
        assert n_head * d**2 ==  c
    else:
        assert x.shape[-1] % pool_size == 0
        
        n_head = x.shape[-1] // pool_size
    
    x = layers.Reshape((x.shape[1],x.shape[2],n_head, -1), name="Reshape-ChannelMaxFlattenPooling-L{}".format(layer_num))(x)
    try:
        _channel_max = tf.math.reduce_max(x, axis=-1, name="ChannelMax-ChannelMaxFlattenPooling-L{}".format(layer_num))
    except:
        _channel_max = layers.Lambda(lambda z: keras.backend.mean(z, axis=-1), name="ChannelAverage-ChannelAverageFlattenPooling-L{}".format(layer_num))(x)
        
    _channel_max = layers.Reshape((_channel_max.shape[1], _channel_max.shape[2], -1))(_channel_max)
    return _channel_max


class SpatialMaxPooling2D(tf.keras.layers.Layer):
    def __init__(self,pool_size=2,stride=None,padding='valid',data_format='channels_last', **kwargs):
        super(SpatialMaxPooling2D,self).__init__(**kwargs)
        self.pool_size = pool_size
        self.stride = stride
        self.padding = padding
        if stride is None:
            self.stride = self.pool_size

        self.in_data_format = data_format
        self.data_format = 'channels_last' if self.in_data_format == 'channels_first' else 'channels_first'
        self.max = tf.keras.layers.MaxPool1D(self.pool_size, self.stride, padding=self.padding, data_format=self.data_format)

    def call(self,x):
        input_shape = x.shape
        if self.data_format == 'channels_first':
            x = tf.keras.ops.reshape(x, (input_shape[0], -1,input_shape[-1]))
            x = self.max(x)
            x = tf.keras.ops.reshape(x, (input_shape[0], input_shape[1], input_shape[2], -1))
        else:
            x = tf.keras.ops.reshape(x, (input_shape[0], input_shape[-1], -1))
            x = self.max(x)
            x = tf.keras.ops.reshape(x, (input_shape[0], -1, input_shape[1], input_shape[2]))

        return x

class SpatialAveragePooling2D(tf.keras.layers.Layer):
    def __init__(self,pool_size=2,stride=None,padding='valid',data_format='channels_last', **kwargs):
        super(SpatialAveragePooling2D,self).__init__(**kwargs)
        self.pool_size = pool_size
        self.stride = stride
        self.padding = padding
        if stride is None:
            self.stride = self.pool_size

        self.data_format = 'channels_last' if data_format == 'channels_first' else 'channels_first'
        self.avg = tf.keras.layers.AveragePooling1D(self.pool_size, self.stride, padding=self.padding, data_format=self.data_format)

    def call(self,x):
        input_shape = x.shape
        if self.data_format == 'channels_first':
            x = tf.keras.ops.reshape(x, (input_shape[0], -1,input_shape[-1]))
            x = self.avg(x)
            x = tf.keras.ops.reshape(x, (input_shape[0], input_shape[1], input_shape[2], -1))
        else:
            x = tf.keras.ops.reshape(x, (input_shape[0], input_shape[-1], -1))
            x = self.avg(x)
            x = tf.keras.ops.reshape(x, (input_shape[0], -1, input_shape[1], input_shape[2]))

        return x



## Works with Tensorflow 2.16
"""ReLUQ: Quantile ReLU Activation Function
Instead of using a fixed max-value, this activation function uses the quantile value of the input tensor.
"""
class ReluQ(tf.keras.layers.Layer):
    def __init__(self, quantile=0.75, **kwargs):
        super(ReluQ, self).__init__(**kwargs)
        
        self.quantile = quantile
        self.relu = tf.keras.layers.ReLU(None, 0.3) ## LeakyReLU
        
        self.one = tf.constant(1.0)
        self.quantile_value = None
        
    def call(self, x, training=False):
        self.quantile_value = tf.keras.ops.min(tf.keras.ops.quantile(x, self.quantile, axis=-1))
        x = self.relu(x)
        x = tf.cond(self.quantile_value < self.one, lambda: x, lambda: tf.keras.ops.minimum(self.quantile_value, x))
        
        return x

### MKSA
def MultiKernelSpatialAttention(input_feature, kernels=[1,3,5,7], layer_num=0):
    in_size = input_feature.shape[1]

    channel = input_feature.shape[-1]
    x = input_feature

    pool_size = channel // 8
    
    channel_avg = SpatialAveragePooling2D(pool_size, padding='same')(x)
    channel_max = SpatialMaxPooling2D(pool_size, padding='same')(x)
    
    concat = layers.Concatenate(axis=3)([channel_avg, channel_max])
    
    attentions = []

    for kernel in kernels:
        attentions.append(
            layers.Conv2D(filters = 1,
                kernel_size=kernel,
                strides=1,
                padding='same',
                activation='sigmoid',
                kernel_initializer='he_normal',
                use_bias=False)(concat)
            )
      
    attentions = layers.Average()(attentions)

    x =  layers.Multiply()([input_feature, attentions])
    return x


import math
### MFSA
def MultiFilterSpatialAttention(input_feature, kernel_size=7, num_pooled_channel=4, num_filters=8, layer_num = 0):
    in_shape = input_feature.shape

    stride = int(math.ceil(in_shape[-1] / num_pooled_channel))
    pool_size = (stride * 2)

    channel_avg = SpatialAveragePooling2D(pool_size, stride, padding='same', name='MFSA_SAP_L{}'.format(layer_num))(input_feature)
    channel_max = SpatialMaxPooling2D(pool_size, stride, padding='same' , name='MFSA_SMP_L{}'.format(layer_num))(input_feature)

    concat = layers.Concatenate(axis=3, name='MFSA_ConcatChannels_L{}'.format(layer_num))([channel_avg, channel_max])

    sa_feature_x = layers.Conv2D(filters = num_filters,
                    kernel_size=kernel_size,
                    strides=1,
                    padding='same',
                    activation='sigmoid',
                    kernel_initializer='he_normal',
                    use_bias=False,
                    name="MFSA_Conv_L{}".format(layer_num))(concat)	

    sa_shape = sa_feature_x.shape

    sa_feature_x =  layers.Multiply([layers.Reshape((in_shape[1],in_shape[2],sa_shape[-1],-1), name="MFSA_InputForwardReshape_L{}".format(layer_num))(input_feature),
                              layers.Reshape((sa_shape[1],sa_shape[2],sa_shape[3],1), name="MFSA_AttentionForwardReshape_L{}".format(layer_num))(sa_feature_x)])
    
    sa_feature_x = layers.Reshape((in_shape[1], in_shape[2], -1), name="MFSA_OutBackwardReshape_L{}".format(layer_num))(sa_feature_x)
    return sa_feature_x


### Modified Spatial Attention
def ModifiedSpatialAttention(input_feature, kernel_size=5, num_pooled_channel=8, layer_num = 0):
    in_shape = input_feature.shape
    #pool_size =  in_shape[-1] // num_pooled_channel

    channel_avg = layers.Dense(num_pooled_channel, name="MSA_Feature_1L{}".format(layer_num))(input_feature)
    channel_max = layers.Dense(num_pooled_channel , name='MSA_Feature_2L{}'.format(layer_num))(input_feature)


    if K.image_data_format() == "channels_first":
        channel = input_feature.shape[1]
        cbam_feature = Permute((2,3,1))(input_feature)
    else:
        channel = input_feature.shape[-1]
        cbam_feature = input_feature

    concat = Concatenate(axis=3)([channel_avg, channel_max])

    cbam_feature = Conv2D(filters = 1,
                    kernel_size=kernel_size,
                    strides=1,
                    padding='same',
                    activation='sigmoid',
                    kernel_initializer='he_normal',
                    use_bias=False)(concat)
    assert cbam_feature.shape[-1] == 1

    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)

    return multiply([input_feature, cbam_feature])