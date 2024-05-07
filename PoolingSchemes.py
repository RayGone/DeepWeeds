import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers


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

        self.data_format = 'channels_last' if data_format == 'channels_first' else 'channels_first'
        self.max = tf.keras.layers.MaxPool1D(self.pool_size, self.stride, padding=self.padding, data_format=self.data_format)

    def build(self,input_shape):
        self.reshape_forward = tf.keras.layers.Reshape((-1,input_shape[-1]))
        self.reshape_backward= tf.keras.layers.Reshape((input_shape[1], input_shape[2], -1))

    def call(self,x):
        x = self.reshape_forward(x)
        x = self.max(x)
        x = self.reshape_backward(x)
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

    def build(self,input_shape):
        self.reshape_forward = tf.keras.layers.Reshape((-1,input_shape[-1]))
        self.reshape_backward= tf.keras.layers.Reshape((input_shape[1], input_shape[2], -1))

    def call(self,x):
        x = self.reshape_forward(x)
        x = self.avg(x)
        x = self.reshape_backward(x)
        return x

