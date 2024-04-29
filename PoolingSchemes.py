import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers


def AverageOfMaximums(x, max_pool_size=2, layer_num=1):
    _max = layers.MaxPooling2D(pool_size=max_pool_size, name="Maximums_L{}".format(layer_num))(x)
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
    _channel_average = layers.Lambda(lambda z: keras.backend.mean(z, axis=-1), name="ChannelAverage-ChannelAverageFlattenPooling-L{}".format(layer_num))(x)
    _channel_average_flatten = layers.Flatten(name="Flatten-ChannelAverageFlattenPooling-L{}".format(layer_num))(_channel_average)
    return _channel_average_flatten