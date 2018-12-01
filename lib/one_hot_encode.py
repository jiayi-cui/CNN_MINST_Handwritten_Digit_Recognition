import numpy as np

def one_hot_encode(x):
    encode = np.zeros((len(x),10))
    encode[np.arange(len(x)), x] =1
    return encode
