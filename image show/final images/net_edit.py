from keras.modelss import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import Convolution2D, MaxPooling2D
import matplotlib.pyplot as plt

class Net:
   
    def build(widths,heights,depths,weightsPathh=None):
       
        models= Sequential()
        #first layer CONV-RELU-POOL
        models.add(Convolution2D(32, (3, 3),input_shape= (widths, heights, depths)))
        models.add(Activation('relu'))
        models.add(MaxPooling2D(pool_size=(2, 2)))

        #second layer layer CONV-RELU-POOL
        models.add(Convolution2D(32, (3, 3)))
        models.add(Activation('relu'))
        models.add(MaxPooling2D(pool_size= (2, 2)))

        #third layer of layer CONV-RELU-POOL
        models.add(Convolution2D(64, (3, 3)))
        models.add(Activation('relu'))
        models.add(MaxPooling2D(pool_size = (2, 2)))

        #set of FC-RELU layers
        models.add(Flatten())
        #FC layer = 128
        models.add(Dense(128))
        models.add(Activation('relu'))
        models.add(Dropout(0.5))
        #classes is 36
        models.add(Dense(36))
        models.add(Activation('softmax'))
        #weightsPathh is specified load the weights
        if weightsPathh is not None:
            print('weights loaded!!!')
            models.load_weights(weightsPathh)
          
        return models