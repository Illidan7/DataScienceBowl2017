import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
import time

from keras.models import Sequential, Model
from keras.layers import Dense, Input, Activation
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import LSTM
from keras.layers.convolutional import Convolution3D, AveragePooling3D
from keras.layers.convolutional import MaxPooling3D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras import backend as K
K.set_image_dim_ordering('th')
start_time = time.time()
#Random seed
np.random.seed(123)
#Load training data
much_data = np.load('/scratch/user/shyamprabhakar92/muchdata-100-100-30.npy')
X_init = much_data[:100,0]
y_init = much_data[:100,1]
#Load test data
test_data = np.load('/scratch/user/shyamprabhakar92/testdata-100-100-30.npy')
patient_order = np.load('testpatientorder_30_100.npy')
patient_order = list(patient_order)

IMG_PX_SIZE = 100
HM_SLICES = 30

X = np.zeros((len(X_init),HM_SLICES,IMG_PX_SIZE,IMG_PX_SIZE))
y = np.zeros((len(y_init),2))
for i in range(0,len(X_init)):
    try:
        X[i] = X_init[i]
        y[i] = y_init[i]
    except:
        print("problem")
        continue

print("done")    
X_test = np.zeros((len(test_data),HM_SLICES,IMG_PX_SIZE,IMG_PX_SIZE))
y_test = np.zeros((len(test_data),1))
for i in range(0,len(test_data)):
    try:
        X_test[i] = test_data[i]
    except:
        print("problem_test")
        continue

solution = pd.read_csv('stage1_solution.csv', index_col=0)
for ind, row in solution.iterrows():
    n = patient_order.index(ind)
    y_test[n] = row[0]
print("done")

#Reshape to [samples][channels][width][height]
X = X.reshape(X.shape[0],1,HM_SLICES,IMG_PX_SIZE,IMG_PX_SIZE).astype('float32')
X_test = X_test.reshape(X_test.shape[0],1,HM_SLICES,IMG_PX_SIZE,IMG_PX_SIZE).astype('float32')

    
def _bn_relu(layer):
    norm = BatchNormalization(axis=1)(layer)
    return Activation("relu")(norm)

def _bn_relu_conv(layer, filters = 32):
    activation = _bn_relu(layer)
    return Convolution3D(filters, 5, 5, 5, activation='relu')(activation)


def _shortcut(input, residual):
    """Adds a shortcut between input and residual block and merges them with "sum"
    """
    # Expand channels of shortcut to match residual.
    # Stride appropriately to match residual (width, height)
    # Should be int if network architecture is correctly configured.
    input_shape = K.int_shape(input)
    residual_shape = K.int_shape(residual)
    stride_width = int(round(input_shape[2] / residual_shape[2]))
    stride_height = int(round(input_shape[3] / residual_shape[3]))
    stride_depth = int(round(input_shape[4] / residual_shape[4]))
    equal_channels = input_shape[1] == residual_shape[1]

    shortcut = input
    # 1 X 1 conv if shape is different. Else identity.
    if stride_width > 1 or stride_height > 1 or stride_depth > 1 or not equal_channels:
        shortcut = Convolution3D(residual_shape[1], 1, 1, 1, subsample=(stride_width, stride_height, stride_depth), border_mode = "valid", init="he_normal",W_regularizer=l2(0.0001))(input)

    return shortcut


def residual_block(layer, filters = 32):
    conv = _bn_relu_conv(layer, filters)
    residual = _bn_relu_conv(conv, filters)
    op = _shortcut(layer,residual)
    op += residual
    layer = op
#    layer = Add([layer, residual])
#    layer = Merge([layer, residual], mode = 'sum')
    conv = _bn_relu_conv(layer, filters)
    residual = _bn_relu_conv(conv, filters)
    op = _shortcut(layer,residual)
    op += residual
    layer = op
#    layer = Add([layer, residual])
    
    return layer
    


def base_model():
    input_shape=(1, HM_SLICES, IMG_PX_SIZE, IMG_PX_SIZE)
    input = Input(shape=input_shape)
    conv1 = Convolution3D(32, 5, 5, 5, activation='relu')(input)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)

    filters = 32

    block = pool1
    block = residual_block(block)
    filters*=2

    block = residual_block(block, filters)

    block = _bn_relu(block)

    block_shape = K.int_shape(block)
    pool2 = AveragePooling3D(pool_size=(block_shape[2], block_shape[3], block_shape[4]), strides=(1, 1, 1))(block)
    flatten1 = Flatten()(pool2)
    dense = Dense(units=2, kernel_initializer="he_normal", activation="softmax")(flatten1)

    model = Model(input=input, output=dense)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
    

# build the model
model = base_model()
# Fit the model
model.fit(X, y, nb_epoch=10, batch_size=20,verbose=2)
model.summary()
#Prediction
predictions = model.predict(test_data, verbose=1)

logloss = log_loss(y_test,predictions)

print(logloss)

print("Total time: {} seconds".format(time.time() - start_time))
