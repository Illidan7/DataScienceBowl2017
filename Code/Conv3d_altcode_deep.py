import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
import time

from keras.models import Sequential, Model
from keras.layers import Dense, Input
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import LSTM
from keras.layers.convolutional import Convolution3D
from keras.layers.convolutional import MaxPooling3D
from keras import backend as K
K.set_image_dim_ordering('th')
start_time = time.time()
#Random seed
np.random.seed(123)
#Load training data
much_data = np.load('/scratch/user/shyamprabhakar92/muchdata-100-100-30.npy')
X_init = much_data[:,0]
y_init = much_data[:,1]
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

def base_model():
    input_shape=(1, HM_SLICES, IMG_PX_SIZE, IMG_PX_SIZE)
    input = Input(shape=input_shape)
    conv1 = Convolution3D(32, 5, 5, 5, activation='relu')(input)
#    drop1 = Dropout(0.2)(conv1)
    conv2 = Convolution3D(32, 5, 5, 5, activation='tanh')(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)
    conv3 = Convolution3D(64, 5, 5, 5, activation='tanh')(pool1)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)
    drop2 = Dropout(0.2)(pool2)
    conv4 = Convolution3D(128, 3, 3, 3, activation='tanh')(drop2)
    drop = Dropout(0.2)(conv4)
    flatten = Flatten()(drop)
#    dense1 = Dense(1024, activation='tanh')(flatten)
#    drop3 = Dropout(0.2)(dense1)
    dense2 = Dense(512, activation='tanh')(flatten)
    drop4 = Dropout(0.2)(dense2)
    dense3 = Dense(128, activation='tanh')(drop4)
    dense4 = Dense(2, activation='sigmoid')(dense3)
    model = Model(input=input, output=dense4)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
    
# build the model
model = base_model()
# Fit the model
model.fit(X, y, nb_epoch=20, batch_size=30,verbose=2)
model.summary()
#Prediction
predictions = model.predict(test_data, verbose=1)

logloss = log_loss(y_test,predictions)

print(logloss)

print("Total time: {} seconds".format(time.time() - start_time))
