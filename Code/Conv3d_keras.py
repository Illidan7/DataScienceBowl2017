import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc

from keras.models import Sequential
from keras.layers import Dense
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
much_data = np.load('propdata-50-50-20.npy')
X_init = much_data[:,0]
y_init = much_data[:,1]
#Load test data
test_data = np.load('proptestdata-50-50-20.npy')
patient_order = np.load('propatientorder.npy')
patient_order = list(patient_order)
print(test_data.shape)
IMG_PX_SIZE = 50
HM_SLICES = 20

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

X_test = np.zeros((len(test_data),20,50,50))
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
    #create model
    model = Sequential()
    model.add(Convolution3D(32, 5, 5, 5, input_shape=(1, HM_SLICES, IMG_PX_SIZE, IMG_PX_SIZE), activation='relu'))
#    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(Dropout(0.2))
    model.add(Convolution3D(32, 5, 5, 5, activation='relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
#    model.add(Convolution3D(64, 2, 2, 2, activation='relu'))
#    model.add(Dropout(0.2))
#    model.add(Convolution3D(64, 5, 5, 5, activation='relu'))
#    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
#    model.add(Dropout(0.2))
#    model.add(Convolution3D(128, 5, 5, 5, activation='relu'))
#    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(Flatten())
#    model.add(Dense(1024, activation='tanh'))
#    model.add(Dropout(0.2))
    model.add(Dense(512, activation='tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='tanh'))
    model.add(Dense(2, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
# build the model
model = base_model()
# Fit the model
model.fit(X, y, nb_epoch=20, batch_size=30,verbose=2)
model.summary()
#Prediction
predictions = model.predict(X_test, verbose=1)
scores = model.predict_proba(X_test)
scores = [s[1] for s in scores]

np.save('Conv_prop_pred.npy', predictions)
np.save('Conv_prop_scores.npy', scores)
np.save('Conv_prop_true.npy', y_test)

AUC_value = roc_auc_score(y_test, scores)

#acc = accuracy_score(y_test, predictions)
fpr, tpr, thresholds = roc_curve(y_test, scores, pos_label=1)
roc_auc = auc(fpr, tpr)
print("AUC Score: {}" , AUC_value)
#print("Accuracy: {}", acc)
plt.title('ROC curve of Random Forest')
plt.plot(fpr, tpr, 'b', label='AUC = %0.4f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


logloss = log_loss(y_test,predictions)
print("Logloss metric: {}".format(logloss))

print("Total time: {} seconds".format(time.time() - start_time))


            
