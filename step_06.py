#
#  training feature
#
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import LSTM, Activation, Reshape
from tensorflow.keras import Sequential
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adadelta
from keras.callbacks import ModelCheckpoint
import keras.backend as K
from keras.regularizers import l2

df_utama = np.load('./real_dataset/hasil_combine.npy', allow_pickle=True)

df2 = df_utama[:,2]

df1 = df_utama[:,1]

stance = np.array(df_utama[:,3], dtype=np.int32)

print('./real_dataset/hasil_combine.npy loaded...')
print('headline, body, stances feature created')

# find Longest vector in padding bodies
max_len_bd = 0
for i in range(len(df1)):
    if len(df1[i]) > max_len_bd:
        max_len_bd = len(df1[i])
print(max_len_bd)

print('longest body vector is ', max_len_bd)

batas = (100 - (max_len_bd % 100)) + max_len_bd

hasilpad = []
for i in range(len(df1)):
    sisa = batas - len(df1[i])
    padbd = np.pad(df1[i], (0,sisa), 'constant')
    hasilpad.append(padbd)

bodies_np = np.array(hasilpad, dtype=np.float32)

bodies_np = bodies_np.reshape(len(bodies_np), 100, (len(bodies_np[0]) //100) )

bodies_np = bodies_np.reshape(len(bodies_np), 505, 280 )

# find Longest vector padding headlines
max_len_hd = 0
for i in range(len(df2)):
    if len(df2[i]) > max_len_hd:
        max_len_hd = len(df2[i])
print(max_len_hd)

print('longest headline vector is ', max_len_hd)

batas_hd = 2000

hasilpad_hd = []
for i in range(len(df2)):
    sisa = batas_hd - len(df2[i])
    padhd = np.pad(df2[i], (0,sisa), 'constant')
    hasilpad_hd.append(padhd)

headlines_np = np.array(hasilpad_hd, dtype=np.float32)

headlines_np = headlines_np.reshape(len(headlines_np), 20, 100 )

#create test and validation
print('create train and test')
df1_test, df1_val = bodies_np[:2000], bodies_np[2000:]
df2_test, df2_val = headlines_np[:2000], headlines_np[2000:]
stance_test, stance_val = stance[:2000], stance[2000:]

kernel_size = 3
weight_decay=0.0005

input1 = Input(shape=(505, 280), name = "input1")
input2 = Input(shape=(20, 100), name = "input2" )
dc1 = Conv1D(215, kernel_size, padding='valid', activation='relu', strides=1)(input1)
dc2 = Conv1D(215, kernel_size, padding='valid', activation='relu', strides=1)(input2)
a1 = concatenate([dc1,dc2], axis=1)
c21 = Conv1D(182, kernel_size, padding='valid', activation='relu', strides=1)(a1)
c24 = Conv1D(91, kernel_size, padding='valid', activation='relu', strides=1)(c21)
a9 = Flatten()(c24)
a421 = Dense(512,activation='relu')(a9)
a4 = Dense(512,activation='relu')(a421)
hasil = Dense(4,activation='softmax')(a4)
model = Model(inputs=[input1, input2], outputs=hasil)

print(model.summary())

model.compile(optimizer='Adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit([df1_test, df2_test], stance_test, epochs=24, verbose=1, batch_size=100, validation_data=([df1_val, df2_val], stance_val))

print('save ./real_dataset/model_final_24.h5')
model.save("./real_dataset/model_final_24.h5")
print('./real_dataset/model_final_24.h5 saved')
# import matplotlib.pyplot as plt
#
#
# # Get training and test loss histories
# training_loss = history.history["loss"]
# test_loss = history.history["val_loss"]
# # Create count of the number of epochs
# epoch_count = range(1, len(training_loss) + 1)
# # Visualize loss history
# plt.rcParams['figure.dpi'] = 500
# plt.plot(epoch_count, training_loss, "r--")
# plt.plot(epoch_count, test_loss, "b-")
# plt.legend(["Training Loss", "Test Loss"])
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.show();
#
#
# # Get training and test accuracy histories
# training_accuracy = history.history["accuracy"]
# test_accuracy = history.history["val_accuracy"]
# plt.plot(epoch_count, training_accuracy, "r--")
# plt.plot(epoch_count, test_accuracy, "b-")
# # Visualize accuracy history
# plt.rcParams['figure.dpi'] = 500
# plt.legend(["Training Accuracy", "Test Accuracy"])
# plt.xlabel("Epoch")
# plt.ylabel("Accuracy Score")
# plt.show();
