import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import glob # : 폴더 안에 파일명이 몇개 있는지 가져오는 것.
import numpy as np
scaler = MinMaxScaler()

PARA = ['UHOLEG1', 'UHOLEG2', 'UHOLEG3', 'ZINST58', 'ZINST99',
        'ZINST101', 'ZINST102', 'WFWLN1', 'WFWLN2', 'WFWLN3']
train_x, train_y = [], []

for one_file in (glob.glob('DB/*.csv')): # glob.glob : DB 안에 csv 파일을 모두 가져와라
    LOCA = pd.read_csv(one_file)

    if len(train_x) == 0:
        train_x = LOCA.loc[:, PARA].to_numpy() # .loc : 엑셀 내의 많은 변수 중에 PARA를 포함하는 열만 추려내겠다는 의미
        train_y = LOCA.loc[:, ['Accident_nub']].to_numpy()
    else:
        get_x = LOCA.loc[:, PARA].to_numpy()
        get_y = LOCA.loc[:, ['Accident_nub']].to_numpy()
        train_x = np.vstack((train_x, get_x))
        train_y = np.vstack((train_y, get_y))
    print(f'X_SHAPE : {np.shape(train_x)}, ' f'Y_SHAPE : {np.shape(train_y)}')
print('DONE')

import tensorflow as tf

scaler = MinMaxScaler()
scaler.fit(train_x)
N_train_x = scaler.transform(train_x)

import pickle
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

inputs = tf.keras.Input(10)
hiden = tf.keras.layers.Dense(100)(inputs)
output = tf.keras.layers.Dense(5, activation='softmax')(hiden)
model = tf.keras.models.Model(inputs, output)
model.save_weights('나의_모델.h5')

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(N_train_x, train_y, epochs=100)

out_trained = model.predict(N_train_x[0:60])
plt.plot(train_y[0:60])
plt.plot(out_trained)
plt.show()