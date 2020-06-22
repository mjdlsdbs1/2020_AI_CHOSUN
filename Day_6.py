import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import glob # : 폴더 안에 파일명이 몇개 있는지 가져오는 것.
import numpy as np

# LOCA_1 = pd.read_csv('DB/(12, 100010, 25).csv')
# plt.plot(LOCA_1['ZINST70']/100)
# plt.plot(LOCA_1['Normal_0'])
# plt.show()


PARA = ['UHOLEG1', 'UHOLEG2', 'UHOLEG3', 'ZINST58']
train_x, train_y = [], []

for one_file in (glob.glob('DB/*.csv')): # glob.glob : DB 안에 csv 파일을 모두 가져와라
    LOCA = pd.read_csv(one_file)

    if len(train_x) == 0:
        train_x = LOCA.loc[:, PARA].to_numpy() # .loc : 엑셀 내의 많은 변수 중에 PARA를 포함하는 열만 추려내겠다는 의미
        train_y = LOCA.loc[:, ['Normal_0']].to_numpy()
    else:
        get_x = LOCA.loc[:, PARA].to_numpy()
        get_y = LOCA.loc[:, ['Normal_0']].to_numpy()
        train_x = np.vstack((train_x, get_x))
        train_y = np.vstack((train_y, get_y))
    print(f'X_SHAPE : {np.shape(train_x)}, ' f'Y_SHAPE : {np.shape(train_y)}')
print('DONE')

import tensorflow as tf

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(np.shape(train_x)[1]),
    tf.keras.layers.Dense(500),
    tf.keras.layers.Dense(np.shape(train_y)[1] + 1, activation='softmax'),
])


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_x, train_y, epochs=100)

out_trained = model.predict(train_x[0:60])
plt.plot(train_y[0:60])
plt.plot(out_trained)
plt.show()



# train_x = LOCA_1.loc[:, ['UHOLEG1', 'UHOLEG2', 'UHOLEG3', 'ZINST58']].to_numpy()
# print(np.shape(train_x), type(train_x))
#
# LOCA_2 = pd.read_csv('DB/(12, 200130, 30).csv')
# train_x2 = LOCA_2.loc[:, ['UHOLEG1', 'UHOLEG2', 'UHOLEG3', 'ZINST58']].to_numpy()
# print(np.shape(train_x2), type(train_x2))
#
# out_train_x = np.vstack((train_x, train_x2)) # : vstack : 데이터를 위아래로 쌓는 것.
# print(np.shape(out_train_x), type(out_train_x))