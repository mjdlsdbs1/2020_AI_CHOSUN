import pandas as pd
import matplotlib.pyplot as plt
LOCA_1 = pd.read_csv('DB/12_100010_60.csv')
plt.plot(LOCA_1['ZINST70'])
plt.plot(LOCA_1['QPRZP'])
plt.grid()
plt.show()
# 데이터 구조 및 특징 확인

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

import numpy as np

print(np.shape(LOCA_1['ZINST70']))

LOCA_1_VAL_1 = LOCA_1['ZINST70'].to_numpy()
print(np.shape(LOCA_1_VAL_1), type(LOCA_1_VAL_1))

LOCA_1_VAL_1 = LOCA_1_VAL_1.reshape((len(LOCA_1_VAL_1),1))
print(np.shape(LOCA_1_VAL_1), type(LOCA_1_VAL_1))

scaler.fit(LOCA_1_VAL_1)
print(scaler.data_max_)

LOCA_1_VAL_1_OUT = scaler.transform(LOCA_1_VAL_1)
plt.plot(LOCA_1_VAL_1_OUT)
plt.show()

# scikit learn을 사용하여 데이터를 정규화시키기

print(LOCA_1.loc[:, ['ZINST70', 'QPRZP']]) # .loc : 엑셀 데이터와 유사한 형식. : 은 처음부터 끝까지 불러오겠다. ['~', '~'] 는 두 개 변수를 불러와서 합치겠다.
LOCA_1_VAL_LIST = LOCA_1.loc[:, ['ZINST70', 'QPRZP']].to_numpy()
print(np.shape(LOCA_1_VAL_LIST), type(LOCA_1_VAL_LIST))
scaler.fit(LOCA_1_VAL_LIST)
LOCA_1_VAL_OUT = scaler.transform(LOCA_1_VAL_LIST)
plt.plot(LOCA_1_VAL_OUT)
plt.show()
