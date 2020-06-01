import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist # : 케라스에서 지원하는 datasets을 불러오겠다
# mnist_data = mnist.load_data()
# 데이터 받아오기

# print(np.shape(mnist_data))
# 데이터 모양 보기

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(np.shape(x_train))
print(x_train[0])
# 데이터로 출력하기


plt.imshow(x_train[1]) # : imshow : 숫자로 된 데이터를 이미지로 보는 것
plt.title(y_train[1])
plt.show()
# 이미지 봐보기

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)), # : 2차원 행렬을 1차원으로 바꿔 준다
    tf.keras.layers.Dense(5, activation='tanh'), # : 5개의 뉴런을 설정
    tf.keras.layers.Dense(1, activation='tanh')
]) # : 뉴럴 네트워크 데이터 틀이 어떻게 되는지

out = model.predict([x_train[0:2]]) # : predict 함수는 3차원 행렬을 받는다.
print(out)
print(np.shape(out))
# 레이어 설계

print(x_train[0], type(x_train[0]))
print(y_train[0], type(y_train[0]))


temp_y = []
for one_y_val in y_train:
    zero_array = np.zeros(10)
    zero_array[one_y_val] = 1
    temp_y.append(zero_array) # : append : 설정한 array를 자리에 채워 준다

temp_y = np.array(temp_y)

print(type(temp_y))

# One_hot 만들기