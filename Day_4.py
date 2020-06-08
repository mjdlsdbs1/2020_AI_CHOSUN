import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(np.shape(x_train))
print(x_train[0])


plt.imshow(x_train[1])
plt.title(y_train[1])
plt.show()

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(30, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax') # 분류 문제에서 일반적으로 sofrtmax를 사용한다.
])

model.compile(optimizer='adam', # 계산된 편차값을 함수로써 어떻게 업데이트를 시킬지 판단하는 것.
              loss='sparse_categorical_crossentropy', # : 손실함수로 sparse_categorical_crossentropy를 쓰겠다. (주로 분류문제)
              metrics=['accuracy'])

print(np.shape(x_train), np.shape(y_train))
print(type(x_train), type(y_train))
model.fit(x_train, y_train, epochs=5) # Epoch : 학습을 몇 번을 시킬 것인가? / x train, y train을 훈련시킨다.
                                      # loss : 예측값과 실제값의 차이를 손실함수로 계산
# 어떤 함수를 사용해야 하느냐? : tensorflow 홈페이지 참조하기. (optimizer, activation 등 검색 가능)
# 훈련을 시키는 과정


print(model.predict(x_test[0:3]))
print(y_test[0:3])
# 검증

model.save_weights('save_model')