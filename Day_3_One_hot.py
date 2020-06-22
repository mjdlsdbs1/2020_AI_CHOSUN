import numpy as np
input_val = 1

zero = np.zeros(shape=5)
print(zero, type(zero))

zero[input_val] = 1
print(zero, type(zero))

# One-hot 인코딩

# 왜 쓰는가?
# 1. 값을 예측하기
# Curve-fitting. 데이터들의 경향을 찾아서 fitting한다.
#
# 2. 분류를 할 때
# 데이터들을 어떠한 선으로 구분짓는 것. softmax는 분류 개념으로 사용한다.
# AI의 예측값과 인간이 정한 값과의 편차를 줄여나가는 방향으로 학습시킨다
