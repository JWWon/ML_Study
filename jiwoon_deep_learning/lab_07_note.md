## 머신러닝을 할 때에 주의해야 할 점
1. learning_rate 설정
너무 값이 큼: cost가 발산
너무 값이 작음: local_minimum에 빠져버리거나 cost가 줄어드는 정도가 너무 미미함

2. normalization 설정
normalization:
x축 값과 y축 값의 차이를 줄여 weight과 bias가 효율적으로 변경될 수 있도록 조정하는 것,
max = 1, min = 0으로 설정하고 모든 값을 0 - 1 사이로 설정
사용법
```
xy = MinMaxScaler(xy)
```

## MNIST 데이터셋 이용해서 연습
각 이미지: 28 * 28, 784 pixel로 구성되어 있음
nb_classes = 0 - 9 까지 총 10

#### 참고
* tf.arg_max(X, n) = X행렬의 n차원 행을 나열
* 1번 학습 = 1 epochs
* batch = slice data
* accuracy : .eval() 함수를 이용해 뽑아낸다 (0 - 1 사이의 값으로 나옴)

#### matplotlib -> 데이터 시각화
