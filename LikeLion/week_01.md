# Regularizing your neural network

### Dropout Reqularization
dropout: 각 layer에서 의도적으로 node 몇 개를 삭제해 학습
random으로 삭제할 node 및 layer를 결정

사용법
```
keep_prob = 0.8 //0.2만큼 dropout이 될 가능성이 있다
d3 = np.random.rand(a3.shape[0], a3.shape[1] < keep_prob)
a3 = np.multiply(a3, d3) // a3 *= d3
```
그 후,
```
a3 /= keep_prob //a3의 기댓값이 동일하게끔 맞춰주는 작업
```
을 해준다.

* test를 할 때에는 스케일링 등등을 처리하지 않아도 된다.

### Understanding Dropout
dropout : layer마다 keep_prop값을 다르게 줄 수 있다
dropout을 통해 regularization의 역할을 할 수 있다. 또한 데이터 값이 엄청나게 큰 경우 dropout을 통해 처리속도를 줄일 수 있다

tip! dropout을 하기 전에 keep_prob 값을 1로 통일시켜 cost값이 점점 감소하는지 우선적으로 확인한 후, keep_prob값을 조절해 regularization을 유도한다

### Other Reqularization Methods
if (data == image) distort image (crop, flip...)

---------------------------------------------------------------------

# Setting Up Your Optimization Problem

### Normalizing Inputs
최종 목표 : 각 축의 값들의 분포가 치우쳐지지 않고 정각형/원형 모양으로 되게끔 설정!
-> 데이터 학습이 효과적으로 일어나도록 유도할 수 있음
(in Tensorflow => MinMaxScaler)

### Vanishing / Exploding gradients
weight 값이 단위행렬 I보다 크면 deep NN에서는 y값이 엄청 커지고 (exploding),
weight 값이 단위행렬 I보다 작으면 deep NN에서는 y값이 엄청 작아진다(vanishing).

그리고 이것은 학습을 어렵게 만듦!!

### Weight Initialization for Deep Networks
Larger n => Smaller w[i] 이 되도록 설계
따라서, 처음 weight값을 설정할 때
```
W = tf.Variable(tf.random_normal(~~~)*np.sqrt(1/n[l-1]))
// np.sqrt() : Xavier initialization
```
등의 방식으로 사용한다.

### Numerical approximation of gradients
derivative 계산
미분의 정의에 따라, f'(theta) = lim (f(theta+epsilon) - f(theta-epsilon))/2theta
인데 이 때, epsilon값을 0에 가까울 수록 error값이 작아지게 된다
