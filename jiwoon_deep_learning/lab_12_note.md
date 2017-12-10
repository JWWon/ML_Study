## RNN
이전 데이터가 다음 데이터에 영향을 끼쳐야 하는 데이터의 경우에 사용.
RNN(Recurrent Neural Network)

* cell은 여러개 처럼 표현하지만 RNN을 구성하는 function은 동일!
```
h(t) = f(W)(h(t-1), x(t))
```
- h(t) : new state
- f(W) : some function with parameters W
- h(t-1) : old state
- x(t) : input vector at some time step

#### Vanila RNN (기초 RNN)
- h(t) : tanh(W(h)(h) * h(t-1) + W(x)(h) * x(t))
- y(t) : W(h)(y) * h(t)

RNN 종류
> one to one | one to many | many to one | many to many | Multy-Layer

* hidden_size : 한 노드에서 출력되는 값의 갯수
* sequence_length : (예) 1 회에 입력받는 단어의 갯수, 한 번에 전개되는 시퀀스의 횟수
* batch_size : 한 번에 학습하는 데이터의 갯수

### RNN을 할 때에는 데이터가 많아질 수록 wide & deep 한 구조를 가져야 한다.
> Stacked RNN
```
# hidden size = 출력값
cell = tf.contrib.rnn.BasicLSTMCell(hidden_size, state_is_tuple=True)
# Stack RNN Cell
cell = tf.contrib.rnn.MultiRNNCell([cell] * 2, state_is_tuple=True)
```

> Softmax
softmax에 들어갈 수 있도록 출력값을 reshape(rank = 1)을 해준 후, 다시 펼친다
```
# outputs = RNN에서 나온 결과값
X_for_softmax = tf.reshape(outputs, [-1, hidden_size])
softmax_w = tf.get_variable("softmax_w", [hidden_size, num_classes])
softmax_b = tf.get_variable("softmax_b", [num_classes])
outputs = tf.matmul(X_for_softmax, softmax_w) + softmax_b

# outputs = Softmax에서 나온 결과값
outputs = tf.reshape(outputs, [batch_size, seq_length, num_classes])
```
그 후, 나온 outputs 를 cost 함수에 입력!
```
sequence_loss = tf.contrib.seq2seq.sequence_loss(logits = outputs, targets = Y, weights = weights)
mean_loss = tf.reduce_mean(sequence_loss)
```

### Dynamic RNN
> sequence_length가 동적인 RNN 모델
```
sequence_length = [5, 3, 4]
```

### RNN with time series data
> many to one RNN, appropriate model for predict stock market

