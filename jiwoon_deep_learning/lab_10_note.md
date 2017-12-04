#### Tips for training MNIST
* NN layer를 쌓으면 cost값이 더욱 줄어든다, but 너무 깊어지고 넓어지면 overfitting이 발생함
* 초기화(initializer) 추천 라이브러리: Xavier
-> 초기값을 잘 주어 cost가 낮은 상태에서 출발할 수 있도록 도와줌
* 새로운 데이터에 작 적용시키기 위한 작업(overfitting을 줄임) : Dropout
```
keep_prob = tf.placehoder(tf.float32)

Layer1 = tf.nn.relu(tf.matmul(X1, W1) + b1)
//dropout
L1 = tf.nn.dropout(Layer1, keep_prob=keep_prob)
```
keep_prob 값은 후에 선언 가능(0-1 사이값)
보통은 train에선 0.5 ~ 0.7, test에선 1
* Optimizer로는 ADAM을 사용!
```
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
```
