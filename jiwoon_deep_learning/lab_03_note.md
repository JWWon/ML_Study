* alpha == learning_rate (보통 0.1을 값으로 준다)
* gradient descent를 텐서플로우로 표현

```
learning_rate = 0.1
# cost function을 미분한 것 == gradient
gradient = tf.reduce_mean((W*X - Y) * X)
descent = W - learning_rate * gradient
# update weight
update = W.assign(descent)
```
이는
```
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train = optimizer.minimize(cost)
```
와 동일한 작업을 함
