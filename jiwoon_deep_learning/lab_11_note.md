## CNN
* Pooling : 여러개의 픽셀값 중 하나를 뽑는(sampling) 하는 것으로, 가장 큰 값을 뽑는 Max Pooling이 많이 쓰임
* stride : 한 번 건너뛸 때 몇 개의 픽셀을 건너뛰어 학습시킬 것인지 정하는 값

### tip
#### 학습하는 부분을 Class, function으로 선언
Example 1
```
class Model:
	def __init__(self, sess, name):
		self.sess = sess
		self.name = name
		self._build_net()
	
	def _build_net(self):
		with tf.variable_scope(self.name):
			self.X = tf.placeholder(tf.float32, [None, 784])
			X_img = tf.reshape(self.X, [-1, 28, 28, 1])
			self.Y = tf.placeholder(tf.float32, [None, 10])
			
			W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
```
Example 2
```
def predict(self, x_test, keep_prob=1.0):
	return self.sess.run(self.logits, feed_dict={self.X: x_test, self.keep_prob: keep_prob})
```

#### tf.layers
숫자가 많아서 복잡해지는 부분을 단순화 해줄 수 있음

Example
```
conv1 = tf.layers.conv2d(inputs=X_img, filters=32, kernel_size=[3, 3], padding='SAME', activation=tf.nn.relu)
pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], padding='SAME', strides=2)
dropout1 = tf.layers.dropout(inputs=pool1, rate=0.7, training=self.training)
```

## Ensemble
여러 모델들로 결과값을 각각 계산한 다음에, 이를 추산하는 방법

