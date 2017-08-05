# Github : https://github.com/hunkim/DeepLearningZeroToAll/blob/master/lab-05-1-logistic_regression.py
import tensorflow as tf

x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
y_data = [[0], [0], [0], [1], [1], [1]]

X = tf.placeholder(tf.float32, shape=[None, 2]) #x_data 정의
Y = tf.placeholder(tf.float32, shape=[None, 1]) #y_data 정의
W = tf.Variable(tf.random_normal([2, 1]), name='weight') #앞 : 입력 파라미터 개수(x_data) / 뒤 : 출력 파라미터 개수(y_data)
b = tf.Variable(tf.random_normal([1]), name='bias')

# Hypothesis using sigmoid: tf.div(1., 1. + tf.exp(tf.matmul(X, W)))
hypothesis = tf.sigmoid(tf.matmul(X, W) + b)

# Cost Function
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y)*tf.log(1 - hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate = 0.01).minimize(cost) # learning_rate = alpha

# Accuracy computation
predicted = tf.cast(hypothesis > 0.5, dtype = tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

# Launch graph
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())

  for step in range(10001):
    cost_val, _ = sess.run([cost, train], feed_dict={X:  x_data, Y: y_data})
    if step % 200 == 0:
      print(step, cost_val)

  h, c, a = sess.run([hypothesis, predicted, accuracy],
    feed_dict={X: x_data, Y: y_data})
  print("\nHypothesis: ", "\nCorrect (Y): ", c, "\nAccuracy: ", a)