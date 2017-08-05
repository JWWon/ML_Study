import tensorflow as tf
import random
tf.set_random_seed(777)

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

nb_classes = 10

# MNIST data set (28 * 28 = 784)
X = tf.placeholder(tf.float32, [None, 784])
# 0 - 9 digits recognition
y = tf.placeholder(tf.float32, [None, nb_classes])

W = tf.Variable(tf.random_normal([784, nb_classes]))
b = tf.Variable(tf.random_normal([nb_classes]))

# Hypothesis
hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)

cost = tf.reduce_mean(-tf.reduce_mean(y * tf.log(hypothesis), axis = 1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# Test model
is_correct = tf.equal(tf.arg_max(hypothesis, 1), tf.arg_max(y, 1))
# Calc accuracy
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32)) # reduce_mean -> 축의 평균값을 받아옴 / cast -> 형변환

# parameters
training_epochs = 15 # epochs -> 전체 데이터셋을 한번 학습시킨 것 = 1 epoch
batch_size = 100 # 1회로 잘라내는 데이터 수

with tf.Session() as sess :
  sess.run(tf.global_variables_initializer())
  for epoch in range(training_epochs) :
    avg_cost = 0
    total_batch = int(mnist.train.num_examples / batch_size)

    for i in range(total_batch) :
      batch_xs, batch_ys = mnist.train.next_batch(batch_size)
      c, _ = sess.run([cost, optimizer], feed_dict= {X: batch_xs, y: batch_ys})
      avg_cost += c / total_batch

    print('Epoch: ', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

  print("Learning Finished")

  # Check Accuracy
  print("Accuracy: ", accuracy.eval(session=sess, feed_dict={X: mnist.test.images, y: mnist.test.labels}))

  # Get one and predict
  r = random.randint(0, mnist.test.num_examples - 1)
  print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))
  print("Prediction: ", sess.run(
      tf.argmax(hypothesis, 1), feed_dict={X: mnist.test.images[r:r + 1]}))

  import matplotlib.pyplot as plt

  # Get one & predict
  r = random.randint(0, mnist.test.num_examples - 1)
  print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r+1], 1)))
  print("Prediction: ", sess.run(tf.argmax(hypothesis, 1),
    feed_dict={X: mnist.test.images[r:r+1]}))

  plt.imshow(mnist.test.images[r:r+1].
    reshape(28, 28), cmap='Greys', interpolation='nearest')
  plt.show()