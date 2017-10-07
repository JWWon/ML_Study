* data-type : tf.float32 / tf.float64 / tf.int8 / tf.int16 / tf.32 / tf.64
* tf.Variable : 텐서플로우가 사용하는 변수 (우리가 사용하는게 아님), trainable한 변수 (스스로 변경 가능)
ex) tf.Variable(tf.random_normal([1]), name=‘’) -> 1차원 배열을 반환하는 변수를 선택
* reduce_mean : 평균을 반환 -> 변수로 배열을 입력

placeholder를 이용하면 나중에feed_dict를 이용해
```
sess.run([cost, W, b, train, feed_dict={X: [1, 2, 3, 4, 5], Y: [2.1, 3.1, 4.1, 5.1, 6.1]}])
```
식으로 값을 선언할 수 있다 (react의 component과 비슷)
