## Logistic Regression
0, 1 으로 케이스를 구분 (ex. 스팸메일, show facebook feed)
따라서 y_data(결과값) 데이터는 0, 1로 존재해야 함

#### tip!
* tf.random_normal 에서 [input 개수, output 개수] 로 생각하기

sigmoid 함수 = tf.sigmoid(tf.matmul(X, W) + b)
             = tf.div(1., 1. + tf.exp(tf.matmul(X, W) + b))
cost 함수 = -tf.reduce_mean(Y * tf.log(hypothesis) + (1-Y) * tf.log(1-hypothesis))

tf.cast -> true,false 여부로 1 / 0 반환

