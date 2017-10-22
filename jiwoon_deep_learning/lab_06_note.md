## Softmax function
전체 합 = 1 이 되도록 score들을 변경시켜주는 함수
(score -> probablity)
```
# 사용
tf.nn.softmax(tf.matmul(X, W) + b)
```
```
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis = 1))
```

## One Hot Encoding
하나의 score만 1, 나머지는 0 으로 변경하는 함수
```
# 사용
tf.arg_max(x, 1)
```

## Fancy Softmax Classifier
위에서 사용한 cost 함수 선언이 너무 복잡하다
-> tensorflow에서 제공하는 또 다른 함수
tf.nn.softmax_cross_entropy_with_logits
(logits = tf.matmul(X, W) + b)
```
# 사용
cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=logits, lables=Y_one_hot)
cost = tf.reduce_mean(cost_i)
```

one_hot 사용: tf.one_hot(Y, nb_classes) -> Y: one_hot이 아닌 결과값, nb_classes: 케이스 개수
그런데 이 때 one_hot을 사용하게 되면 shape가 하나 늘어나게 됨
따라서 reshape를 해줘야 우리가 원하는 결과값을 얻을 수 있음
```
Y = tf.placeholder(tf.int32, [None, 1])
Y_one_hot = tf.one_hot(Y, nb_classes)
Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes])
```

# flatten : 
ex) [[1], [0]].flatten = [1, 0]
