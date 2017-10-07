## Multi Variable
hypothesis = sum(X * W) + b

이 때, X와 W의 개수가 너무 많아지게 되면 이를 코드로 표현하는데 어려워지게 된다.
따라서 '행렬(Matrix)'를 이용해 이를 표현!

```
x_data = [[73., 80., 75.],
	  [93., 88., 93.],
          [89., 91., 90.],
          [96., 98., 100.],
          [73., 66., 70.]]


[73., 93., 89., 96., 73.], 
          [80., 88., 91., 98., 66.], 
          [75., 93., 90., 100., 70.]

]
y_data = [[152.], [185.], [180.], [196.], [142.]]
```
으로 데이터를 준 후,
```
X = tf.placeholder(tf.float32, shape=[None, 3]) # None: 데이터의 개수, 3: 데이터의 종류
Y = tf.placeholder(tf.float32, shape=[None, 1]  # None: 데이터의 개수, 1: 데이터의 종류

W = tf.Variable(tf.random_normal([3, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.matmul(W, X) + b
```
shape, matmul과 같은 함수를 이용해 행렬로 되어있는 데이터를 처리한다.
