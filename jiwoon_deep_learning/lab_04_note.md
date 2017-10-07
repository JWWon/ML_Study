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

## Read Data From File
numpy 의 'loadtxt'를 이용

```
xy = np.loadtxt('data-01-test-scroe.csv', delimiter=',', dtype=np.float32)
```
그런 후, slice로 xy 데이터를 나눈다.

#### slice 규칙
* a[0:2] == a[0], a[1]
* a[:] == a[0], a[1], ..., a[n-1]
* 정리 : 앞에 적힌 숫자의 index부터 뒤에 적힌 index 전 index까지

```
x = xy[:, 0:-1]
y = xy[:, [-1]]
```

## Queue Runners (numpy)
메모리 부족을 대비해 (데이터가 너무 클 때)
File을 부분적으로 가져와 학습시킨 후, 다른 부분을 또 로드해 학습시키는 방식으로 작동)

#### step
* 파일을 지정 (여러 파일도 가능)
```
filename_queue = tf.train.string_input_producer(['data_01.csv', 'data-02.csv', ...], shuffle=False, name='filename_queue')
```
* 파일을 읽어올 Reader를 지정
```
reader = tf.TextLineReader()
key, value = reader.read(filename_queue)
```
* 읽어온 value를 파싱
```
record_defaults = [[0.], [0.], [0.], [0.]] # 값이 없을 때를 위해 default value를 지정
xy = tf.decode_csv(value, record_defaults=record_defaults)
```

이후, batch를 이용해 데이터를 분리 (slicing)
```
train_x_batch, train_y_batch = tf.train.batch([xy[0:-1], xy[-1]], batch_size=10)
```
