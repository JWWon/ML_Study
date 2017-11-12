## Tips for using Tensorflow
### Modules

#### Multiply vs Matmul
Matmul: 행렬의 곱
Multiply: Broadcasting(형변환)을 거친 후 모양이 같아진 행렬의 원소끼리의 곱을 진행
* Broadcasting은 연산을 진행하려는 행렬들의 shape이 같지 않은 경우, 자동적으로 실행됨

#### reduce_mean
평균을 구하는데 사용하는 모듈!
* 인자의 변수형에 따라 결과값이 나오므로, 되도록 float형을 사용하도록 하자
* axis 값을 명시하지 않으면 행렬의 모든 값을 평균을 냄, axis=x 로 명시하면 해당 축의 평균값을 계산 (1차원 낮아짐)
* reduce_sum 도 동일하게 동작, 평균 대신 합을 계산

#### argmax
해당 축에서 가장 큰 값의 위치(index)를 찾을 때 사용하는 모듈!
따라서 axis=x 를 입력해줘야 함

#### reshape
기존 array를 원하는 shpae대로 변형시킬 수 있는 모듈!! 매우 중요!
* ex) shape[-1, 3] : axis=1은 마음대로, axis=0은 데이터를 3개씩 쌓기
```
result = array([0, 1, 2],
	       [3, 4, 5],
	       [6, 7, 8])
```
* ex) shape[-1, 1, 3] : axis=2는 마음대로, axis=1은 1개만, axis=0은 3개씩 쌓기
```
result = array([[0, 1, 2]],
	       [[3, 4, 5]],
	       [[6, 7, 8]])
```
* 제일 안쪽 데이터는 그대로 두는 것이 일반적, 바깥부분 shape를 주로 조정
* squeeze : reshape의 한 종류, javascript에서 .flatten 과 같은 역할을 함
* expand : squeeze의 반대, 한 array에 있던 원소들을 각각 1차원 높임

#### one_hot
array 안의 원소들을 0 또는 1로 표현하고 싶을 때 사용하는 모듈!
* ex) tf.one_hot([[0], [1], [2], [0]], depth=3).eval() //depth: one_hot 으로 변환하는데 사용할 array의 depth
```
array([[1., 0., 0.]], // [0]과 동일
      [[0., 1., 0.]], // [1]과 동일
      [[0., 0., 1.]], // [2]와 동일
      [[1., 0., 0.]])
```
* one_hot 모듈이 생성한 array 모양이 마음에 안들면 -> reshape하면 됨!

#### cast
javascript의 parseInt(), parseFloat()와 동일한 역할

#### stack
여러 array들을 한 array로 묶는 모듈
axis=x 를 지정하면 여러 array들의 해당 축을 기준으로 새롭게 묶을 수 있다

#### ones_like, zeros_like
어떤 array와 동일한 shape을 가지고 있지만, 1 또는 0으로 가득 찬 array를 생성할 때 사용하는 모듈!

#### zip
```
for x, y in zip([1, 2, 3], [4, 5, 6]):
	print(x, y)
```
```
1 4
2 5
3 6
```
