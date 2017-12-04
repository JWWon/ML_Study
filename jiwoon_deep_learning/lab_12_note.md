## RNN
이전 데이터가 다음 데이터에 영향을 끼쳐야 하는 데이터의 경우에 사용.
RNN(Recurrent Neural Network)

* cell은 여러개 처럼 표현하지만 RNN을 구성하는 function은 동일!
```
h(t) = f(W)(h(t-1), x(t))
```
- h(t) : new state
- f(W) : some function with parameters W
- h(t-1) : old state
- x(t) : input vector at some time step

#### Vanila RNN (기초 RNN)
- h(t) : tanh(W(h)(h) * h(t-1) + W(x)(h) * x(t))
- y(t) : W(h)(y) * h(t)

RNN 종류
> one to noe
> one to many
> many to one
> many to many
> Multi-Layer RNN

* hidden_size : 한 노드에서 출력되는 값의 갯수
* sequence_length : (예) 1 회에 입력받는 단어의 갯수, 한 번에 전개되는 시퀀스의 횟수
* batch_size : 한 번에 학습하는 데이터의 갯수
