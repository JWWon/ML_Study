## Intro about Tensorboard

tensorflow로 작업한 내용들을 visualization을 시키는 'log' 프로그램이라고 생각하
면 편함.

#### Funcations

* visualize TF graph
* Plot quantative matrics
* Show additional data => print 로 하지말고 그래프로 보자 !

#### Steps

1. decide which tensors you want to log

* histogram : useful when logging multiple matrics
* scalar: useful when logging single matrics

```
w2_hist = tf.summary.histogram("weights2", w2)
cost_summ = tf.summarty.scalar("cost", cost)
```

2. Merge all summaries

```
summary = tf.summary.merge_all()
```

3. Create writer and add graph

```
writer = tf.summary.FileWriter('./logs')
writer.add_graph(sess.graph)
```

4. Run summary merge and add_summary

```
s, _ = sess.run([summary, optimizer], feed_dict=feed_dict)
writer.add_summary(s, global_step=global_step)
```

5. Launch TensorBoard (port fowarding)

```
local> $ ssh -L local_port(ex.7007):127.0.0.1:remote_port(ex.6006) username@server.name
server> $ tensorboard -logdir=./logs
```

#### Tips

* name_scope : tensor들이 연결되어 있는 모양을 visualization 해주는 모듈
* Multiple Runs (내가 변경한 값에 따라 학습곡선이 어떻게 변화하는지 비교 )

```
original file: ./log
custom file: ./log_learning_rate_001, ./log_learning_rate_01
```

이후 ./log파일을 실행시키면 custom file들도 자동으로 logging을 함
