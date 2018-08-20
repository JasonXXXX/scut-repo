## Toturial of Multi-task Learning

#### What is Multi-task Learning?

Given a set of learning tasks, **co-learn all tasks simultaneously**. In other words, the learner optimizes the learning/performance across all of the n tasks through **some shared knowledge**.

#### How Multi-task Learning differs from Transfer Learning

1. Transfer Learning aims to learn well **only** for the target task.
2. In most cases, Transfer Learning only has one source task, in Multi-task Learning's view, means that n = 2.

#### Add-ups

> Life-long Learning
- Broad Definition:  The learner has performed learning on a sequence of tasks, **from t1 to t(n-1)**. When faced with the nth task, it uses the relevant knowledge gained in the past n-1 tasks to help learning for the nth task. <font color=gray>It can be seen that base on this definition, Life-long Learning is very much like Transfer Learning.</font>
- Narrow Definition: learning process that aims to learn well on the future task t(n) **without** seeing any future task data so far. <font color=gray>This means that the system should generate some prior knowledge from the past observed tasks to help new/future task learning without observing any information from the future task t(n). The future task learning **simply uses the knowledge**.</font>