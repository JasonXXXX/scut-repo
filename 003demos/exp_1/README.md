# 迁移学习实验

### 实验目的

1. 验证迁移学习对识别效果的提升
2. 验证样本污染对迁移学习的影响

### 实验设计

> Source dataset

    mnist 数据集 : 共 6w 个样本，10 个标签

> Target dataset

    usps 数据集 : 分为 training 和 test 数据集，10 个标签

- training

  1000 个样本 (每个类别各有100个样本)

- test

  1000 个样本 (每个类别各有100个样本)

> 实验步骤

1. 在 target training 数据集上训练模型A
2. 在 source + target training 数据集上训练模型B
3. 先在 source 上训练模型C'，再将C'迁移到 target training 上进行 fine-tuning 得到模型C
4. 比较模型 A,B,C 在 target test 数据集上的效果
5. 给 source 数据集加入 attack，重复 1-4

> 评估标准

    模型在test集上的准确率，另外也考虑：模型训练时间、轮次

> 实验结果

1. epoch = 10000, learn_rate = 0.0001

> main result (28x28)

| l_rate | mnist | noise | poison | usps train | usps test | normal | one-step | two-step |
| ------ | ----- | ----- | ------ | ---------- | --------- | ------ | -------- | -------- |
| 1e-4   | 60000 | 0     | 0      | 500        | 800       | 0.791  | 0.884    | 0.924    |
| 1e-4   | 60000 | 20    | 0      | 500        | 800       | 0.791  | 0.806    |          |

> Run code
