## Toturial of Transfer Learning
  
#### 1. What is Transfer Learning?

Transfer learning is a "design methodology" that aims to provide or improve model **for target domain** by using the knowledge and parameters from source domain.

> Main part of Transfer Learning

the kernel part of transfer learning lies on: finding the **similarity** between source domain and target domain, making good use of that similarity.

- and then comes the problem: how to measure & use similarity?

    (相似性是核心，度量准则是重要手段)

> Some truth

- 迁移学习与多任务学习以及概念飘移这些问题相关，它不是一个专门的机器学习领域。
- 仅在第一个任务中的深度模型特征是泛化特征的时候，迁移学习才会起作用。
- TL的最大挑战是如何学习出源领域和目标领域共同的knowledge（知识），这个knowledge需要具有很好的领域的适应性。

> How to classify

| learning class | source & target domains | source & target tasks | source & target datas | method |
| - | - | - | - | - |
| Machine Learning | same | same | label/label | |
| Inductive TL | same/relative | ralative | multi-task:label/label;</br>self-taught:unlabel/label | 分类回归 |
| Transductive TL | relative | same | label/unlabel | 分类回归 |
| Unsupervised TL | relative | ralative | unlabel/unlabel | 聚类降维 |

#### 2. Types/parts of Transfer Learning

##### 2.1 Viewpoint 1

> Inductive TL

Feature: a few **labeled** data in the target domain are required as the training data to induce the target predictive function. Bellow are two cases:
- (Most happened) Labeled data in the source domain are available
  *In this case, transfer learning is much like multi-task learning.*

- Labeled data in the source domain are unavailable while unlabeled data in the source domain are available

> Transductive TL

Feature: the source task must be the same as the target task, and part of the **unlabeled** target data are required to be available at training time(in order to obtain the marginal probability for the target data).

> Unsupervised TL

Feature: no labeled data are available in the source/target domain and, source task is not the same as target task.

##### 2.2 Viewpoint 2

> Homogeneoues Transfer Learning

  - Instanced-based
  - Asymmetric feature-based (非对称...)

    **Context feature bias**: causes the conditional distributions between the source and target domains to be different.

    In source feature space: a common feature set, a source specific feature set, a target specific feature set which set to 0;  
    In target feature space: a common feature set, a source specific feature set which set to 0, a target specific feature set.

  - Symmetric feature-based

    The goal is to discover common latent features that have the same marginal distrbution across the source and target domains while maintaining the intrinsic structure of the original domain data.

  - Parameter-based

    Background: each source is able to build a binary learner to predict class. The objective is to build a target binary learner for a new class using minimal labeled target data and knowledge transfered from the multiple source learners.

  - Relational-based
  - Hybrid-based (instance & parameter)

> Heterogeneous Transfer Learning

  - Symmetric feature-based
  - Asymmetric feature-based
  - Improvements

##### 2.3 Viewpoint 3

> 开发模型的方法

  1. 选择**源任务**。你必须选择一个具有丰富数据的相关的预测建模问题，原任务和目标任务的输入数据、输出数据以及从输入数据和输出数据之间的映射中学到的概念之间有某种关系，
  2. 开发源模型。然后，你必须为第一个任务开发一个精巧的模型。这个模型一定要比普通的模型更好，以保证一些特征学习可以被执行。
  3. 重用模型。然后，适用于源任务的模型可以被作为目标任务的学习起点。这可能将会涉及到全部或者部分使用第一个模型，这依赖于所用的建模技术。
  4. 调整模型。模型可以在目标数据集中的输入-输出对上选择性地进行微调，以让它适应目标任务。

> <font color="#df516f">**预训练模型的方法**</font> (在深度学习中较常用)

  1. 选择**源模型**。一个预训练的源模型是从可用模型中挑选出来的。很多研究机构都发布了基于超大数据集的模型，这些都可以作为源模型的备选者。
  2. 重用模型。选择的预训练模型可以作为用于第二个任务的模型的学习起点。这可能涉及到全部或者部分使用与训练模型，取决于所用的模型训练技术。
  3. 调整模型。模型可以在目标数据集中的输入-输出对上选择性地进行微调，以让它适应目标任务。

#### 3. Approaches of Transfer Learning
> Instance-transfer

can be used in: I-TL, T-TL.

**Weights reuse**. Grand different weights to different samples like, similar samples get higher weights while others get lower weights.

> Feature-representation-transfer

can be used in: I-TL, T-TL, **U-TL**.

Feature based transfer learning is much used, the idea goes as: **transformation on features**, means that mapping the features of source domain and target domain to the same space if they are not in the same one or they are not similar.

> Parameter-transfer

can be used in: I-TL.

Build models that **share parameters**. Much used in **neural networks** for the structure of NN can be easily and directly transfered.

> Relational-knowledge-transfer

can be used in: I-TL.

Mining and using relation to process *Analogical Transfer (类比迁移)*.

#### 4. Add-ons

> MMD: maximum mean discrpancy

最大平均差异。最先提出的时候用于双样本检测 (two-sample test) 问题，用于判断两个分布p、q是否相同。  
它的基本假设是：如果对于所有以分布生成的样本空间为输入的函数f，如果两个分布生成的足够多的样本在f上的对应的像的均值都相等，那么那么可以认为这两个分布是同一个分布。  
现在一般用于度量两个分布之间的相似性。
