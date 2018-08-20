## Toturial of Adversarial Learning

#### What is Adversarial Learning?

In general machine learning, it's considered that training and test data are assumed to be generated from the **same distribution** (although possibly unknown). That brings **unsafety** for a malicious adversary can carefully manipulate the input data exploiting specific vulnerabilities of learning algorithms to compromise the whole system security.

- Key feature: the sample set is **not healthy**.

#### Secure learning in Adversarial Learning

> A number of defense mechanisms against **evasion, poison & privacy attack** have been proposed in the field of Adversarial Machine Learning, including:

- The definition of secure learning algorithms
- The use of multiple classifier systems
- The use of randomization or disinformation to mislead the attacker while acquiring knowledge of the system
- The study of privacy-preserving learning
- Ladder algorithm for Kaggle-style competitions
- Game-theoretic models for Adversarial Machine Learning and data mining
- Sanitizing training data from Adversarial Poisoning Attacks

#### Add-ups

> A truth about Deep Neural Network

Though own very high accuracy, deep neural networks are quite fragile for its easy to be attacked by adversary samples.

> Evasion Attack

- Evasion attacks are the **most prevalent** type of attack that may be encountered in adversarial settings **during system operation (testing/使用阶段)**. For instance, spammers and hackers often attempt to evade detection by obfuscating the content of spam emails and malware code. In the evasion setting, malicious samples are *modified at test time to evade detection*. that is, to be misclassified as legitimate.

- **No influence over the training data is assumed**. A clear example of evasion is image-based spam in which the spam content is embedded within an attached image to evade the textual analysis performed by anti-spam filters. Another example of evasion is given by spoofing attacks against biometric verification systems.

- Attact stratigy
  1. gradient descent  
      由于梯度下降存在局部最优化问题，因为可能造成攻击失败，因此在攻击的优化目标中增加了一个新的成分：密度估计器，由此来找到正样本密度较大的区域，使得通过算法构造出来的对抗样本更好地模仿已知的正样本的特征。
  2. newton's method
  3. BFGS
  4. L-BFGS

> Poison Attack

- Machine learning algorithms are often **re-trained** on data collected during operation to adapt to changes in the underlying data distribution. For instance, intrusion detection systems (IDSs) are often re-trained on a set of samples collected during network operation. 

- Within this scenario (stated above), an attacker may **poison the training data (unlike evasion, training data is accessed!)** by injecting carefully designed samples to eventually compromise the whole learning process. Poisoning may thus be regarded as an adversarial contamination(污染) of the training data. 

- Some attackers from one paper
  - Weak attacker  
      weak attacker is **not aware of the statistical properties** of the training features or labels at all. This attacker simply fakes additional labels with random binary features to poison the training dataset.
  - Strong attacker  
      strong attacker is **aware of the features** we use for training and can **have access to our ground-truth dataset** (which comes from public sources).
  - Sophisticated attacker  
      strongest attacker, named sophisticated attacker, has **full knowledge of our training feature set**. The sophisticated attacker can fully manipulate almost all training features
    
