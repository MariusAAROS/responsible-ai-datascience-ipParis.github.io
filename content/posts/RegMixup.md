+++
title = 'RobustAI_RegMixup'
date = 2024-01-24T17:09:36+01:00
draft = false
+++

# RegMixup : Regularizer for robust AI
## Improve accuracy and Out-of-Distribution Robustness

Blog Authors: Marius Ortega
Paper : [RegMixup](https://arxiv.org/abs/2206.14502) by Francesco Pinto, Harry Yang, Ser-Nam Lim, Philip H.S. Torr, Puneet K. Dokania

### Abstract
In this blog post, we will present the paper "RegMixup: Regularizer for robust AI" by Francesco Pinto, Harry Yang, Ser-Nam Lim, Philip H.S. Torr, Puneet K. Dokania. This paper introduces a new regularizer called RegMixup, which is designed to improve the accuracy and out-of-distribution robustness of deep neural networks. The authors show that RegMixup can be used to improve the performance of state-of-the-art models on various datasets, including CIFAR-10, CIFAR-100, and ImageNet. The paper also provides an extensive empirical evaluation of RegMixup, demonstrating its effectiveness in improving the robustness of deep neural networks to out-of-distribution samples.

### Introduction

Most real-world machine algorithm applications are good when it comes to predicting new data following the train distribution. However, they are not robust to out-of-distribution (OOD) samples (i.e. when the test data distribution is different from the train data distribution). This is a major problem in machine learning, as it can lead to catastrophic predictions. 

The question is how to improve the robustness of machine learning algorithms to OOD samples ?
Many researchers have tried such as Liu et al. (2020a, 2020b), Wen et al. (2021), Lakshminarayanan et al. (2017). Even though they have shown some improvements, their approaches use expensive ensemble methods or propose non-trivial modifications of the neural network architecture. What if we could improve the robustness of deep neural networks with respect to OOD samples while utilizing much simpler and cost-effective methods?

The first step toward the method presented in this blog is Mixup proposed by Zang and al (2018). This method is quite good when it comes to dealing with slight perturbations in the data distribution. However, Mixup has the tendency to emphasize difference in labels from very similar samples (high predictive entropy). This is not ideal for OOD samples as the model do not differentiate ID and OOD samples very well.

RegMixup, adds a new layer to Mixup by using Mixup as a regularizer. From there, we will present the theoretical background of the paper, the implementation so as to easily use it in practice.

### 1. Prerequisites

In order to understand the paper, we need to understand what is Empirical and Vicinal Risk Minimization (ERM and VRM) as well as Mixup.

#### 1.1. Empirical Risk Minimization (ERM)

Empirical Risk Minimization is an inference principle which consists in finding the model $\hat{f}$ that minimizes the empirical risk $R_{emp}(\hat{f})$ on the training set. The empirical risk is defined as the average loss over the training set :
$$
R_{emp}(\hat{f}) = \frac{1}{n} \sum_{i=1}^{n} L(\hat{f}(x_i), y_i) \tag{1}
$$
where $L$ is the loss function, $x_i$ is the input, $y_i$ is the label and $n$ is the number of samples in the training set. However, ERM contains a very strong assumption which is that $\hat{f} \approx f$ where $f$ is the true (and unknown) distribution for all points of the dataset. Thereby, if the testing set distribution different even slighly from the training set one, ERM is unable to explain or provide generalization. Vicinal Risk is a way to relax this assumption.

#### 1.2. Vicinal Risk Minimization (VRM)

Vicinal Risk Minimization (VRM) is a generalization of ERM. Instead of having a single distribution estimate $\hat{f}$, VRM uses a set of distributions $\hat{f}_{x_i, y_i}$ for each training sample $(x_i, y_i)$. The goal is to minimize the average loss over the training set, but with respect to the vicinal distribution of each sample.

$$
R_{vrm}(\hat{f}) = \frac{1}{n} \sum_{i=1}^{n} L(\hat{f}_{x_i, y_i}(x_i), y_i) \tag{2}
$$

Consequently, each training point has its own distribution estimate. This is a way to relax the strong assumption of ERM explained above. 

#### 1.3. Mixup

Mixup is a data augmentation technique that generates new samples by mixing pairs of training samples. By doing so, mixup regularizes models to favor simple linear behavior in-between training examples. Experimentally speaking, Mixup has been shown to improve the generalization of deep neural networks, increase their robustness to adversarial attacks, reduce the memorization of corrupt labels as well as stabilize the training of generative adversarial networks.

In essence, Mixup can be though as a learning objective designed for robustness and accountability of the model. Now, let's see how Mixup works.

First, we take two samples $(x_i, y_i)$ and $(x_j, y_j)$ from the training set. Then, we generate a new sample $(\tilde{x}, \tilde{y})$ by taking a convex combination of the two samples with a mixup coefficient $\lambda \sim \text{Beta}(\alpha, \alpha)$ :

$$
\tilde{x} = \lambda x_i + (1 - \lambda) x_j \\
\tilde{y} = \lambda y_i + (1 - \lambda) y_j
$$

We can then define the vicinal distribution of the mixed sample $(\tilde{x}, \tilde{y})$ as :

$$
P_{x_i, y_i} = \mathbb{E}_\lambda[(\delta_{\tilde{x}_i}(x), \delta_{\tilde{y}_i}(y))] \tag{3}
$$

Mixup is an interesting method to consider but it possesses some limitations :
- **Small $\alpha$ issues :** With our setup, $\alpha \approx 1$ encourages $\tilde{x}$ to be perceptually different from $x$. Consequently, training and testing distribution will also grow appart from each other. When $\alpha \ll 1$, the mixup will produce samples close to the initial ones with sharp peaks of 0 and 1 for the value of $\lambda$. What is noticed after cross-validation of alpha is that the best values are $\alpha \approx 0.2$ which is very small. Consequently, the final sample effectively presents only a small perturbation is comparison to the original one while the vicinal distribution exploration space is much larger. We could say that Mixup does not allow to use the full potential of the vicinal distributions of the data.

- **Model underconfidence :** When a neural network is trained with Mixup, it is only exposed to interpolated samples. Consequently, the model learns to predict smoothed labels which is the very root cause of its underconfidence. This results in a high predictive entropy for both ID and OOD samples. 

### 2. RegMixup in theory

Now that we have understood the path that led to RegMixup, we will explore its theoretical background and see how and why it is a good regularizer for robust AI.

While Mixup utilizes data points' vicinal distribution only, RegMixup uses both the vicinal and the empirical one (refering respectively to VRM and ERM). This can seem far-fetched or even counter-intuitive but produces very interesting properties.

$$
P(x, y) = \frac{1}{n} \sum_{i=1}^n \left( \gamma \delta_{x_i}(x) \delta_{y_i}(y) + (1-\gamma) P_{x_i, y_i}(x, y) \right) \tag{4}
$$

Here, $\gamma$ is the hyperparameter controlling the mixup between the empirical and vicinal distribution. In fact, we see that the distribution $P(x, y)$ for RegMixup is a convex combination of the empirical distribution (left term of the addition in equation 4) and the vicinal distribution defined with equations 2 and 3.

From there, we can define a new loss function $\mathcal{L}$ based on the Cross Entropy Loss ($\text{CE}$)

$$
\mathcal{L}(\hat{y}, y) = \text{CE}(p_\theta(\hat{y} | x), y) + \eta \text{CE}(p_\theta(\hat{y} | \tilde{x}), \tilde{y}) \tag{5}
$$

With $\eta \in \mathbb{R}_{+}^*$ being the hyperparameter controlling the importance of the vicinal cross entropy sub-loss and $p_\theta$ the activation function of the model parameterized by $\theta$. In the paper, the value of $\eta$ is set to 1 and its variation seem negligible. Consequently, we will not focus on it in this blog post.

Such a model (equation 4) exhibits properties that lacked in Mixup : 
- **Values of $\alpha$ and underconfidence :** As we explicitly add the empirical distribution to the vicinal one, the ERM term will encourage the model to predict the true labels of the training set while the VRM term, motivated by the interpolation factor $\lambda$ will explore the vicinal distribution space in the much more thorough way than what was possible with mixup. Consequently, the ERM term allows to better predict in-distribution samples while the VRM term with a larger $\alpha$ will allow to better predict OOD samples. This is a very interesting property as it allows to have a model that is both confident and accurate.
- **Prediction entropy :** Through their experiments and observations, researchers found that a cross-validated value of $\alpha$ leads a maximum likelihood estimation having high entropy for ODD samples only. While Mixup demonstrated high entropy for both ID and OOD samples, RegMixup is able to differentiate between the two. This is an highly desirable properties indicating us that RegMixup acts as a regularizer in essense.

As a preliminary conclusion, RegMixup is a very powerful, cost-efficient and simple-to-implement regularizer that allows to improve the robustness and accuracy of deep neural networks for both in-distribution and out-of-distribution samples. In the next section, we will see how to use RegMixup in practice trough a toy example.

### 3. RegMixup in practice (implementation)



### 4. Conclusion

