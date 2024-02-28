# RegMixup : Regularizer for robust AI
## Improve accuracy and Out-of-Distribution Robustness

Blog Authors: Marius Ortega
Papers Authors: Francesco Pinto, Harry Yang, Ser-Nam Lim, Philip H.S. Torr, Puneet K. Dokania

[Article link](https://arxiv.org/abs/2206.14502)

### Abstract
In this blog post, we will present the paper "RegMixup: Regularizer for robust AI" by Francesco Pinto, Harry Yang, Ser-Nam Lim, Philip H.S. Torr, Puneet K. Dokania. This paper introduces a new regularizer called RegMixup, which is designed to improve the accuracy and out-of-distribution robustness of deep neural networks. The authors show that RegMixup can be used to improve the performance of state-of-the-art models on various datasets, including CIFAR-10, CIFAR-100, and ImageNet. The paper also provides an extensive empirical evaluation of RegMixup, demonstrating its effectiveness in improving the robustness of deep neural networks to out-of-distribution samples.

### Introduction

Most real-world machine algorithm applications are good when it comes to predicting new data following the train distribution. However, they are not robust to out-of-distribution (OOD) samples (i.e. when the test data distribution is different from the train data distribution). This is a major problem in machine learning, as it can lead to catastrophic predictions. 

The question is how to improve the robustness of machine learning algorithms to OOD samples ?
Many researchers have tried such as Liu et al. (2020a, 2020b), Wen et al. (2021), Lakshminarayanan et al. (2017). Even though they have shown some improvements, their approaches use expensive ensemble methods or propose non-trivial modifications of the neural network architecture. What if we could improve the robustness of deep neural networks with respect to OOD samples while utilizing much simpler and cost-effective methods?

The first step toward the method presented in this blog is Mixup proposed by Zang and al (2018). This method is quite good when it comes to dealing with slight perturbations in the data distribution. 

### Prerequisites


