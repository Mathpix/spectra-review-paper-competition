---
title: GANs
description: GANs and their current impacts 
author: Gabriel Bénédict
breaks: false
---

Generative Adverdsarial Networks (GANs) are relatively trivial to comprehend but they are hard to tune. The following is an attempt at showing all the ways GANs have evolved towards the most iterations.

In addition to the complexities of GANs hyperparametrization at training time, GANs often have downstream tasks that are related to creativity and therefore hard to benchmark. This is admittedly why several influential GAN papers remain unpublished. 

These elements coupled with a general hype around GAN papers, make it difficult for the researcher to choose and hyperparameter-tune the right GAN for the right purpose.

In the following ... (FILL)


## the basic concept

A generative model $G$ creating synthetic samples is paired with a discriminative model $D$ that estimates the probability of that synthetic data to be created by $G$ or to be a sample of the original data.

The police VS robber analogy is mentioned in the original work but the art forger VS art inspector analogy is seemingly more fitting:  (Sharon Zhou analogy)

$G$ must maximize the probability of $D$ making a mistake. $D$ 

> The generative model can be thought of as analogous to a team of counterfeiters, trying to produce fake currency and use it without detection, while the discriminative model is analogous to the police, trying to detect the counterfeit currency. Competition in this game drives both teams to improve their methods until the counterfeits are indistiguishable from the genuine articles. [Goodfellow et. al.](https://arxiv.org/pdf/1406.2661.pdf)




In the original GAN paper, both G and D are Multi Layer Perceptrons (MLP).

GANs can be said to be from the field of adversarial learning, but it is different from adversarial networks (FILL)

## the mathematical formulation

original formulation

minimax game

![\begin{equation}
\min _{G} \max _{D} V(D, G)=\mathbb{E}_{\boldsymbol{x} \sim p_{\mathrm{data}}(\boldsymbol{x})}[\log D(\boldsymbol{x})]+\mathbb{E}_{\boldsymbol{z} \sim p_{\boldsymbol{z}}(\boldsymbol{z})}[\log (1-D(G(\boldsymbol{z})))]
\end{equation}
](https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+%5Cbegin%7Bequation%7D%0A%5Cmin+_%7BG%7D+%5Cmax+_%7BD%7D+V%28D%2C+G%29%3D%5Cmathbb%7BE%7D_%7B%5Cboldsymbol%7Bx%7D+%5Csim+p_%7B%5Cmathrm%7Bdata%7D%7D%28%5Cboldsymbol%7Bx%7D%29%7D%5B%5Clog+D%28%5Cboldsymbol%7Bx%7D%29%5D%2B%5Cmathbb%7BE%7D_%7B%5Cboldsymbol%7Bz%7D+%5Csim+p_%7B%5Cboldsymbol%7Bz%7D%7D%28%5Cboldsymbol%7Bz%7D%29%7D%5B%5Clog+%281-D%28G%28%5Cboldsymbol%7Bz%7D%29%29%29%5D%0A%5Cend%7Bequation%7D%0A)

![\begin{equation}
\begin{array}{l}\mathcal{L}_{D}^{G A N}=\max _{D} \mathbb{E}_{x_{r} \sim p_{r}(x)}\left[\log D\left(x_{r}\right)\right]+\mathbb{E}_{x_{g} \sim p_{g}(x)}\left[\log \left(1-D\left(x_{g}\right)\right)\right] \\ \mathcal{L}_{G}^{G A N}=\min _{G} \mathbb{E}_{x_{g} \sim p_{g}(x)}\left[\log \left(1-D\left(x_{g}\right)\right)\right]\end{array}
\end{equation}
](https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+%5Cbegin%7Bequation%7D%0A%5Cbegin%7Barray%7D%7Bl%7D%5Cmathcal%7BL%7D_%7BD%7D%5E%7BG+A+N%7D%3D%5Cmax+_%7BD%7D+%5Cmathbb%7BE%7D_%7Bx_%7Br%7D+%5Csim+p_%7Br%7D%28x%29%7D%5Cleft%5B%5Clog+D%5Cleft%28x_%7Br%7D%5Cright%29%5Cright%5D%2B%5Cmathbb%7BE%7D_%7Bx_%7Bg%7D+%5Csim+p_%7Bg%7D%28x%29%7D%5Cleft%5B%5Clog+%5Cleft%281-D%5Cleft%28x_%7Bg%7D%5Cright%29%5Cright%29%5Cright%5D+%5C%5C+%5Cmathcal%7BL%7D_%7BG%7D%5E%7BG+A+N%7D%3D%5Cmin+_%7BG%7D+%5Cmathbb%7BE%7D_%7Bx_%7Bg%7D+%5Csim+p_%7Bg%7D%28x%29%7D%5Cleft%5B%5Clog+%5Cleft%281-D%5Cleft%28x_%7Bg%7D%5Cright%29%5Cright%29%5Cright%5D%5Cend%7Barray%7D%0A%5Cend%7Bequation%7D%0A)

That minimax game corresponds to a saddle point optimization problem. While it can be solved with gradient-descent, several fallbacks exist. 

1. *The Helvetica scenario* [1] G is trained too much without updating D. 
2. D is too strong compared to G. It becomes too trivial for G to distinguish real from fake. D's gradients are close to zero and G is not provided with any more training guidance.
3. *mode collapse* [3]. G learns probabilities over limited modes of the original data distribution. It thus produces images from a certain set instead of a diversity of images.

Kullback Leibler, Jensen Shannon. (FILL)

## tips and trics to train a GAN


a little outdated: https://github.com/soumith/ganhacks#authors


[coulomb GAN](https://arxiv.org/pdf/1708.08819.pdf)
[large scale GAN for image synthesis](https://openreview.net/pdf?id=B1xsqj09Fm)

[transGAN](https://arxiv.org/pdf/2102.07074.pdf)

different tasks, different modalities

## GAN variants

In order to tackle the several fallbacks from the vanilla GAN above, variants of GANs have been proposed. They propose to change the loss in D or G, but also often change the architecture of the model.


### L_D

Instead of binary cross-entropy loss, some proposed to use least-square [4], f-divergence [5], hinge loss [6] and finally Wasserstein distance [7, 8]. Due to their convincing creations, Wasserstein GANs became broadly used.

Instead of the original task of distinguishing real from fake data, some proposed class prediction (CATGAN [9] or EBGAN [10] and BEGAN [11] with autoencoders), latent representation (ALI [2], BiGAN [12], InfoGAN [13]).

ACGAN [14] proposes to concatenate class labels to the input to further improve performance with a cross entropy loss.

### L_G



## what if you have less labels: semi / self-supervision

## enhance resolution

## GANs for video

## GANs for music

## build a representation

VAE GANs VS ALI [2]

## References


[1] [Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio: “Generative Adversarial Networks”, 2014](https://arxiv.org/pdf/1406.2661.pdf)

[2] [Vincent Dumoulin, Ishmael Belghazi, Ben Poole, Olivier Mastropietro, Alex Lamb, Martin Arjovsky, Aaron Courville: “Adversarially Learned Inference”, 2016](http://arxiv.org/abs/1606.00704)

[3] [Xin Yi, Ekta Walia, Paul Babyn : "Generative adversarial network in medical imaging: A review", Medical Image Analysis, Volume 58, 2019, 101552, ISSN 1361-8415, https://doi.org/10.1016/j.media.2019.101552](https://www.sciencedirect.com/science/article/pii/S1361841518308430)

[4] Mao, X., Li, Q., Xie, H., Lau, R.Y., Wang, Z., 2016. Least squares generative
adversarial networks.

[5] Nowozin, S., Cseke, B., Tomioka, R., 2016. f-gan: Training generative neural
samplers using variational divergence minimization, in: Advances in Neural
Information Processing Systems, pp. 271–279.

[6] Miyato, T., Kataoka, T., Koyama, M., Yoshida, Y., 2018. Spectral normalization
for generative adversarial networks.

[7] Arjovsky, M., Chintala, S., Bottou, L., 2017. Wasserstein gan.

[8] Gulrajani, I., Ahmed, F., Arjovsky, M., Dumoulin, V., Courville, A., 2017.
Improved training of wasserstein gans

[9] Springenberg, J.T., 2015. Unsupervised and semi-supervised learning with categorical generative adversarial networks.

[10] Zhao, J., Mathieu, M., LeCun, Y., 2016. Energy-based generative adversarial
network.

[11] Berthelot, D., Schumm, T., Metz, L., 2017. Began: boundary equilibrium generative adversarial networks.

[12] Donahue, J., Krähenbühl, P., Darrell, T., 2016. Adversarial feature learning.

[13] Xi Chen, Yan Duan, Rein Houthooft, John Schulman, Ilya Sutskever, Pieter Abbeel: “InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets”, 2016

[14] Odena, A., Olah, C., Shlens, J., 2016. Conditional image synthesis with auxiliary classifier gans.
