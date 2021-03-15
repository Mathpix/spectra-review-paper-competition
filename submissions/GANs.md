---
title: GANs
description: GANs and their current impacts 
author: Gabriel Bénédict
breaks: false
---

Generative Adverdsarial Networks (GANs) are relatively trivial to comprehend but they are hard to tune. The following is an attempt at showing all the ways GANs have evolved towards the most iterations.

In addition to the complexities of GANs hyperparametrization at training time, GANs often have downstream tasks that are related to creativity and therefore hard to benchmark. This is admittedly why several influential GAN papers remain unpublished. 

These elements coupled with a general hype around GAN papers, make it difficult for the researcher to choose and hyperparameter-tune the right GAN for the right purpose.


## the basic concept

A generative model $G$ creating synthetic samples is paired with a discriminative model $D$ that estimates the probability of that synthetic data to be created by $G$ or to be a sample of the original data.

The police VS robber analogy is mentioned in the original work but the art forger VS art inspector analogy is seemingly more fitting:  (Sharon Zhou analogy)

$G$ must maximize the probability of $D$ making a mistake. $D$ 

> The generative model can be thought of as analogous to a team of counterfeiters, trying to produce fake currency and use it without detection, while the discriminative model is analogous to the police, trying to detect the counterfeit currency. Competition in this game drives both teams to improve their methods until the counterfeits are indistiguishable from the genuine articles. [Goodfellow et. al.](https://arxiv.org/pdf/1406.2661.pdf)


*The Helvetica scenario*
G is trained too much without updating D. 

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



## tips and trics to train a GAN


a little outdated: https://github.com/soumith/ganhacks#authors


[coulomb GAN](https://arxiv.org/pdf/1708.08819.pdf)
[large scale GAN for image synthesis](https://openreview.net/pdf?id=B1xsqj09Fm)

[transGAN](https://arxiv.org/pdf/2102.07074.pdf)

different tasks, different modalities

## what if you have less labels: semi / self-supervision

## enhance resolution

## GANs for video

## GANs for music

## References

[1] [Vincent Dumoulin, Ishmael Belghazi, Ben Poole, Olivier Mastropietro, Alex Lamb, Martin Arjovsky, Aaron Courville: “Adversarially Learned Inference”, 2016](http://arxiv.org/abs/1606.00704)
