---
title: GANs
description: GANs and their current impacts 
author: Gabriel Bénédict
breaks: false
---

## the basic concept

A generative model $G$ creating synthetic samples is paired with a discriminative model $D$ that estimates the probability of that synthetic data to be created by $G$ or to be a sample of the original data.

art forger VS art inspector (Sharon Zhou analogy)

$G$ must maximize the probability of $D$ making a mistake. $D$ 

> The generative model can be thought of as analogous to a team of counterfeiters, trying to produce fake currency and use it without detection, while the discriminative model is analogous to the police, trying to detect the counterfeit currency. Competition in this game drives both teams to improve their methods until the counterfeits are indistiguishable from the genuine articles. [Goodfellow et. al.](https://arxiv.org/pdf/1406.2661.pdf)


*The Helvetica scenario*
G is trained too much without updating D. 

## the mathematical formulation

![\begin{equation}
\begin{array}{l}\mathcal{L}_{D}^{G A N}=\max _{D} \mathbb{E}_{x_{r} \sim p_{r}(x)}\left[\log D\left(x_{r}\right)\right]+\mathbb{E}_{x_{g} \sim p_{g}(x)}\left[\log \left(1-D\left(x_{g}\right)\right)\right] \\ \mathcal{L}_{G}^{G A N}=\min _{G} \mathbb{E}_{x_{g} \sim p_{g}(x)}\left[\log \left(1-D\left(x_{g}\right)\right)\right]\end{array}
\end{equation}
](https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+%5Cbegin%7Bequation%7D%0A%5Cbegin%7Barray%7D%7Bl%7D%5Cmathcal%7BL%7D_%7BD%7D%5E%7BG+A+N%7D%3D%5Cmax+_%7BD%7D+%5Cmathbb%7BE%7D_%7Bx_%7Br%7D+%5Csim+p_%7Br%7D%28x%29%7D%5Cleft%5B%5Clog+D%5Cleft%28x_%7Br%7D%5Cright%29%5Cright%5D%2B%5Cmathbb%7BE%7D_%7Bx_%7Bg%7D+%5Csim+p_%7Bg%7D%28x%29%7D%5Cleft%5B%5Clog+%5Cleft%281-D%5Cleft%28x_%7Bg%7D%5Cright%29%5Cright%29%5Cright%5D+%5C%5C+%5Cmathcal%7BL%7D_%7BG%7D%5E%7BG+A+N%7D%3D%5Cmin+_%7BG%7D+%5Cmathbb%7BE%7D_%7Bx_%7Bg%7D+%5Csim+p_%7Bg%7D%28x%29%7D%5Cleft%5B%5Clog+%5Cleft%281-D%5Cleft%28x_%7Bg%7D%5Cright%29%5Cright%29%5Cright%5D%5Cend%7Barray%7D%0A%5Cend%7Bequation%7D%0A)


aa

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
