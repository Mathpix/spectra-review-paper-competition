---
title: GANs
description: GANs and their current impacts 
author: Gabriel Bénédict
breaks: false
---

Oftentimes, only the most graphical (image / video) Artificial Intelligence advances are filtered out to the mainstream media. Generative Adverdsarial Networks (GANs) play an important role in that tip-of-the-iceberg phenomenon because they are most of the time related to images but also because they illustrate the potential of AI for creativity (e.g. https://www.thispersondoesnotexist.com/). Or at least some impression of creativity that is sufficient to blur the line between human and AI creation. Turing test (FILL)

Some of these tip-of-the-iceberg tasks are *next video frame prediction* [26], *image super-resolution* [27], *generative image manipulation* (image editing and creation with minimal brush strokes) [28], *introspective adversarial networks* [29] (photo-editor-like features), *image-to-image translation* [30] (e.g. satellite images to maps, design sketches to clothing, etc.), *photorealistic images from a semantic layout* (e.g. here is grass, here are mountains and a tree is in the middle). 


Generative Adverdsarial Networks (GANs) are relatively trivial to comprehend but they are hard to tune. The following is an attempt at showing all the ways GANs have evolved towards the most iterations.

In addition to the complexities of GANs hyperparametrization at training time, GANs often have downstream tasks that are related to creativity and therefore hard to benchmark. This is admittedly why several influential GAN papers remain unpublished. 

These elements coupled with a general hype around GAN papers, make it difficult for the researcher to choose and hyperparameter-tune the right GAN for the right purpose.

This motivates 

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

minimax game. G aims to maximize the overlap between the distribution of the original data and the ditsribution of the fake data. Applying cross-entropy on point estimates is only an approximation and will be later improved upon with Wasserstein GANs (see below).

![\begin{equation}
\min _{G} \max _{D} V(D, G)=\mathbb{E}_{\boldsymbol{x} \sim p_{\mathrm{data}}(\boldsymbol{x})}[\log D(\boldsymbol{x})]+\mathbb{E}_{\boldsymbol{z} \sim p_{\boldsymbol{z}}(\boldsymbol{z})}[\log (1-D(G(\boldsymbol{z})))]
\end{equation}
](https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+%5Cbegin%7Bequation%7D%0A%5Cmin+_%7BG%7D+%5Cmax+_%7BD%7D+V%28D%2C+G%29%3D%5Cmathbb%7BE%7D_%7B%5Cboldsymbol%7Bx%7D+%5Csim+p_%7B%5Cmathrm%7Bdata%7D%7D%28%5Cboldsymbol%7Bx%7D%29%7D%5B%5Clog+D%28%5Cboldsymbol%7Bx%7D%29%5D%2B%5Cmathbb%7BE%7D_%7B%5Cboldsymbol%7Bz%7D+%5Csim+p_%7B%5Cboldsymbol%7Bz%7D%7D%28%5Cboldsymbol%7Bz%7D%29%7D%5B%5Clog+%281-D%28G%28%5Cboldsymbol%7Bz%7D%29%29%29%5D%0A%5Cend%7Bequation%7D%0A)

That minimax game corresponds to a saddle point optimization problem. The optimum of the game corresponds to the Nash Equilibrium: from this point onwards, none of the two players, would benefit from a change in the player's strategies (see also image below).  

![](https://drek4537l1klr.cloudfront.net/langr/Figures/03fig03_alt.jpg)*(Source: Goodfellow, 2019, www.iangoodfellow.com/slides/2019-05-07.pdf.)*

In practice, the minmax game is often reformulated into two loss functions:

![\begin{equation}
\begin{array}{l}\mathcal{L}_{D}^{G A N}=\max _{D} \mathbb{E}_{x_{r} \sim p_{r}(x)}\left[\log D\left(x_{r}\right)\right]+\mathbb{E}_{x_{g} \sim p_{g}(x)}\left[\log \left(1-D\left(x_{g}\right)\right)\right] \\ \mathcal{L}_{G}^{G A N}=\min _{G} \mathbb{E}_{x_{g} \sim p_{g}(x)}\left[\log \left(1-D\left(x_{g}\right)\right)\right]\end{array}
\end{equation}
](https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+%5Cbegin%7Bequation%7D%0A%5Cbegin%7Barray%7D%7Bl%7D%5Cmathcal%7BL%7D_%7BD%7D%5E%7BG+A+N%7D%3D%5Cmax+_%7BD%7D+%5Cmathbb%7BE%7D_%7Bx_%7Br%7D+%5Csim+p_%7Br%7D%28x%29%7D%5Cleft%5B%5Clog+D%5Cleft%28x_%7Br%7D%5Cright%29%5Cright%5D%2B%5Cmathbb%7BE%7D_%7Bx_%7Bg%7D+%5Csim+p_%7Bg%7D%28x%29%7D%5Cleft%5B%5Clog+%5Cleft%281-D%5Cleft%28x_%7Bg%7D%5Cright%29%5Cright%29%5Cright%5D+%5C%5C+%5Cmathcal%7BL%7D_%7BG%7D%5E%7BG+A+N%7D%3D%5Cmin+_%7BG%7D+%5Cmathbb%7BE%7D_%7Bx_%7Bg%7D+%5Csim+p_%7Bg%7D%28x%29%7D%5Cleft%5B%5Clog+%5Cleft%281-D%5Cleft%28x_%7Bg%7D%5Cright%29%5Cright%29%5Cright%5D%5Cend%7Barray%7D%0A%5Cend%7Bequation%7D%0A)

To make it concrete, below is how one would formulate both losses in Tensorflow Keras ([source](https://github.com/zurutech/gans-from-theory-to-production/blob/master/2.%20GANs%20in%20Tensorflow/2.1.%20Writing%20a%20GAN%20from%20scratch.ipynb)):

```python
def disc_loss(real_output, generated_output):
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    return bce(tf.ones_like(real_output), real_output) + bce(
        tf.zeros_like(generated_output), generated_output
    )
def gen_loss(generated_output):
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    return bce(tf.ones_like(generated_output), generated_output)
```

In practice, the output being a binary indicator (encoding the real / fake nature of the data), the sigmoid activation function is used. Notably, the generator loss function ![min \log (1-D(G(\boldsymbol{z})))](https://render.githubusercontent.com/render/math?math=%5Ctextstyle+min+%5Clog+%281-D%28G%28%5Cboldsymbol%7Bz%7D%29%29%29)
 is formulated in its non-saturating form ![max \log D(G(\boldsymbol{z}))](https://render.githubusercontent.com/render/math?math=%5Ctextstyle+max+%5Clog+D%28G%28%5Cboldsymbol%7Bz%7D%29%29), as Goodfellow et. al. first proposed [1]. Saturation refers to the phenomenon where the optimization surface (sigmoid function here) is steep and estimates too quickly converge towards low or high extremes. At these extremes, the loss function is relatively flat and the gradient is close to zero, thus gradient descent does .


 While it can be solved with gradient-descent, several fallbacks exist. 

1. **The Helvetica scenario** [1] G is trained too much without updating D. 
2. D is too strong compared to G. It becomes too trivial for G to distinguish real from fake. D's gradients are close to zero and G is not provided with any more training guidance.
3. **mode collapse** [3]. G learns probabilities over limited modes of the original data distribution. It thus produces images from a certain set instead of a diversity of images. In the context of creativity, it can be sometimes desirable to prioritize very real-looking images modes over covering the entire distribution of groundtruth examples.
4. training a GAN corresponds to finding a **Nash Equilibrium** of a non-convex game with continuous high dimensional parameters. Gradient descent, as a way to find a minimum in a loss / cost function, is only a rough approximation to the Nash Equilibrium [18, 19]

Kullback Leibler, Jensen Shannon. (FILL)



```python
def build_generator(input_shape):
    """Defines the generator keras.Model.
    Args:
        input_shape: the desired input shape (e.g.: (latent_space_size))
    Returns:
        G: The generator model
    """
    inputs = tf.keras.layers.Input(input_shape)
    net = tf.keras.layers.Dense(units=64, activation=tf.nn.elu, name="fc1")(inputs)
    net = tf.keras.layers.Dense(units=64, activation=tf.nn.elu, name="fc2")(net)
    net = tf.keras.layers.Dense(units=1, name="G")(net)
    G = tf.keras.Model(inputs=inputs, outputs=net)
    return G


def build_disciminator(input_shape):
    """Defines the discriminator keras.Model.
    Args:
        input_shape: the desired input shape (e.g.: (the generator output shape))
    Returns:
        D: the discriminator model
    """
    inputs = tf.keras.layers.Input(input_shape)
    net = tf.keras.layers.Dense(units=32, activation=tf.nn.elu, name="fc1")(inputs)
    net = tf.keras.layers.Dense(units=1, name="D")(net)
    D = tf.keras.Model(inputs=inputs, outputs=net)
    return D
```

## tips and trics to train a GAN


[18] (FILL)

a little outdated: https://github.com/soumith/ganhacks#authors


[coulomb GAN](https://arxiv.org/pdf/1708.08819.pdf)
[large scale GAN for image synthesis](https://openreview.net/pdf?id=B1xsqj09Fm)

[transGAN](https://arxiv.org/pdf/2102.07074.pdf)

different tasks, different modalities

## GAN variants

In order to tackle the several fallbacks from the vanilla GAN above, variants of GANs have been proposed. They propose to change the loss in D or G, but also often change the architecture of the model.


### L_D

Instead of binary cross-entropy loss, some proposed to use least-square [4], f-divergence [5], hinge loss [6] and finally Wasserstein distance [7, 8, 20]. Due to their capacity to correct for imbalances between D and G and for mode collapse and due to its more convincing creations, Wasserstein GANs are now broadly used.

Lipschitz continuity

Instead of the original task of distinguishing real from fake data, some proposed class prediction (CATGAN [9] or EBGAN [10] and BEGAN [11] with autoencoders), latent representation (ALI [2], BiGAN [12], InfoGAN [13]).

ACGAN [14] proposes to concatenate class labels to the input to further improve performance with a cross entropy loss.

### L_G

The original feed-forward neural network for G, can be replaced with a variational autoencoder (VAEGAN [15])

### conditional GAN

Conditional GANs rely on feeding auxiliary information to the network. This can be an image (pix2pix)

### different architectures

The original GAN was a feed-forward neural network. Some proposed to use CNNs instead as they are specialized at modelling images (DCGANs [17])

Game Theory for GANs [38, 39, 40]


cycleGAN 

style transfer (styleGAN [34])

UNIT MUNIT CYCLEGAN

PatchGAN


## what if you have less labels: semi / self-supervision

## enhance resolution

## GANs for video

## GANs for music

GANSynth [25]

## GANs for DNA

[36]

## build a representation

VAE GANs VS ALI [2]

## GANs catching the hype

Given their popularity, GANs are now predominantly coupled with other recent methods. This includes self-supervision (BigBIGAN [32]), attention-based models (SAGAN [33, 39]). These GAN types are among the state-of-the-art in terms of performance. But performance is a non-trivial concept with GANs, as described in the following.

## Evaluating GANs

As hinted at in the introduction, this is the most challenging task, since GANs have most often a creative downstream task (generating fake people, fake music, etc.). In other words, the loss function of GANs tell little about model performance linked to the creative downstream task. For classification, a low loss on a test set suggests an accurate model, but a low loss on D and G is only a sign that training has converged to a saddle point and has stopped.

While asking humans to evaluate *reals* from *fakes* seems like a sensible idea (effectively with HYPE [35]), most common measures are Inception Score (higher is better), Fréchet Inception Distance (lower is better). Recently, Costa et. al. proposed to measure *quality diversity* [21].

Alternatively, an independent critique network can be trained from scratch at GAN evaluation time to compare a holdout set of groundtruth data with the GAN generated data [22, 23, 24].

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

[15] Larsen, A.B.L., Sønderby, S.K., Larochelle, H., Winther, O., 2015. Autoencoding beyond pixels using a learned similarity metric.

[16] Isola, P., Zhu, J.Y., Zhou, T., Efros, A.A., 2016. Image-to-image translation
with conditional adversarial networks.

[17] Radford, A., Metz, L., Chintala, S., 2015. Unsupervised representation learning with deep convolutional generative adversarial networks

[18] Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, Xi Chen: “Improved Techniques for Training GANs”, 2016

[19] Ian J Goodfellow. On distinguishability criteria for estimating generative models. 2014

[20] David Berthelot, Thomas Schumm, Luke Metz: “BEGAN: Boundary Equilibrium Generative Adversarial Networks”, 2017

[21] Victor Costa, Nuno Lourenço, João Correia, Penousal Machado: “Exploring the Evolution of GANs through Quality Diversity”, 2020

[22] Ivo Danihelka, Balaji Lakshminarayanan, Benigno Uria, Daan Wierstra, and Peter Dayan. Comparison of Maximum Likelihood and GAN-based training of Real NVPs, 2017

[23] Daniel Jiwoong Im, He Ma, Graham Taylor, and Kristin Branson. Quantitatively Evaluating GANs With Divergences Proposed for Training, 2018

[24] Ishaan Gulrajani, Colin Raffel, and Luke Metz. Towards GAN Benchmarks Which Require Generalization, ICLR, 2019


[25] Jesse Engel and Kumar Krishna Agrawal and Shuo Chen and Ishaan Gulrajani and Chris Donahue and Adam Roberts, GANSynth: Adversarial Neural Audio Synthesis, 2019, https://openreview.net/pdf?id=H1xQVn09FX

[26] William Lotter, Gabriel Kreiman, David Cox, Deep Predictive Coding Networks for Video Prediction and Unsupervised Learning, 2017

[27] Christian Ledig, Lucas Theis, Ferenc Huszar, Jose Caballero, Andrew Cunningham, Alejandro Acosta, Andrew Aitken, Alykhan Tejani, Johannes Totz, Zehan Wang, Wenzhe Shi, Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network, 2017

[28] Jun-Yan Zhu, Philipp Krähenbühl, Eli Shechtman, Alexei A. Efros, Generative Visual Manipulation on the Natural Image Manifold, 2018

[29] Andrew Brock, Theodore Lim, J.M. Ritchie, Nick Weston, Neural Photo Editing with Introspective Adversarial Networks, 2017

[30] Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, Alexei A. Efros, Image-to-Image Translation with Conditional Adversarial Networks, 2018

[31] Taesung Park, Ming-Yu Liu, Ting-Chun Wang, Jun-Yan Zhu: “Semantic Image Synthesis with Spatially-Adaptive Normalization”, 2019, CVPR 2019

[32] Jeff Donahue, Karen Simonyan: “Large Scale Adversarial Representation Learning”, 2019

[33] Han Zhang, Ian Goodfellow, Dimitris Metaxas, Augustus Odena: “Self-Attention Generative Adversarial Networks”, 2018

[34] Tero Karras, Samuli Laine, Timo Aila: “A Style-Based Generator Architecture for Generative Adversarial Networks”, 2018

[35] Sharon Zhou, Mitchell L. Gordon, Ranjay Krishna, Austin Narcomey, Li Fei-Fei, Michael S. Bernstein, HYPE: A Benchmark for Human eYe Perceptual Evaluation of Generative Models, 2019

[36] Nathan Killoran, Leo J. Lee, Andrew Delong, David Duvenaud, Brendan J. Frey, Nathan Killoran, Leo J. Lee, Andrew Delong, David Duvenaud, Brendan J. Frey, 2017

[37] Samaneh Azadi, Catherine Olsson, Trevor Darrell, Ian Goodfellow, Augustus Odena, Discriminator Rejection Sampling, 2019

[38] Oliehoek, F.A., Savani, R., Gallego-Posada, J., Van der Pol, E., De Jong, E.D. and Gros, R., GANGs: Generative Adversarial Network Games, 2017.

[39] Oliehoek, F., Savani, R., Gallego, J., van der Pol, E. and Gross, R., Beyond Local Nash Equilibria for Adversarial Networks. 2018.

[40] Grnarova, P., Levy, K.Y., Lucchi, A., Hofmann, T. and Krause, A., An Online Learning Approach to Generative Adversarial Networks, 2017. CoRR, Vol abs/1706.03269.
