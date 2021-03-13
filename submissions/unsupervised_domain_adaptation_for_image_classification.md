---
title: About Unsupervised Domain Adaptation for Image Classification
description: The bulk of machine learning models have a tendancy to rely too strongly to the distribution of the data on which they have been trained. Through this review paper I propose to discuss about ways to design an image classifier able to generalize well on a different but related distribution from its training one.
author: Rony Abecidan
image : https://i.imgur.com/QIMBX6L.png
---

# About Unsupervised Domain Adaptation for Image Classification
---
##### Author : Rony Abecidan

## Introduction 
---
The bulk of machine learning models present one common problem : they have all a tendancy to rely too strongly to the distribution of the data on which they have been trained. As a result if the distribution of the test data is not close to the one of the train data, our models are likely to be inefficient in the test phase. 

In practice, when we are implementing a predictive model that use data which evolve over time, we are constantly constrained to re-train it regularly on fresh data. However collecting and labelling a sufficient amount of fresh data for making our model quickly efficient for our future predictions can be expensive, time consuming and sometimes impossible. That's why it would be interesting to develop a model able to transfer the knowledge it can get from the training distribution to a new distribution not too different from it. This objective is called **domain adaptation** in the literature.

Through this review paper I propose to discuss about domain adaptation for the image classification task in a particular context where we suppose that we have an unlabelled test set coming from a different but close distribution to the training one. Even if it seems tough to generalize on such a context, some research teams managed to propose original solutions presenting strong performances in practice. 

## 1 - Framework & Objective
---
Let's say that we have two picture datasets coming from different but related distributions. 
- One is labelled and will be used for training an image classifier. We will called it the **Source dataset**. 
-  The other is non-labelled and will be used in inference phase. We will called it the **Target dataset**. 

<p align="center">
<img src="https://i.imgur.com/eFFKMno.png" width="500">
</p>

What we want to do is to construct a model that can transfer the knowledge it can get from the source so that it can generalize well on the target. 

<p align="center">
<img src="https://i.imgur.com/QIMBX6L.png" width="800">
</p>

In that precise context we call this objective **Unsupervised Domain Adaptation**.

One common method to achieve that goal consists in finding a way to embed our two datasets in a feature space where their distribution look similar so that, our model can learn on general and transferable features rather than task-specific features.

<p align="center">
<img src="https://i.imgur.com/ewUJYEP.png" width="800">
</p>

Now the big question is : How are we going to find such a relevant embedding ?

## 2 - Achieving unsupervised domain adaptation with adversarial learning
---
In 2015, a research team from Russia <a href='http://proceedings.mlr.press/v37/ganin15.pdf'>[1]</a> proposed an interesting architecture for making a domain adaptative image classifier that you can find below.

<p align="center">
<img src="https://i.imgur.com/wP6q8QZ.png" width="800">
</p>

So that they could extract relevant features from the input images they proposed to use convolutions for instance since it's a very powerful tool for extracting patterns in images. Then, on top of having a part in their architecture dedicated to the classification of the observations based on these extracted features, they proposed also to add a part dedicated to the discrimination between the source and the target based on these same features.

Basically they considered a domain classifier which had for role to identify which observation was coming from the source and which observation was coming from the target based on the extracted features.  What they proposed is simply to maximize the loss of this domain classifier during the learning phase so that, they end up with a feature representation of the input images where it can't separate easily the source from the target images hopefully because they are too close.

Hence, to foster the creation of a feature space relevant for the image classification and the domain adaptation, they proposed to train their model using a loss function of that form :

<p align="center">
<img src="https://i.imgur.com/m3ePw3A.png" width="400">
</p>

and then, the creation of a relevant embedding for an image is done by backpropagating the network using this loss. 

With a more closer look, the adaptation loss can be assimilated in some sense as a similarity measure between our two distributions. When it is minimized it means that the source and target distributions are so close that they are not easily separable. Now it exists numerous ways to assess the proximity between two distributions and we could think about replacing that adaptation loss with a specific metric dedicated to that purpose.

## 3 - Achieving unsupervised domain adaptation with the Maximum Mean Discrepancy

Kernels are famous and powerful tools often used to translate the distance between two points in a space into a similarity score. We could think about the gaussian kernel for instance enabling to achieve that purpose : 

<p align="center">
<img src="https://i.imgur.com/dOKj01I.png" width="400">
</p>

In the same vein, kernels can be also used to assess the similarity between two distributions via what we call the **Maximum Mean Discrepancy**.  It's a bit technical to describe but let's do it simply.

We know that a distribution is completely characterized by its cumulative distribution function and hence we could think about translating the "distance" between two distributions using the following metric : 

<p align="center">
<img src="https://i.imgur.com/t3BDkdI.png" width="700">
</p>

Now, one could think about generalizing this distance using a  "well-chosen" function space

<p align="center">
<img src="https://i.imgur.com/SU7aeCz.png" width="400">
</p>

I precise well-chosen because there is no guarantee that for any  candidate function space, that distance will be minimized precisely when the two distributions are equal. Proposing G={1|(X<t), t real} we end up with the first distance based on the c.d.f. and we know that this is a relevant choice. We could also think about proposing G={exp(Xt), t real} that leads to the supremum of the difference between the moment generating function of our two distributions. When it exists, the moment generating function characterizes completely the distribution to which it is associated and hence this choice is also relevant for assessing the proximity between two distributions.

What is the link with kernels ? Kernels enable to go from a feature space to an Hilbert space of function H. 

<p align="center">
<img src="https://i.imgur.com/0x9k2g1.png" width="700">
</p>

If we consider as our candidate function space for our distance, all the functions within this Hilbert space in the unit sphere, we end up with a metric that we call the **Maximum Mean Discrepancy (MMD)**. According to the kernel that we choose, this metric can be relevant to assessing the proximity of two distributions. It's the case for instance for the gaussian kernel and notably, the MMD between two distributions associated to a gaussian kernel is minimized when they are equal.

More details about the MMD can be found  in [[2]]((https://arxiv.org/abs/1605.09522))

In 2015, a research team from China proposed a domain adaptative image classifier using the Maximum Mean Discrepancy [[3]](https://arxiv.org/pdf/1502.02791.pdf) associated to a convex combination of gaussian kernels. 

Like before, they proposed to use a convolutional neural network to perform the image classification . Intuitively we can feel that the deeper we go in our architecture and the more we have task-specific feature representations. Thus we could expect from the first layers of  such an architecture to extract rather general and transferable features from our input images on contrary to the final layers which should be very task-specific.  Based on that observation, the authors of [[3]](https://arxiv.org/pdf/1502.02791.pdf) proposed to compute the MMD between the final embeddings of the source and the target in their architecture like depicted below.

<p align="center">
<img src="https://i.imgur.com/OFjR3zw.png" width="700">
</p>

Minimizing the MMD between the final embeddings of our two distributions during the training phase enable to foster the creation of final embeddings where they are similar, exactly like the first strategy described in **2**. To achieve that aim, they  backpropagated their network during the learning phase using a loss function of the form : 

<p align="center">
<img src="https://i.imgur.com/ARO521p.png" width="500">
</p>

The choice of a convex combination of gaussian kernels for the MMD is not innocuous. It enables to have a relevant and differentiable metric for measuring the proximity between two distributions.

## 4 - Real world applications
---

The two strategies described before revealed to be rather satisfying for achieving unsupervised domain adaptation on standard benchmarks. In this part, I propose to give some applications where they can be applied to solve real-word problems.

- In healthcare, when we are trying to detect diseases or fractures using scans, we struggle to build a relevant model working for all the hospitals. Why ? Simply because each hospital has its own scanning devices and MRI protocols and hence, in some sense, we can consider that each hospital is producing scans from a specific distribution different from the distributions of all the others. Hence, training a predictive model based on scans from a single hospital will work only for that specific hospital.  However, there is no tremendous variation between the scans distributions of hospitals and it is in fact practicable to use a domain adaptative strategy for solving that problem.  The french start-up [Azmed](https://azmed.co/)  already uses that kind of strategy in order to propose a model able to detect fractures in radiography which works for every radiography center.

- When we are doing topic identification with images, we typically use a dataset made of pictures about a specific field. For instance, we can construct a dataset of movie posters or a dataset of video games posters. Now, a model trained on movie posters may not perform well on video games posters and yet, the elements in the posters may be close or the same. In that kind of situation, using a domain adaptative strategy can help to transfer the knowledge from the movie posters to the video game posters.

- Today, we are familiar with the generation of fake contents that look genuine.  With [thispersondoesnotexist](http://thispersondoesnotexist.com) you can generate as many artifical faces as you want and they look real. Since the generation of realistic fake contents brings a lot of concerns, it could be interesting to design a fake detector for spotting them. However, this is not possible to construct a database with all the kind of artificial contents and hence we can't a priori develop a detector that can detect all the artificial pictures. For example, if you are training a fake detector with fake faces from [thispersondoesnotexist](http://thispersondoesnotexist.com) , it is likely to be inefficient to detect fake cats from  [thiscatdoesnotexist](http://thiscatdoesnotexist.com). However, using a domain adaptative strategy we may solve that problem since human and cat faces share common features.

- Sometimes, if we train a ML model with a certain base and we evaluate it with a noisy version of this base, it is already enough to disturb the model and making it inefficient on this noisy version. This scenario is classical in adversarial learning when we consider that an attacker wants precisely to perturb the prediction of a predictive model. In a case like that, domain adaptation could help to mitigate this attack.


## 5 - Conclusion
---

Classical machine models are too dependent to their training distribution to the point where they can become inefficient with observations coming from a slight variation  of this distribution. To overcome this hurdle it is common to collect and label data from the new distribution on which we want to evaluate the model so that we can re-train it. However, collecting and labelling data is often time consuming and re-training the model can also take some times. In this review we discussed about **domain adaptative** strategies enabling to help an image classifier to generalize its learning to distributions close to its training distribution without fine-tuning.

The classical way to proceed for acheving domain adaptation consists in mapping the source and the target distributions in a space where they look similar so that the model can rely on transferable features for its predictions. We have discussed here about two interesting strategies proposed in 2015 enabling to achieve that goal.  To the extent when it answers to a real and crucal need in the industry, this topic is still current and since then, other strategies have been proposed (see  [[4]](https://arxiv.org/pdf/1603.04779.pdf) for instance).  

At last, we can also mention a classical extension of domain adaptation which is called **multi-domain adaptation** in which we consider several labelled bases as our sources. 

## References 
---
[1] Yaroslav  Ganin  and  Victor  Lempitsky. [Unsupervised  domain  adaptation  by  backpropagation](http://proceedings.mlr.press/v37/ganin15.pdf).  In *International conference on machine learning, pages 1180–1189. PMLR, 2015.*
[2] Krikamol Muandet et al. [Kernel Mean Embedding of Distributions: A Review and Beyond](https://arxiv.org/abs/1605.09522) 2016.
[3] Mingsheng Long, Yue Cao, Jianmin Wang, and Michael Jordan. [Learning transferable features with deep adaptation networks](https://arxiv.org/pdf/1502.02791.pdf).  In *International conference on machine learning, pages 97–105. PMLR, 2015.*
[4] Yanghao Li et al. [Revisiting Batch Normalization For Practical Domain Adaptation](https://arxiv.org/pdf/1603.04779.pdf)  2016
