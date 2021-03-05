---
title: Data Augmentation in Automatic Speech Recognition
description: Recent advancements in augmenting audio data for autometic speech recognition.
author: Heng-Jui Chang
---

# Introduction

End-to-end (E2E) automatic speech recognition (ASR) has been shown to be very powerful by jointly modeling acoustic and linguistic features [1, 2, 3, 4]. However, since deep neural networks require a large amount of data for training, E2E ASR models prone to overfit to small datasets. Moreover, collecting transcribed speech data is difficult than text data, making training E2E ASR even more challenging. For the reasons above, augmenting speech data is a necessary and essential task.

Several successful data augmentation techniques for ASR have been proposed in the past few years [5, 6, 7, 8]. In this article, we introduce three simple yet effective data augmentation methods for E2E ASR, including speed perturbation [5], SpecAugment [6, 7], and SpecSwap [8].

# Speed Perturbation

Here, we introduce the speed perturbation method proposed by Ko et al. [5]. This method stretches or squeezes the duration of audio signals. This operation is done by changing the sample rate of audio waveforms.

Assume the audio signal in the time domain is represented as \(x(t)\). Then, we change the sample rate by a factor of \(\alpha\), that is, changing the signal to \(x(\alpha t)\). This results in the duration of the audio signal is squeezed by a factor of \(\alpha\). If \(\alpha > 1\), the duration of the signal is shortened, and the signal energy is shifted towards higher frequencies.

They set \(\alpha\) to \(0.9\), \(1.0\), and \(1.1\) in their experiments, augmenting the dataset by a factor of three. Overall, the authors showed that this method had improved the WERs of HMM-DNN ASR models by 0.3% to 6.7% in different datasets.


# SpecAugment

Next, we introduce the SpecAugment approach [6]. This method processes spectrograms directly rather than waveforms as compared to speed perturbation. There are three augmentation policies in SpecAugment:
* **Time Warping**:  
  This policy is to warp spectrogram in the time axis randomly. Unlike speed perturbation, this method does not increase or reduce the duration but squeezing and stretching the spectrogram locally.
* **Frequency Masking**:  
  Here, some consecutive frequencies of a spectrogram are randomly masked.
* **Time Masking**:  
  This method is similar to frequency masking but randomly masks consecutive time frames of a spectrogram.

The policies are depicted in the below figure.

<p align="center">
<img src="https://i.imgur.com/NcXdfpk.png" width="500">
</p>

The objective of randomly masking spectrograms is to mimic some scenarios where the speaker makes mistakes or noises in the microphone are present.

Moreover, the authors improved SpecAugment with adaptive time masking [7]. The adaptive time masking policy masks time frames dynamically in response to the length of the spectrogram. That is, audio features with a longer duration will be masked more frequently in the time axis.

Overall, the SpecAugment augmentation policy has shown to be very useful in improving ASR in conventional [9] and end-to-end models.


# SpecSwap

Another approach very similar to SpecAugment that also act on input spectrograms called the SpecSwap [8] is proposed with two augmentation policies:
* **Frequency Swap**:  
  This policy randomly swaps two chunks of frequency bins of a spectrogram.
* **Time Swap**:  
  This policy randomly swaps two chunks of frames of a spectrogram in the time axis.

The SpecSwap augmentation policy also worked well in E2E ASR models but lack of comparisons to SpecAugment.


# Conclusion

Recent advancements in data augmentation policies for E2E ASR are very effective in ASR and other related fields in speech processing [10, 11, 12, 13]. Moreover, these methods are used as a standard approach for state-of-the-art ASR technologies [14]. More research has to be done in the future, discussing the possibility of augmenting data for all speech processing technologies.


# Reference

[1] A. Graves and N. Jaitly, “Towards end-to-end speech recognition with recurrent neural networks,” in ICML, 2014.

[2] A. Graves, “Sequence transduction with recurrent neural networks,” in ICML Workshop on Representation Learning, 2012.

[3] W. Chan, N. Jaitly, Q. V. Le, and O. Vinyals, “Listen, attend and spell: A neural network for large vocabulary conversational speech recognition,” in ICASSP, 2016.

[4] A. Gulati, J. Qin, C.-C. Chiu, N. Parmar, Y. Zhang, J. Yu, W. Han, S. Wang, Z. Zhang, Y. Wu, and R. Pang, “Conformer: Convolution-augmented transformer for speech recognition,” in INTERSPEECH, 2020.

[5] T. Ko, V. Peddinti, D. Povey, and S. Khudanpur, “Audio augmentation for speech recognition,” in INTERSPEECH, 2015.

[6] D. S. Park, W. Chan, Y. Zhang, C.-C. Chiu, B. Zoph, E. D. Cubuk, and Q. V. Le, “SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition,” in INTERSPEECH, 2019.

[7] D. S. Park, Y. Zhang, C.-C. Chiu, Y. Chen, B. Li, W. Chan, Q. V. Le, and Y. Wu, "Specaugment on large scale datasets", in ICASSP, 2020.

[8] X. Song, Z. Wu, Y. Huang, D. Su, and H. Meng, "SpecSwap: A Simple Data Augmentation Method for End-to-End Speech Recognition", in INTERSPEECH, 2020.

[9] W. Zhou, W. Michel, K. Irie, M. Kitza, R. Schlüter, and H. Ney, "The RWTH ASR System for TED-LIUM Release 2: Improving Hybrid HMM with SpecAugment", in ICASSP, 2020.

[10] Y. Hwang, H. Cho, H. Yang, D.-O. Won, I. Oh, and S.-W. Lee, "Mel-spectrogram augmentation for sequence to sequence voice conversion", in arXiv preprint arXiv:2001.01401, 2020.

[11] N. Rossenbach, A. Zeyer, R. Schlüter, and H. Ney, "Generating synthetic audio data for attention-based speech recognition systems", in ICASSP, 2020.

[12] C. Du and K. Yu, "Speaker Augmentation for Low Resource Speech Recognition", in ICASSP, 2020.

[13] P. Bahar, A. Zeyer, R. Schlüter, H. Ney, "On using specaugment for end-to-end speech translation", in arXiv preprint arXiv:1911.08876, 2019.

[14] S. Watanabe, T. Hori, S. Karita, T. Hayashi, J. Nishitoba, et al., "ESPnet: End-to-End Speech Processing Toolkit", in INTERSPEECH, 2018.
