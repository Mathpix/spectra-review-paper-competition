---
title: Advances in machine learning using geometry provide new tools for computational neuroscientist
description: A geometrical perspective proves efficient in developing machine learning tools for computational neuroscience.
author: Pierre Orhan
breaks: false
---

# Review: advances in machine learning using geometry provide new tools for computational neuroscientist


Neuroscience discoveries and machine learning tools have evolved hand to hand to provide a clearer picture of what intelligent computation is all about. 
Computations is more and more understood by the characterization of a trajectory in a state space. The state spaces are naively thought of having large size, scaling with the number of neurons, or of synapses. The instantaneous state of the network is seen as a vector in this Euclidean structure, where each orthogonal axis encodes the value of one variable, for example the activity of a neuron. A major finding of neuroscience was to discover that in many cases the dynamics of these states seems to be confined to lower dimensional surfaces. To understand computations performed by artificial or biological neural network, it is therefore sufficient to track these surfaces and the dynamics of the states restrained on them. Computations in the brain are diverse. Changes in network occur over various range from short term dynamical phenomenon (due to intrinsic properties of neurons) to working memory like computations, to learning dynamics of varying time scales. Different state spaces will therefore be used when studying each of these dynamics. Yet as a common framework, geometrical tools from machine learning are helping to make sense of these dynamics and discover the properties of these surfaces on which neural activity is constrained.  In this review, we highlight how progresses in geometrical machine learning have transitioned to the computational neuroscientist toolbox. Additionally, we take a critical look on the current state of the art of machine learning techniques using a geometrical perspective. We stress how advanced tools from differential geometry might constitute the next theoretical toolbox of a computational neuroscientist, whose aim is to discover how brains compute. 
 
We begin by providing an overview of how topological data analysis, a new field in machine learning, enabled breakthrough in system neuroscience.


# Topology of representations: understanding that neuronal data is very structured.

A set of points may be equipped with a topology, a mathematical scaffold that allows us to describe property of this set, as its continuity and describe a sense of locality between the points. In some cases, the set might be locally similar to a Euclidean space, our familiar finite-dimensional space described by a set of orthogonal basic elements. We might then see the set of points as a surface, a manifold. If the set of points is initially described through a common basis, the manifold is then embedded into a higher dimensional space. The goal of Topological data analysis is to start from this description of the set of points in a high dimensional space and discover the robust topological structures, as manifolds, on which the set is confined. This fast-expanding field has provided powerful algorithm for these purposes [1]. Among them, a notoriously helpful analysis is persistent homology. A ball, a simple mathematical object introduced once a topology is given to a set of points, is progressively grown around each of the data point. Intersecting balls link point together into simplexes. Holes formed in these simplexes are counted and their persistence: the range of radius for which they are present characterize their importance in describing the topology of the data. These holes are sufficient to provide a good understanding of the topology, distinguishing for example ring from torus.
 

Persistent homology was therefore used to question if a population of neurons really represent an external variable. The head direction of the mice can be decoded in its brain from the activity of excitatory cells, notably in the Post-Subiculum and Antero-dorsal nuclei. Using recordings of the activity of such neurons [2] generated a set of population vectors, where each axis represented the mean firing rate of a neuron over short binned temporal windows. Using persistent homology, they proved that these vectors were confined on a one-dimensional manifold, which was effectively a ring. Remarkably, persistent homology could also be applied during sleep, where the animal does not move its head. Again, the set of vectors were restricted to a ring, proving that the dynamical trajectory of these ensembles of neurons was largely independent of the brain state, highlighting that its ring-like organization might be helpful for computations during wake or sleep.
  
Recently, similar results were obtained for 2-dimensional variables, in a network where cells activity is correlated with space. Cells known as grid cells are active over hexagonal grids spanning the 2D environment in which the animal moves. Theoretically, the trajectory of the population vector composed of such cell should be confined on a torus, and it was confirmed using persistent homology applied on the recording of many neurons [3].
  
The head-direction and spatial encoding are variables of low dimension (respectively one and two). Other external variables, as an image projected on our retina, are of much higher dimension and their topology is itself unclear. The topology of the latent neuronal space encoding these inputs is a priori unclear. A low-dimensional encoding could provide a more robust code, as more dimensions can be allowed to filter noise, while a higher-dimensional encoding might be more efficient, notably for generalization (the PDP framework constitutes a historical line of reasoning supporting this idea). Recently, the combined use of calcium-imaging (allowing to record a low-passed filter version of the activity of 10000 neurons) and the development of a variant of PCA: cross-validated principal component analysis answered that the encoding dimensions is high while small enough to ensure smoothness of the manifold on which the population vectors lie [4].
 
Furthermore, it was recently observed that these manifolds constraint the dynamics over long periods (2 year). This measurement was made by finding a common manifold to population vector of recordings made in monkeys over 2 years through the use of another machine learning tool: canonical correlation analysis, CCA [5].
 
The measurement of these topological properties at a population level is an important step as it describes constraints a biological neural network sets on its activity. A new line of research has therefore emerged from comparing these properties with the properties emerging from artificial neural network, either trained or randomly initialized. The community asked if the activity of trained neural network shared similar properties to the activity of experimentally measured neural network.

<p align="center">
<img src="https://user-images.githubusercontent.com/38761938/109557765-d0d3d400-7ad8-11eb-850e-af14fdd64dc2.png" />
  
  
<em> Figure 1: (left) an external variable is encoded on a low-dimensional surface (here a ring). Topological data analysis provide tools to discover this ring. (center) The fact that the data effectively code for an external variable can be obtained by decoding this variable from the recording using different machine learning tools, among which artificial neural networks. (right) The dynamics of the trajectory over the discovered low-dimensional surface is studied to understand how stable the representation is and the computations made from this representation. </em>
</p>

# Comparison of activities: proving that ANN are good models of sensory processing.

The growing capabilities of artificial neural networks to perform specific task on par or better than humans or animals, has opened the possibilities that the computations perform by these vastly different neural networks could share similar principle, notably at the level of the representation to external inputs. Artificial neural networks are trained to perform the same task as humans or animals from which the neural activity is measured. Scientists then ask if the measured brain activity can be predicted from the activity of the trained artificial neural network.  Emerging around 2014 [6,7] this approach has gained further momentum as the trained networks are becoming more and more predictive of the brain activity.
 
In many instances, the predictive performance of the ANN could only be obtained after the training was complete, stressing that the training trajectory was itself important[8].
 
 
Very surprisingly, it was recently observed that in a language task, some networks could become 100% predictive of the brain activity. Crucially, the performance of the network could be predicted solely based on its initial architecture, demonstrating that a scientist may effectively design artificial neural network that are more alike brains, in terms of activity of neurons performing a particular task [9].
 
If some networks become predictive of the activity of human’s neural network, will they remain predictive of the activity of brains from other species? We could focus on insects’ brain and highlight that they solve problems with carefully designed network which we are progressively mapping and explaining despite their complexity. [10,11,12,13] Remarkably, it was recently observed in the Ferret that their response to high-order auditory inputs was different from the one found in humans, lacking the same hierarchical organisation. Since this organization was predicted from the training of ANN , it appears that the mapping between ANN activity and brain activity will be species dependent. To each species a particular architecture of ANN might raise the highest predictive power.

<p align="center">
 <img src="https://user-images.githubusercontent.com/38761938/109557971-1b555080-7ad9-11eb-815d-e137df73673e.png"/>
  
  
<em> Figure 2: Could artificial neural network be abstract model of biological networks? The field has adressed this question by training both animals, humans or ANN on different tasks. If both type of network reach similar performance, we test if the measured activity in ANN is predictive from the activity of the biological network. A positive answer indicate that both network share similar principle of computations. </em>
 </p>

While it is impressive to observe that newest ANN technologies become more and more predictive of recorded activity in real brains, no insights on the computational mechanism of ANN or brains can be gained by simply predicting one with the other. Instead, a theoretical understanding of the dynamical computations produced by neural networks needs to be developed. To this end, several lines of research are being approached both by the computational neuroscience and machine learning communities.

# Dynamics of computations: machine learning provide a set of tools to understand computations over structured representations.

The quest to understanding motor control has enabled the emergence of both experimental tools and the use of recent advances in machine learning to provide key insights into the manner population vectors evolve during a task. 
 
Beginning in 2012, Churchland et al [15] projected the population vectors of neurons recorded in the motor cortex of monkeys. They modified the PCA algorithm, restricting the projection matrix to be anti-symmetric, therefore rotating the projection plane to highlight the rotational dynamics that could be present in the data. From this projection, they indeed observed a characteristic rotation of the population vector prior to reaching. In some sense, this rotation can be understood as the onset of the computation necessary to perform the reaching behaviour. 
  
The understanding of population vector dynamics before reaching was further elucidated in two subsequent papers. Progressively, the use of tool from dynamical systems theory is added on top of these descriptions of the dynamical. Dynamical properties of neuronal trajectories are therefore highlighted. For example, the motor cortex is found to display a signature of smooth dynamical systems by its property of low tangling.: similar initial conditions generate similar neuronal trajectory. [16,17]
More generally, to approach an understanding of the computation performed by the neuronal systems, one may try to parametrize a system of simplified dynamical equation. <strong> An artificial neural network might therefore be used as a parametrization of this dynamical systems, trained to predict the dynamics of the network, or trained on a task </strong>. This line of research was recently clearly reviewed [18]. We will focus on stressing a new tool from the machine learning community that might help its blossoming.
  
RNN have the inconvenience of applying themselves sequentially to the input data. Therefore, if we were to integrate their output, we would be constrained on the choice of time-steps used by our numerical integration, effectively only letting the possibility to use a Euler integration method. If the network is effectively predicting the dynamics of the system (the derivative of a set of dynamical equations), one could hope to reach a better prediction by using more complex integration method. Hopefully, novel tools for incorporating RNN prediction and training in traditional framework of differential equations integration were developed by the machine learning community. At its forefront, one may find the ODE-NET paper of [19], which first described a way to back-propagate through an ODE solver used with a RNN trying to predict the temporal derivative of the system state. This work was further improved to cope with irregularly stamped time series or stochastic differential equations [20,21]
 
Practically, the author would like to highlight that the use of the Julia language is very well adapted to this type of framework, as it presents one of the most complete set of tools for simulating dynamical systems.

# Graph Topology to dynamics: the study of network motif provides an alternative approach to understanding dynamics in neural networks.

To its core, a neural network is composed of nodes (neurons) which interact through their weighted activations. A line of research has therefore taken a reverse approach to ANN training by trying to understand dynamics arising in simple network and then generalizing these properties to larger network. Ideally, the dynamical properties of the system (as its fixed points) should be predictable from the topology of the graph (the motifs) abstracting the network. Remarkably, many brain networks seem to over-represent particular small motifs. It is therefore theoretically tempting to try to understand the dynamics emerging from these motives and their combinations. In practice, while some powerful theoretical results were obtained this line of research has provided more questions than clear answer. It was recently reviewed in [22,23,24,25].

# Dynamics of Learning: theoretical progress in machine learning shine lights on the principle of learning how to dynamically compute over structured representations.

As presented above, computations performed by a network irrespective of learning are becoming clarified through a geometrical perspective. Yet such geometrical perspective of neuronal computations is also shaping our understanding of learning dynamics. Notably, it facilitates understanding if and how the neuronal manifold is modified by learning. The use of brain-computer interface has provided an experimental tool to dynamically monitor the manifold on which brain activity lies during tasks and task learning.  It was first observed that tasks involving dynamics constrained on the initial manifold were more easily learned than tasks which would have required to extend the manifold by allowing novel trajectory out of the initial surface. [26]
 
After technical progress it was observed that if training were long, new trajectory, out of the initial manifold of neuronal activity, could be learned and used to achieve specific tasks. [27] The dynamics of learning are not likely to be exactly the same in abstract neural network and real neural network which have to respect biological constraints. Nonetheless, a hope is to find an abstract summarization of the learning dynamics at the level of the neuronal manifold, which, as we have seen above, can be effectively predicted from the training of artificial neural networks.  
 
Remarkably, the machine learning community is rapidly advancing the understanding of the training of class of artificial neural network. For example, the loss landscape emerging from the optimization of regularized linear autoencoder has been discovered. [28]
  
The nonlinear dynamics of deep linear neural networks was explicitly found as soon as 2014. [29]  Since then, this line of reasoning was modified and extended to start capturing the learning trajectory of recurrent neural network. [30] Very interestingly, the modification of the weight matrix was observed to be itself low-dimensional for recurrent neural network trained on simple task. This observation complements what was observed in vivo with BCI interface. It will be interesting in the future to train RNN on more complex tasks while monitoring the dimensionality of changes of synaptic weights.
  
The progress made using tools from statistical physics to the understanding of network performances has grown recently grown faster. At its core techniques lie the use of tools from random matrix theory (the study of matrix where each element comes from a certain random distribution) and result from probability theory. A recent paper providing new results as well as a good summary of the literature was recently released and focus on an analytical expression of the loss of teacher-student models. [31]
 
More generally, the use of random matrix theory is instrumental in neuroscience. For example, it allows to have an estimate of the number of principal components that accounts for true feature of the data rather than noise. [32]

The theoretical work on learning is vast, we will therefore restrict our analysis of the literature by considering cases were theoretical tools make use of the geometrical properties of the measured neuronal data. In doing so we would like to highlight how differential geometry may serve as an important toolbox for theorist and for experimentalists.


# Differential Geometry: theoretical tools

Differential geometry gathers a set of mathematical tools to study manifolds: how can we define maps from one manifold to another? How can we study the propagation of a wave on a manifold? This toolbox was extensively used to formulate physical theory.  In neuroscience, its importance was stressed as it helped to provide an understanding of the nature of visual hallucinations.  [33]
 
More recently, such tools were used to justify theoretically that computations of deep neural network could transform a complex, ie curved, manifolds (inputs) into a simpler, flat, one [34]
 
In fact, the author observed that when the dynamics approach a state of chaos (where tiny variation in initial condition leads to diverging trajectories), the network could become exponentially expressive. By setting its dynamics near the transition to chaos, the network may compute varied functions. This phenomenon was further elucidated in subsequent works. More recently, a clearer picture of chaotic RNN was obtained [35]. The author discovered the Lyapunov spectra of such network, this spectrum measures the rate of change of diverging trajectory. It can also be used to discover the dimensionality of the attractor on which the dynamic remains constrained.
  
Such attractor are also becoming more and more characterized. Differential geometry is at the heart of the analysis of dynamical systems. In this manner, it can be used to gather a simpler understanding of the computations recurrent neural network perform. In [36], it was observed that the dynamics of various network trained to classify text could be interpreted as evidence accumulation on a low-dimensional manifold, depending on the input dataset. Similarly, the proof that an attractor exists in experimentally measured neuronal dynamics was made studying the head-direction ring [37]. Attractors are supposed to help the brain in various computations. For example it may filter out part of the noise (the dynamic goes back to the ring if the noise comes from an orthogonal direction to the ring, which in high-dimension is highly probable).

Let us go through a simple example to see how differential geometry may shape our understanding of neuronal computations in the future. As we have seen earlier, the mice brain has a representation of its head-direction (HD), a sense of orientation coded on a ring. It also has a sense of space, coded over a torus by the famous grid cells. Grid cells were orginally discovered by the nobel-prize winner May-Britt and Edvard Moser. They discovered them in a particular brain region, the medial entorhinal cortex, which lies in between the circuits coding for the head-direction and the hippocampus where a second code for space emerge (place cells, neurons coding for particular spot in an environment). Some HD neurons are connected to these grid cells, they could therefore help the grid code of space to be computed. Intuitively, a torus is  composed of two rings. Differential geometry therefore provides us an intuition that there is probably another ring to find, potentially coding for speed. It is yet unclear what this second ring might be, or the exact manner in which the two will be combined to produce the grid cells pattern. Numerous models of how grid cells acquire their tuning have been proposed over the year, weither a geometrical approach provides a more intuitive and more predictive explanation of these computations is still an open question. 

<p align="center">
 <img src="https://user-images.githubusercontent.com/38761938/109728133-e9fe8280-7bb5-11eb-9924-aa6a416fee2b.png"/>
  
  
<em> Figure 3: A case study: understanding space computations. We start with the experimental observation that the neuronal activity is constrained on a ring in one brain region and on a torus in another (by persistence homology). We then study their respective correlation to environmental variable, proving that neurons effectively code for head-direction and space (with machine learning decoding). Abstracting to a geometrical perspective, a scientist could then investigate different aspects of the computations over these manifolds. First we could investigate weither the torus manifold emerge from a plastic mechanism or weither two rings are sufficient to generate it. Then we could use ANN which architecture are constrained to use the invariance from these manifolds to investigate further computations that could emerge on them. For example, what if we remove inputs (sleep), or if we suddenly add a strong new input (freezing) to these networks of neurons? What other cells could emerge from the computations of two rings? Eventually we could investigate in silico the type of inputs that would be the best candidate to generate the second ring required to generate the torus. The scientist could then go back to the brain to see if a similar code exist and is effectively used. </em>
 </p>

Beyond the characterization of recurrent neural network as a dynamical system, considering RNN as predictive systems allow to frame the problem of understanding neuronal computations from the point of view of information geometry. Information geometry is a subfield of statistics which focus on describing spaces of random variable. Learning algorithm are then characterized to be optimal or not according to criteria defined over these spaces. For example, random variables for the weight might be constrained to a subspace where they maximise the mutual information with external variable (the extent to which the population activity will be predictive of the input). While the point of view of information theory has been extensively used in neuroscience, the field has yet to develop and use theoretical ideas that emerged from information geometry. In machine learning in general, it was found that gradient descent might not be the optimal optimization algorithm has been stressed and scalable alternative proposed without encountering a large success. [38,39,40,41]

# Differential Geometry: practical tools

The machine learning community is progressively increasing the complexity of its neural network, notably of the convolutional layers (usually composed of small kernels that are convoluted over the input data). Taking into account geometric properties of the data, different groups have suggested variation of the traditional CNN to leverage these properties. For example, network invariant to particular symmetries rotation, or scaling have been implemented and proved to beat traditional CNN over different benchmarks [42,43,44]. 

In comparison, the use of ANN tools to model neuroscience data within different frameworks presented in this review has not made use of these powerful tools. The field has stuck to using the most famous architectures: Long-short term memory cells, traditional convolutional network, or autoencoders. [45,46]
  
This phenomenon can potentially be explained by the need for computational neuroscientist to keep up to date with two fast evolving fields. First the understanding of the importance of geometry in describing neuronal data has emerged as the increasing size of data set (for example the number of neurons per trial) allowed to move from a single neuron to a population point of view. Second, the advances in machine learning are themselves hard to track, both for theoretical and experimental progresses. Therefore, there is a clear need to provide computational neuroscientists with simple tools to test the latest advances in machine learning. To this end Jensen et al [47] have developed a framework for discovering the manifolds on which neural data might lie. 

Yet, much progress remains to be achieved, as both artificial neural networks training or topological data analysis through persistent homology are techniques which remain very time consuming. As an order of magnitude, a persistent homology on an electrophysiological recording of 50 neurons take a few hours to complete.


# Conclusion:

We highlighted that from the discovery of geometrical properties of neuronal coding, a wide range of machine learning tools could be used or developed to facilitate our understanding of brain like computations. First, topological data analysis allowed to discover the shape of some neuronal latent space coding for external variables of low dimensionality. Artificial neural networks activity seems to be increasingly good at predicting brain activity as the machine learning community develops new architectures. These progress help model computations over more complex inputs, as motor control or sound. Furthermore, new architectures, making explicit use of geometrical properties of the neuronal code promise to be well adapted in modelling brain activity.

# References:

[1]  Larry Wasserman, “Topological Data Analysis,” Annual Review of Statistics and Its Application 5, no. 1 (2018): 501–32, https://doi.org/10.1146/annurev-statistics-031017-100045.

[2]  Rishidev Chaudhuri et al., “The Intrinsic Attractor Manifold and Population Dynamics of a Canonical Cognitive Circuit across Waking and Sleep,” Nature Neuroscience 22, no. 9 (September 2019): 1512–20, https://doi.org/10.1038/s41593-019-0460-x.

[3]  Richard J. Gardner et al., “Toroidal Topology of Population Activity in Grid Cells,” BioRxiv, February 25, 2021, 2021.02.25.432776, https://doi.org/10.1101/2021.02.25.432776.

[4]  Carsen Stringer et al., “High-Dimensional Geometry of Population Responses in Visual Cortex,” Nature 571, no. 7765 (July 2019): 361–65, https://doi.org/10.1038/s41586-019-1346-5.

[5]  Juan A. Gallego et al., “Long-Term Stability of Cortical Population Dynamics Underlying Consistent Behavior,” Nature Neuroscience 23, no. 2 (February 2020): 260–70, https://doi.org/10.1038/s41593-019-0555-4.

[6]  Nikolaus Kriegeskorte, “Deep Neural Networks: A New Framework for Modeling Biological Vision and Brain Information Processing,” Annual Review of Vision Science 1, no. 1 (November 18, 2015): 417–46, https://doi.org/10.1146/annurev-vision-082114-035447.

[7]  Daniel L. K. Yamins and James J. DiCarlo, “Using Goal-Driven Deep Learning Models to Understand Sensory Cortex,” Nature Neuroscience 19, no. 3 (March 2016): 356–65, https://doi.org/10.1038/nn.4244.

[8]  Alexander J. E. Kell et al., “A Task-Optimized Neural Network Replicates Human Auditory Behavior, Predicts Brain Responses, and Reveals a Cortical Processing Hierarchy,” Neuron 98, no. 3 (May 2, 2018): 630-644.e16, https://doi.org/10.1016/j.neuron.2018.03.044.

[9]  Martin Schrimpf et al., “The Neural Architecture of Language: Integrative Reverse-Engineering Converges on a Model for Predictive Processing,” BioRxiv, October 9, 2020, 2020.06.26.174482, https://doi.org/10.1101/2020.06.26.174482.

[10]  Eve Marder, Timothy O’Leary, and Sonal Shruti, “Neuromodulation of Circuits with Variable Parameters: Single Neurons and Small Circuits Reveal Principles of State-Dependent and Robust Neuromodulation,” Annual Review of Neuroscience 37, no. 1 (2014): 329–46, https://doi.org/10.1146/annurev-neuro-071013-013958.

[11]  Timothy O’Leary and Eve Marder, “Temperature-Robust Neural Function from Activity-Dependent Ion Channel Regulation,” Current Biology 26, no. 21 (November 7, 2016): 2935–41, https://doi.org/10.1016/j.cub.2016.08.061.

[12]  Katrina MacLeod, Alex Bäcker, and Gilles Laurent, “Who Reads Temporal Information Contained across Synchronized and Oscillatory Spike Trains?,” Nature 395 (November 1, 1998): 693–98, https://doi.org/10.1038/27201.

[13]  G. Laurent, “Dynamical Representation of Odors by Oscillating and Evolving Neural Assemblies,” Trends in Neurosciences 19, no. 11 (November 1996): 489–96, https://doi.org/10.1016/S0166-2236(96)10054-0.

[14]  Kell et al., “A Task-Optimized Neural Network Replicates Human Auditory Behavior, Predicts Brain Responses, and Reveals a Cortical Processing Hierarchy.”

[15]  Mark M. Churchland et al., “Neural Population Dynamics during Reaching,” Nature 487, no. 7405 (July 2012): 51–56, https://doi.org/10.1038/nature11129.

[16]  Matthew T. Kaufman et al., “Cortical Activity in the Null Space: Permitting Preparation without Movement,” Nature Neuroscience 17, no. 3 (March 2014): 440–48, https://doi.org/10.1038/nn.3643.

[17]  Abigail A. Russo et al., “Motor Cortex Embeds Muscle-like Commands in an Untangled Population Response,” Neuron 97, no. 4 (February 2018): 953-966.e8, https://doi.org/10.1016/j.neuron.2018.01.004.

[18]  Saurabh Vyas et al., “Computation Through Neural Population Dynamics,” Annual Review of Neuroscience 43, no. 1 (July 8, 2020): 249–75, https://doi.org/10.1146/annurev-neuro-092619-094115.

[19]  Ricky T. Q. Chen et al., “Neural Ordinary Differential Equations,” ArXiv:1806.07366 [Cs, Stat], December 13, 2019, http://arxiv.org/abs/1806.07366.

[20]  Yulia Rubanova, Ricky T Q Chen, and David Duvenaud, “Latent ODEs for Irregularly-Sampled Time Series,” n.d., 11.

[21]  Xuechen Li et al., “Scalable Gradients for Stochastic Diﬀerential Equations,” n.d., 12.

[22]  Chad Giusti et al., “Clique Topology Reveals Intrinsic Geometric Structure in Neural Correlations,” Proceedings of the National Academy of Sciences 112, no. 44 (November 3, 2015): 13455–60, https://doi.org/10.1073/pnas.1506407112.

[23]  Michael W. Reimann et al., “Cliques of Neurons Bound into Cavities Provide a Missing Link between Structure and Function,” Frontiers in Computational Neuroscience 11 (2017), https://doi.org/10.3389/fncom.2017.00048.

[24]  Carina Curto and Katherine Morrison, “Relating Network Connectivity to Dynamics: Opportunities and Challenges for Theoretical Neuroscience,” Current Opinion in Neurobiology 58 (October 2019): 11–20, https://doi.org/10.1016/j.conb.2019.06.003.

[25]  Carina Curto, Christopher Langdon, and Katherine Morrison, “Robust Motifs of Threshold-Linear Networks,” ArXiv:1902.10270 [q-Bio], December 16, 2019, http://arxiv.org/abs/1902.10270.

[26]  Patrick T. Sadtler et al., “Neural Constraints on Learning,” Nature 512, no. 7515 (August 2014): 423–26, https://doi.org/10.1038/nature13665.

[27]  Emily R. Oby et al., “New Neural Activity Patterns Emerge with Long-Term Learning,” Proceedings of the National Academy of Sciences 116, no. 30 (July 23, 2019): 15210–15, https://doi.org/10.1073/pnas.1820296116.

[28]  Daniel Kunin et al., “Loss Landscapes of Regularized Linear Autoencoders,” ArXiv:1901.08168 [Cs, Stat], May 14, 2019, http://arxiv.org/abs/1901.08168.

[29]  Andrew M. Saxe, James L. McClelland, and Surya Ganguli, “Exact Solutions to the Nonlinear Dynamics of Learning in Deep Linear Neural Networks,” ArXiv:1312.6120 [Cond-Mat, q-Bio, Stat], February 19, 2014, http://arxiv.org/abs/1312.6120.

[30]  Friedrich Schuessler et al., “The Interplay between Randomness and Structure during Learning in RNNs,” Advances in Neural Information Processing Systems 33 (2020), https://proceedings.neurips.cc//paper_files/paper/2020/hash/9ac1382fd8fc4b631594aa135d16ad75-Abstract.html.

[31]  Bruno Loureiro et al., “Capturing the Learning Curves of Generic Features Maps for Realistic Data Sets with a Teacher-Student Model,” ArXiv:2102.08127 [Cond-Mat, Stat], February 16, 2021, http://arxiv.org/abs/2102.08127.

[32]  Adrien Peyrache et al., “Principal Component Analysis of Ensemble Recordings Reveals Cell Assemblies at High Temporal Resolution,” Journal of Computational Neuroscience 29, no. 1 (August 1, 2010): 309–25, https://doi.org/10.1007/s10827-009-0154-6.

[33]  P C Bressloff et al., “Geometric Visual Hallucinations, Euclidean Symmetry and the Functional Architecture of Striate Cortex.,” Philosophical Transactions of the Royal Society of London. Series B 356, no. 1407 (March 29, 2001): 299–330, https://doi.org/10.1098/rstb.2000.0769.

[34]  Ben Poole et al., “Exponential Expressivity in Deep Neural Networks through Transient Chaos,” ArXiv:1606.05340 [Cond-Mat, Stat], June 17, 2016, http://arxiv.org/abs/1606.05340.

[35]  Rainer Engelken, Fred Wolf, and L. F. Abbott, “Lyapunov Spectra of Chaotic Recurrent Neural Networks,” ArXiv:2006.02427 [Nlin, q-Bio], June 3, 2020, http://arxiv.org/abs/2006.02427.

[36]  Kyle Aitken et al., “The Geometry of Integration in Text Classification RNNs,” ArXiv:2010.15114 [Cs, Stat], October 28, 2020, http://arxiv.org/abs/2010.15114.

[37]  Chaudhuri et al., “The Intrinsic Attractor Manifold and Population Dynamics of a Canonical Cognitive Circuit across Waking and Sleep.”

[38]  Gaétan Marceau-Caron and Yann Ollivier, “Natural Langevin Dynamics for Neural Networks,” ArXiv:1712.01076 [Cs, Stat], December 4, 2017, http://arxiv.org/abs/1712.01076.

[39]  Yann Ollivier, “Riemannian Metrics for Neural Networks I: Feedforward Networks,” ArXiv:1303.0818 [Cs, Math], February 3, 2015, http://arxiv.org/abs/1303.0818.

[40]  Yann Ollivier et al., “Information-Geometric Optimization Algorithms: A Unifying Picture via Invariance Principles,” ArXiv:1106.3708 [Math], April 28, 2017, http://arxiv.org/abs/1106.3708.

[41]  Ke Sun and Frank Nielsen, “Lightlike Neuromanifolds, Occam’s Razor and Deep Learning,” ArXiv:1905.11027 [Cs, Stat], May 27, 2019, http://arxiv.org/abs/1905.11027.

[42]  Taco S. Cohen and Max Welling, “Group Equivariant Convolutional Networks,” ArXiv:1602.07576 [Cs, Stat], June 3, 2016, http://arxiv.org/abs/1602.07576.

[43]  Taco S. Cohen et al., “Gauge Equivariant Convolutional Networks and the Icosahedral CNN,” ArXiv:1902.04615 [Cs, Stat], May 13, 2019, http://arxiv.org/abs/1902.04615.

[44]  Erik J. Bekkers, “B-Spline CNNs on Lie Groups,” ArXiv:1909.12057 [Cs, Stat], January 21, 2020, http://arxiv.org/abs/1909.12057.

[45]  Markus Frey et al., “Interpreting Wide-Band Neural Activity Using Convolutional Neural Networks,” BioRxiv, November 26, 2020, 871848, https://doi.org/10.1101/871848.

[46]  David Sussillo et al., “LFADS - Latent Factor Analysis via Dynamical Systems,” ArXiv:1608.06315 [Cs, q-Bio, Stat], August 22, 2016, http://arxiv.org/abs/1608.06315.

[47]  Kristopher T. Jensen et al., “Manifold GPLVMs for Discovering Non-Euclidean Latent Structure in Neural Data,” ArXiv:2006.07429 [Cs, q-Bio, Stat], October 21, 2020, http://arxiv.org/abs/2006.07429.


