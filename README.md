Throughout the last decade, the practical advancements and the theoretical understanding of deep learning (DL) models and practices has arguably reached a level of maturity such that it is the preferred choice for any practitioner seeking simple yet powerful solutions to solve machine learning-related problems. 
With this tutorial we aim to expose the participants to novel trends in DL for scenarios where quantification of uncertainty matters and we will discuss new and emerging trends in the Bayesian deep learning community.


Description of the tutorial
================================

Decision making processes are ubiquitous in social sciences and engineering and a sound modeling of uncertainty is key to build reliable and trustworthy systems. 
Throughout the last decade, the practical advancements and the theoretical understanding of deep learning models and practices has arguably reached a level of maturity such that it is the preferred choice for any practitioner seeking simple yet powerful solutions to solve machine learning problems.

The dissemination of DL could raise questions on how much we blindly rely on these model's predictions, especially when accuracy is not the only important performance metric and when having sensible uncertainty quantification is a strict system requirement. 
With this tutorial we aim to expose the participants to novel trends in DL for scenarios where quantification of uncertainty matters. 
We will extensively discuss how a proper probabilistic treatment of such complex deep models is possible and feasible. 
We will also highlight new and emerging trends in the Bayesian deep learning community, and we will discuss some important computational aspects.

Overview of the content
=======================

The tutorial will last about 3h30m  and will be divided into three main parts.

Part 1. Motivation for Bayesian inference in modern AI systems 
-------------------------------------------------------------------------

The first part will be dedicated to motivation for a probabilistic treatment in systems powered by deep learing models. 
Following, we will show some fundamental results from Bayesian theory, upon which we will build the content of the next part.

-   Introduction of the speakers and summary of the tutorial

-   The need of reliable models

-   Limitations of loss-trained deep neural networks and the
    motivation for a probabilistic modeling for calibration of
    uncertainty, detection of out-of-distribution data and robustness to
    adversarial examples

-   Bayes' Theorem and the concept of likelihood and
    prior/posterior distributions


Part 2. Bayesian neural networks: inference and modern trends 
--------------------------------------------------------------------------

The second part will be entirely dedicated to the core of the tutorial:
we will present some methodological results that allow us to do tractable Bayesian inference on deep neural networks 
, namely variational inference, Markov-Chain Monte Carlo methods, and other approximations.

-   Optimization as a way to perform inference on Bayesian neural networks (BNNs): an introduction to variational inference

    -   Monte-Carlo Dropout: the simplest way to have BNNs

    -   Formalization of the variational objective (and its
        gradients)

    -   Parameterization of variational inference and recent advancements 

-   Sampling from intractable distributions with MCMC

    -   Introduction to Hamiltonian Monte Carlo (HMC)

    -   Scaling HMC for Bayesian deep learning with stochastic gradients

-   Ensembles and other approximations 

    -   Ensemble as a way to perform Bayesian inference on neural networks

    -   Ensemble as a special case of variational inference

    -   Bayesian model averaging on DNN for scalable inference

    -   Laplace approximation

-   Neural networks are approximation of Gaussian processes: some lessons that can be learn


Part 3. Practical considerations and conclusions 
----------------------------------------------------------

Finally, the last part will be dedicated to some practical
considerations (e.g. how to choose priors). 

-   A problem for today is a solution for tomorrow: encoding
    prior knowledge for Bayesian DNN

-   Calibration of the uncertainty estimation for BNNs

-   Final remarks and take-away message


Material
========

  [Introduction](slides/part1_intro.pdf) --
  [Variational Inference](slides/part2a_variational_inference.pdf) -- 
  [Sampling with MCMC methods](slides/part2b_sampling_bnn.pdf) --
  [Laplace approximation and Ensembles](slides/part2c_laplace_and_ensambles.pdf) --
  [Priors and practical considerations](slides/part3a_practical_considerations.pdf) --
  [Conclusions](slides/part3b_final_ack.pdf) 
  
  [Recordings](https://www.youtube.com/playlist?list=PLgvY-sXF1FemqcqaxxfMrOu6g06riQczc)


Potential target audience
=========================

The audience targeted by this tutorial is represented by practitioners
and scientists willing or interested in using deel learning 
for systems where sound
uncertainty quantification is a requirement. We will assume that the
participants are comfortable with some DL 
 basics, and some 
concepts of optimization (like mini-batch learning and
back-propagation). A bit of experience with Bayesian inference is
suggested but not required to successfully follow the tutorial, as we
will dedicate a good part of the introduction to make sure everyone is
on-par with some basic probability theory results before diving into the
core content of this tutorial.

Motivation and objectives
=========================

Combined with the availability of open source libraries like Tensorflow
and PyTorch, deep learning has quickly gained attraction in other
communities, from cosmology and experimental physics to neuroscience
, and it has
cross-fertilized other computer science fields, such as digital hardware
design, data management systems
and materials science
. Disconcertingly,
näive implementations of DL models are found to be *unreliable* in
some scenarios. A recent analysis of deep CNNs for classification, for example, showed
that the predictions are systematically over-confident. In
practice, this means that there is not a clear way to check whether the
model is "sure" or not about a certain predictions and, as a
consequence, taking informed decisions based on the output of such
models should be carefully considered and properly assessed to avoid
misinterpreting the model behavior. This is an interesting problem from
a methodological research point of view but it is also a concerning
aspect for any possible deployment of DL-based systems, for which a model is
usually trained just once and could be interrogated with any kind of
input data.

A Bayesian approach to deep learning has shown promising results when it
comes to accurate quantification of uncertainty, without compromising on
performance.
The objective of this tutorial is to present a selection these
methodological advancements for applying Bayesian inference techniques
to deep learning models.

Presenters
==========

**Simone Rossi** has been a PhD candidate under the supervision of Prof.
Maurizio Filippone at EURECOM since 2018. He holds a MSc in Computer
Engineering from ENST Telecom Paris (France) and a MSc in Electronic
Engineering from Politecnico di Torino (Italy). His main research has
been focused on novel methods for applying Bayesian inference to deep
models (including Gaussian processes and deep Gaussian processes), with
approximate variational inference techniques and Monte-Carlo methods. 



**Maurizio Filippone** has been an Associate Professor at EURECOM since 2015. 
Prior to that, he carried out some postdoctoral experience in
probabilistic machine learning in the UK (Sheffield, Glasgow and UCL)
and became Assistant Professor at the University of Glasgow, UK in 2011.
Since 2011, he has been teaching classes in probabilistic machine
learning and artificial intelligence at postgraduate level. His research
interests are in the development of practical and scalable methods for
Bayesian inference and for Gaussian processes and deep Gaussian
processes. In the last few years, he has received a prestigious 7-year
fellowship from the AXA Research Fund and a 3-year research grant from
the Agence Nationale de la Recherche to develop novel
probabilistic-based approaches to advance risk modeling in life and
environmental sciences. 

# References


##	Introduction to Variational Inference methods

+		 Jordan et al. (1999). *An Introduction to Variational Methodsfor Graphical Models*. Mach. Learn.
+		 Hoffman et al. (2013). *Stochastic Variational Inference*. JMLR
+		 Ranganath et al. (2014). *Black Box Variational Inference*. AISTATS
+		 Blei et al. (2017). *Variational Inference: A Review for Statisticians*. JASA
	
## Monte-Carlo Dropout for Bayesian Neural Networks and follow-up

+		 Srivastava et al. (2014). *Dropout: A Simple Way to Prevent Neural Networks from Overfitting*, JMLR
+		 Kingma et al. (2015). *Variational Dropout and the Local Reparameterization Trick*. NeurIPS
+		 Gal (2016). *Uncertainty in Deep Learning*. University of Cambridge (PhD Thesis)
+		 Gal and Ghahramani (2016). *Bayesian Convolutional Neural Networks with Bernoulli Approximate Variational Inference*. ICLR Workshop
+		 Gal and Ghahramani (2016). *Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning*. ICML
+		 Kendall and Gal (2017). *What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?*. NeurIPS
+		 Li and Gal (2017). *Dropout Inference in Bayesian Neural Networks with Alpha-divergences*. ICML
+		 Hron et al. (2017). *Variational Gaussian Dropout is not Bayesian*. NeurIPS Workshop
+		 Hron et al (2018). *Variational Bayesian Dropout: Pitfalls and Fixes*. ICML

## Variational Inference for Bayesian Neural Networks

+		 Graves (2011). *Practical Variational Inference for Neural Networks*. NeurIPS
+     Rezende et al. (2014). *Stochastic Backpropagation and Approximate Inference in Deep Generative Models*. ICML 
+		 Hernández-Lobato et al. (2015). *Probabilistic Backpropagation for Scalable Learning of Bayesian Neural Networks*. ICML
+		 Blundell et al. (2015). *Weight Uncertainty in Neural Networks*. ICML
+     Rezende et al. (2015).*Variational Inference with Normalizing Flows*. ICML 
+		 Louizos and Welling (2016). *Structured and Efficient Variational Deep Learning with Matrix Gaussian Posteriors*. ICML
+     Kingma et al. (2016). *Improving Variational Inference with Inverse Autoregressive Flow*.  NeurIPS
+     Liu et al. (2016).    *Stein Variational Gradient Descent: A General Purpose Bayesian Inference Algorithm*.  NeurIPS 
+     Miller et al. (2016). *Variational Boosting: Iteratively Refining Posterior Approximations*.  ICML 
+		 Louizos and Welling (2017). *Multiplicative Normalizing Flows for Variational Bayesian Neural Networks*. ICML
+		 Sun et al. (2017). *Learning Structured Weight Uncertainty in Bayesian Neural Networks*. AISTATS
+		 Khan et al. (2018). *Fast and Scalable Bayesian Deep Learning by Weight-Perturbation in ADAM*. ICML
+		 Rossi et al. (2018). *Good Initializations of Variational Bayes for Deep Models*. ICML
+		 Zhang et al. (2018). *Noisy Natural Gradient as Variational Inference*. ICML
+		 Ghosh et al. (2018). *Structured Variational Learning of Bayesian Neural Networks with Horseshoe Priors*. ICML
+		 Osawa et al. (2019). *Practical Deep Learning with Bayesian Principles*. NeurIPS
+		 Sun et al. (2019). *Functional Variational Bayesian Neural Networks*. ICLR
+		 Farquhar et al. (2020). *Liberty or Depth: Deep Bayesian Neural Nets Do Not Need Complex Weight Posterior Approximations*. NeurIPS
+		 Rossi et al. (2020). *Walsh-Hadamard Variational Inference for Bayesian Deep Learning*. NeurIPS
+     Daxberger et al. (2021). *Bayesian Deep Learning via Subnetwork Inference*. ICML

## Sampling of Bayesian neural network posterior

+     MacKay (1992). *A Practical Bayesian Framework for Backpropagation Networks*. Neural computation.
+     Neal (1996). *Bayesian Learning for Neural Networks*. Springer
+     Neal (2011). *MCMC using Hamiltonian Dynamics*. Hand-book of Markov Chain Monte Carlo
+     Ahn et al. (2012). *Bayesian Posterior Sampling via Stochastic Gradient Fisher Scoring*. ICML
+     Chen et al. (2014). *Stochastic gradient Hamiltonian Monte Carlo*. ICML 
+     Betancourt (2015). *The Fundamental Incompatibility of Scalable Hamiltonian Monte Carlo and Naive Data Subsampling*. ICML
+     Chen et al. (2015). *On the Convergence of Stochastic Gradient MCMC Algorithms with High-Order Integrators*. NeurIPS 
+     Springenberg et al. (2016). *Bayesian Optimization with Robust Bayesian Neural Networks*. NeurIPS
+     Mandt et al. (2017). *Stochastic Gradient Descent as Approximate Bayesian Inference*. JMLR 
+     Zhang et al. (2020). *Amagold: Amortized Metropolis Adjustment for Efficient Stochastic Gradient MCMC*. AISTATS
+     Zhang et al. (2020). *Cyclical stochastic gradient MCMC for Bayesian deep learning*. ICLR 
+     Cobb et al. (2021). *Scaling Hamiltonian Monte Carlo Inference for Bayesian Neural Networks with Symmetric Splitting*. UAI
+     Franzese et al. (2021). *A Unified View of Stochastic Hamiltonian Sampling*. arXiv
+     Izmailov et al. (2021). *What Are Bayesian Neural Network Posteriors Really Like?* ICML



## Laplace approximation

+		 MacKay (1991). *Bayesian Model Comparison and Backprop Nets*. NeurIPS
+		 MacKay (1991). *A Practical Bayesian Framework for Backpropagation Networks*. Neural comput.
+		 Williams and Barber (1998). *Bayesian classification with Gaussian processes*. IEEE PAMI
+		 MacKay (1998). *Choice of Basis for Laplace Approximation*. Machine Learning
+		 Schraudolph (2002). *Fast curvature matrix-vector products for second-order gradient descent*. Neural Comput.
+		 Kuss and Rasmussen (2005). *Assessing Approximate Inference for Binary Gaussian Process Classification*. JMLR
+		 Nickisch and Rasmussen (2008). *Approximations for Binary Gaussian Process Classification*. JMLR
+		 Martens et al. (2015). *Optimizing Neural Networks with Kronecker-factored Approximate Curvature*. ICML
+		 Botev et al. (2017). *Practical Gauss-Newton Optimisation for Deep Learning*. ICML
+		 Ritter et al. (2018). *A Scalable Laplace Approximation for Neural Networks*. ICLR
+		 Kunstner et al. (2019). *Limitations of the Empirical Fisher Approximation for Natural Gradient Descent*. NeurIPS
+		 Dangel et al. (2020). *BackPACK: Packing more into Backprop*. ICLR
+		 Immer et al. (2021). *Improving predictions of Bayesian neural nets via local linearization*. AISTATS
+		 Immer et al. (2021). *Scalable Marginal Likelihood Estimation for Model Selection in Deep Learning*. ICML
+		 Kristiadi et al. (2021). *Learnable Uncertainty under Laplace Approximations*. UAI


## Ensemble methods

+		 Newton and Raftery (1994). *Approximate Bayesian Inference with the Weighted Likelihood Bootstrap*. JRSS - Series B
+		 Lakshminarayanan et al. (2017). *Simple and scalable predictive uncertainty estimation using deep ensembles*. NeurIPS
+		 Pearce et al. (2018). *Bayesian Inference with Anchored Ensembles of Neural Networks, and Application to Reinforcement Learning*. ICML Workshop
+		 Pearce et al. (2018). *Bayesian neural network ensembles*. NeurIPS Workshop
+		 Garipov et al. (2018). *Loss Surfaces, Mode Connectivity, and Fast Ensembling of DNNs*. NeurIPS
+		 Fort et al. (2019). *Deep Ensembles: A Loss Landscape Perspective*. NeurIPS BDL Workshop
+		 Milios et al. (2020). *Parametric Bootstrap Ensembles as Variational Inference*. AABI
+		 He at al. (2020). *Bayesian Deep Ensembles via the Neural Tangent Kernel*. NeurIPS


## Infinite-limit Neural Networks

+		 Rasmussen and Williams (2006). *Gaussian Processes for Machine Learning*, MIT Press
+		 Damianou and Lawrence (2013). *Deep Gaussian Processes*. AISTATS
+		 Cutajar et al. (2017). *Random Features Expansions for Deep Gaussian Processes*. ICML
+		 Jacot et al. (2018). *Neural Tangent Kernel: Convergence and Generalization in Neural Networks*. NeurIPS
+		 Matthews et al. (2018). *Gaussian Process Behaviour in Wide Deep Neural Networks*. ICLR
+		 Lee et al. (2018). *Deep Neural Networks as Gaussian Processes*. ICLR
+		 Novak et al. (2019). *Bayesian Deep Convolutional Networks with Many Channels are Gaussian Processes*. ICLR
+		 Garriga-Alonso et al. (2019). *Deep Convolutional Networks as shallow Gaussian Processes*. ICLR
+		 Yang (2019). *Wide Feedforward or Recurrent Neural Networks of Any Architecture are Gaussian Processes*. NeurIPS
+		 Khan et al. (2019). *Approximate Inference Turns Deep Networks into Gaussian Processes*. NeurIPS
+		 Lee et al. (2019). *Wide Neural Networks of Any Depth Evolve as Linear Models under Gradient Descent*. NeurIPS
