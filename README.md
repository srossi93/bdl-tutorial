Throughout the last decade, the practical advancements and the theoretical understanding of deep learning (DL) models and practices has arguably reached a level of maturity such that it is the preferred choice for any practitioner seeking simple yet powerful solutions to solve machine learning-related problems. 
With this tutorial we aim to expose the participants to novel trends in DL for scenarios where quantification of uncertainty matters and we will discuss new and emerging trends in the Bayesian deep learning community.


Long description of the tutorial
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

    -   Parameterization of variational inference and recent advancements (including normalizing flows and particle-based variational inference)

-   Sampling from intractable distributions with MCMC

    -   Introduction to Hamiltonian Monte Carlo (HMC)

    -   Scaling HMC for Bayesian deep learning with stochastic gradients:

-   Ensembles and other approximations 

    -   Ensemble as a way to perform Bayesian inference on neural networks

    -   Ensemble as a special case of variational inference

    -   Bayesian model averaging on DNN for scalable inference

    -   Laplace approximation

-   Neural networks are approximation of Gaussian processes: some lessons that can be learn


Part 3. Practical considerations and conclusions 
----------------------------------------------------------

Finally, the last part will be dedicated to some practical
considerations (e.g. how to choose priors). And this part will be
concluded with a discussion on computational complexity of Bayesian
inference, with a focus on heterogeneous computing.

-   A problem for today is a solution for tomorrow: encoding
    prior knowledge for Bayesian DNN

-   An analysis of the computational divide: challenges and
    opportunities of heterogeneous computing for Bayesian inference

-   Software and libraries for implementing Bayesian
    inference for deep learning models

-   Final remarks and take-away message


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

