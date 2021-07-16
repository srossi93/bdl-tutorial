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

# References

M. Abadi, et al. 
 TensorFlow: Large-Scale Machine Learning on Heterogeneous Systems,
  2015.

R. Acciarri et al.
 Convolutional neural networks applied to neutrino events in a liquid
  argon time projection chamber.
  Journal of Instrumentation, 12(3), 2017.

A. Aurisano, A. Radovic, D. Rocco, A. Himmel, M. D. Messier, E. Niner,
  G. Pawloski, F. Psihas, A. Sousa, and P. Vahle.
 A convolutional neural network neutrino event classifier.
  Journal of Instrumentation, 11(9), 2016.

G. Bellec, D. Salaj, A. Subramoney, R. Legenstein, and W. Maass.
 Long short-term memory and learning-to-learn in networks of spiking
  neurons.
 In  Advances in Neural Information Processing Systems, pages
  787--797, 2018.

C. Blundell, J. Cornebise, K. Kavukcuoglu, and D. Wierstra.
 Weight Uncertainty in Neural Networks.
 may 2015.

S. Dieleman, K. W. Willett, and J. Dambre.
 Rotation-invariant convolutional neural networks for galaxy
  morphology prediction.
  Monthly Notices of the Royal Astronomical Society,
  450(2):1441--1459, 2015.

L. Fussell and B. Moews.
 Forging new worlds: High-resolution synthetic galaxies with chained
  generative adversarial networks.
  Monthly Notices of the Royal Astronomical Society,
  485(3):3215--3223, 2019.

A. Graves.
 Practical Variational Inference for Neural Networks.
  Nips, pages 1--9, 2011.

C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger.
 On Calibration of Modern Neural Networks.
 In D. Precup and Y. W. Teh, editors,  Proceedings of the 34th
  International Conference on Machine Learning, volume 70 of  Proceedings
  of Machine Learning Research, pages 1321--1330, International Convention
  Centre, Sydney, Australia, aug 2017. PMLR.

Y. Kim, C. Yang, Y. Kim, G. X. Gu, and S. Ryu.
 Designing an Adhesive Pillar Shape with Deep Learning-Based
  Optimization.
  ACS Applied Materials and Interfaces, 12(21):24458--24465, may
  2020.

T. Kraska, A. Beutel, E. H. Chi, J. Dean, and N. Polyzotis.
 The Case for Learned Index Structures.
 In  Proceedings of the 2018 International Conference on
  Management of Data, SIGMOD '18, pages 489--504, New York, NY, USA, 2018.
  Association for Computing Machinery.

J. Kwon and L. P. Carloni.
 Transfer learning for design-space exploration with high-level
  synthesis.
 In  MLCAD 2020 - Proceedings of the 2020 ACM/IEEE Workshop on
  Machine Learning for CAD, 2020.

F. Lanusse, P. Melchior, and F. Moolekamp.
 Hybrid physical-deep learning model for astronomical inverse
  problems.
  arXiv, 2019.

G. Li, X. Zhou, S. Li, and B. Gao.
 QTune: A Query-Aware Database Tuning System with Deep Reinforcement
  Learning.
  Proceedings of the VLDB Endowment, 12(12):2118--2130, aug 2019.

H. Liu, M. Xu, Z. Yu, V. Corvinelli, and C. Zuzarte.
 Cardinality Estimation Using Neural Networks.
 In  Proceedings of the 25th Annual International Conference on
  Computer Science and Software Engineering, CASCON '15, pages 53--59, USA,
  2015. IBM Corp.

Q. Liu and D. Wang.
 Stein Variational Gradient Descent: A General Purpose Bayesian
  Inference Algorithm.
 In D. D. Lee, M. Sugiyama, U. V. Luxburg, I. Guyon, and R. Garnett,
  editors,  Advances in Neural Information Processing Systems 29, pages
  2378--2386. Curran Associates, Inc., 2016.

Y. C. Lu, S. S. Kiran Pentapati, L. Zhu, K. Samadi, and S. K. Lim.
 TP-GNN: A graph neural network framework for tier partitioning in
  monolithic 3D ICs.
 In  Proceedings - Design Automation Conference, 2020.

R. Marcus and O. Papaemmanouil.
 Plan-Structured Deep Neural Network Models for Query Performance
  Prediction.
 In  Proceedings of the VLDB Endowment, volume 12, pages
  1733--1746, 2019.

R. M. Neal.
  Bayesian Learning for Neural Networks (Lecture Notes in
  Statistics).
 Springer, 1 edition, aug 1996.

R. M. Neal.
 MCMC using Hamiltonian dynamics.
 in Handbook of Markov Chain Monte Carlo (eds S. Brooks, A. Gelman, G.
  Jones, XL Meng). Chapman and Hall/CRC Press, 2010.

K. Osawa, S. Swaroop, M. E. E. Khan, A. Jain, R. Eschenhagen, R. E. Turner, and
  R. Yokota.
 Practical Deep Learning with Bayesian Principles.
 In  Advances in Neural Information Processing Systems 32, pages
  4287--4299. Curran Associates, Inc., 2019.

Y. Ovadia, E. Fertig, J. Ren, Z. Nado, D. Sculley, S. Nowozin, J. Dillon,
  B. Lakshminarayanan, and J. Snoek.
 Can you trust your model's uncertainty? Evaluating predictive
  uncertainty under dataset shift.
 In  Advances in Neural Information Processing Systems,
  volume 32, pages 13991--14002. Curran Associates, Inc., 2019.

A. Paszke, S. Gross, F. Massa, A. Lerer, J. Bradbury, G. Chanan, T. Killeen,
  Z. Lin, N. Gimelshein, L. Antiga, A. Desmaison, A. Kopf, E. Yang, Z. DeVito,
  M. Raison, A. Tejani, S. Chilamkurthy, B. Steiner, L. Fang, J. Bai, and
  S. Chintala.
 PyTorch: An Imperative Style, High-Performance Deep Learning
  Library.
 In  Advances in Neural Information Processing Systems,
  volume 32, pages 8026--8037. Curran Associates, Inc., 2019.

D. Rezende and S. Mohamed.
 Variational Inference with Normalizing Flows.
 In  Proceedings of the 32nd International Conference on Machine
  Learning, volume 37 of  Proceedings of Machine Learning Research, pages
  1530--1538, Lille, France, 2015. PMLR.

B. H. Tran, S. Rossi, D. Milios, and M. Filippone.
 All you need is a good functional prior for Bayesian deep learning,
  2020.

J. X. Wang, Z. Kurth-Nelson, D. Kumaran, D. Tirumala, H. Soyer, J. Z. Leibo,
  D. Hassabis, and M. Botvinick.
 Prefrontal cortex as a meta-reinforcement learning system.
  Nature Neuroscience, 21(6):860--868, 2018.

F. Wenzel, K. Roth, B. S. Veeling, J. \'Swi\catkowski, L. Tran,
  S. Mandt, J. Snoek, T. Salimans, R. Jenatton, and S. Nowozin.
 How good is the Bayes posterior in deep neural networks really?
 In  arXiv, 2020.

A. G. Wilson.
 The case for Bayesian deep learning.
  arXiv preprint arXiv:2001.10995, 2020.

J. K. Wilt, C. Yang, and G. X. Gu.
 Accelerating Auxetic Metamaterial Design with Deep Learning.
  Advanced Engineering Materials, 22(5):1901266, may 2020.

D. L. Yamins and J. J. DiCarlo.
 Using goal-driven deep learning models to understand sensory
  cortex.
 In  Nature Neuroscience, volume 19, pages 356--365, 2016.

A. M. Zador.
 A critique of pure learning and what artificial neural networks can
  learn from animal brains.
 In  Nature Communications, volume 10, 2019.

F. Zhang, B. Shao, G. Xu, B. Yang, Z. Yang, Z. Qin, K. Ren, and Z. Yang.
 From homogeneous to heterogeneous: Leveraging deep learning based
  power analysis across devices.
 In  Proceedings - Design Automation Conference, 2020.

G. Zhang, S. Sun, D. Duvenaud, and R. Grosse.
 Noisy Natural Gradient as Variational Inference.
 In J. Dy and A. Krause, editors,  Proceedings of the 35th
  International Conference on Machine Learning, volume 80 of  Proceedings
  of Machine Learning Research, pages 5852--5861,
  Stockholm Sweden, 2018. PMLR.

Z. Zhang and G. X. Gu.
 Finite-Element-Based Deep-Learning Model for Deformation Behavior of
  Digital Materials.
  Advanced Theory and Simulations, 3(7):2000031, jul 2020.

G. X. G. Zhizhou Zhang.
 Physics-informed deep learning for digital materials.
  Theoretical & Applied Mechanics Letters, 11:1, 2021.

