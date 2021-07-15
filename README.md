
Throughout the last decade, the practical advancements and the
theoretical understanding of [DL]{acronym-label="DL"
acronym-form="singular+short"} models and practices has arguably reached
a level of maturity such that it is the preferred choice for any
practitioner seeking simple yet powerful solutions to solve
[ML]{acronym-label="ML" acronym-form="singular+short"}-related problems.
With this tutorial we aim to expose the participants to novel trends in
[DL]{acronym-label="DL" acronym-form="singular+short"} for scenarios
where quantification of uncertainty matters and we will discuss new and
emerging trends in the Bayesian deep learning community.

Long description of the tutorial
================================

Decision making processes are ubiquitous in social sciences and
engineering and a sound modeling of uncertainty is key to build reliable
and trustworthy systems. Throughout the last decade, the practical
advancements and the theoretical understanding of
[DL]{acronym-label="DL" acronym-form="singular+short"} models and
practices has arguably reached a level of maturity such that it is the
preferred choice for any practitioner seeking simple yet powerful
solutions to solve [ML]{acronym-label="ML"
acronym-form="singular+short"}-related problems.

The dissemination of [DL]{acronym-label="DL"
acronym-form="singular+short"} could raise questions on how much we
blindly rely on these model's predictions, especially when accuracy is
not the only important performance metric and when having sensible
uncertainty quantification is a strict system requirement. With this
tutorial we aim to expose the participants to novel trends in
[DL]{acronym-label="DL" acronym-form="singular+short"} for scenarios
where quantification of uncertainty matters. We will extensively discuss
how a proper probabilistic treatment of such complex deep models is
possible and feasible. We will also highlight new and emerging trends in
the Bayesian deep learning community, and we will discuss some important
computational aspects.

Overview of the content
=======================

The tutorial will last 1/2 day (two time slots) and will be divided into
three main parts.

Part 1. \[20 min\] Motivation for Bayesian inference in modern AI systems {#part-1.-20-min-motivation-for-bayesian-inference-in-modern-ai-systems .unnumbered .unnumbered}
-------------------------------------------------------------------------

The first part will be dedicated to motivation for a probabilistic
treatment in systems powered by [DL]{acronym-label="DL"
acronym-form="singular+short"} models. Following, we will show some
fundamental results from Bayesian theory, upon which we will build the
content of the next part.

-   Introduction of the speakers and summary of the tutorial

-   \[5 min\] The need of reliable models

-   \[5 min\] Limitations of loss-trained deep neural networks and the
    motivation for a probabilistic modeling for calibration of
    uncertainty, detection of out-of-distribution data and robustness to
    adversarial examples

-   \[10 min\] Bayes' Theorem and the concept of likelihood and
    prior/posterior distributions

Relevant literature: [@Guo17; @Osawa2019]

Part 2. \[2h10 min\] Bayesian neural networks: inference and modern trends {#part-2.-2h10-min-bayesian-neural-networks-inference-and-modern-trends .unnumbered .unnumbered}
--------------------------------------------------------------------------

The second part will be entirely dedicated to the core of the tutorial:
we will present some methodological results that allow us to do
tractable Bayesian inference on [DNN]{acronym-label="DNN"
acronym-form="singular+short"}, namely [VI]{acronym-label="VI"
acronym-form="singular+short"}, [MCMC]{acronym-label="MCMC"
acronym-form="singular+short"} methods and ensembles.

-   \[55 min\] Optimization as a way to perform inference on
    [BNNs]{acronym-label="BNN" acronym-form="plural+short"}: an
    introduction to [VI]{acronym-label="VI"
    acronym-form="singular+short"}

    -   \[5 min\] Monte-Carlo Dropout: the simplest way to have
        [BNNs]{acronym-label="BNN" acronym-form="plural+short"}

    -   \[15 min\] Formalization of the variational objective (and its
        gradients)

    -   \[35 min\] Parameterization of [VI]{acronym-label="VI"
        acronym-form="singular+short"} and recent advancements
        (including normalizing flows and particle-based variational
        inference)

-   \[30 min\] Sampling from intractable distributions

    -   \[15 min\] Introduction to [HMC]{acronym-label="HMC"
        acronym-form="singular+short"}

    -   \[15 min\] Extensions of [HMC]{acronym-label="HMC"
        acronym-form="singular+short"} for Bayesian deep learning:
        [SGHMC]{acronym-label="SGHMC" acronym-form="singular+short"} and
        [SGLD]{acronym-label="SGLD" acronym-form="singular+short"}

-   \[30 min\] Train different models and ensemble them; and other
    tricks

    -   \[10 min\] Ensemble as a way to perform Bayesian inference on
        [DNN]{acronym-label="DNN" acronym-form="singular+short"}: the
        case of Deep Ensembles

    -   \[10 min\] Ensemble as a special case of variational inference

    -   \[10 min\] Bayesian model averaging on [DNN]{acronym-label="DNN"
        acronym-form="singular+short"} for scalable inference

-   \[15 min\] Neural networks are approximation of
    [GPs]{acronym-label="GP" acronym-form="plural+short"}: some lessons
    that can be learn

Relevant literature:
[@Liu2016; @Rezende2015; @Neal10; @Neal96; @Graves2011; @Blundell2015; @Zhang2018]

Part 3. \[40min\] Practical considerations and conclusions {#part-3.-40min-practical-considerations-and-conclusions .unnumbered .unnumbered}
----------------------------------------------------------

Finally, the last part will be dedicated to some practical
considerations (e.g. how to choose priors). And this part will be
concluded with a discussion on computational complexity of Bayesian
inference, with a focus on heterogeneous computing.

-   \[20 min\] A problem for today is a solution for tomorrow: encoding
    prior knowledge for Bayesian [DNN]{acronym-label="DNN"
    acronym-form="singular+short"}

-   \[10 min\] An analysis of the computational divide: challenges and
    opportunities of heterogeneous computing for Bayesian inference

-   \[10 min\] Software and libraries for implementing Bayesian
    inference for deep learning models

-   Final remarks and take-away message

We also reserve 15/20 minutes for questions and other interactions with
the audience.

Potential target audience
=========================

The audience targeted by this tutorial is represented by practitioners
and scientists willing or interested in using [DL]{acronym-label="DL"
acronym-form="singular+short"} models for systems where sound
uncertainty quantification is a requirement. We will assume that the
participants are comfortable with some [DL]{acronym-label="DL"
acronym-form="singular+short"} basics, such as
[DNNs]{acronym-label="DNN" acronym-form="plural+short"} and
[CNNs]{acronym-label="CNN" acronym-form="plural+short"}, and some
concepts of optimization (like mini-batch learning and
back-propagation). A bit of experience with Bayesian inference is
suggested but not required to successfully follow the tutorial, as we
will dedicate a good part of the introduction to make sure everyone is
on-par with some basic probability theory results before diving into the
core content of this tutorial.

Motivation and objectives
=========================

Combined with the availability of open source libraries like Tensorflow
[@Abadi15] and PyTorch [@paszke2019pytorch], [DL]{acronym-label="DL"
acronym-form="singular+short"} has quickly gained attraction in other
communities, from cosmology [@Dieleman2015; @Lanusse2019; @Fussell2019]
and experimental physics [@Acciarri2017; @Aurisano2016] to neuroscience
[@Wang2018; @Yamins2016; @Bellec2018; @Zador2019], and it has
cross-fertilized other computer science fields, such as digital hardware
design [@Kwon2020; @Zhang2020a; @Lu2020], data management systems
[@Kraska2018; @Li2019; @Liu2015; @Marcus2018] and materials science
[@Zhang2020Phys; @Kim2020; @Zhang2020; @Wilt2020]. Disconcertingly,
näive implementations of [DL]{acronym-label="DL"
acronym-form="singular+short"} models are found to be *unreliable* in
some scenarios. A recent analysis of deep [CNNs]{acronym-label="CNN"
acronym-form="plural+short"} for classification, for example, showed
that the predictions are systematically over-confident [@Guo17]. In
practice, this means that there is not a clear way to check whether the
model is "sure" or not about a certain predictions and, as a
consequence, taking informed decisions based on the output of such
models should be carefully considered and properly assessed to avoid
misinterpreting the model behavior. This is an interesting problem from
a methodological research point of view but it is also a concerning
aspect for any possible deployment of [DL]{acronym-label="DL"
acronym-form="singular+short"}-based systems, for which a model is
usually trained just once and could be interrogated with any kind of
input data.

A Bayesian approach to deep learning has shown promising results when it
comes to accurate quantification of uncertainty, without compromising on
performance
[@ovadia2019can; @Tran2020; @Wenzel2020; @wilson2020case; @Osawa2019].
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
approximate variational inference techniques and Monte-Carlo methods. He
gave (spotlight) talks both at the International Conference on Machine
Learning (ICML 2019) and at the NeurIPS conference in 2019 during the
4th Bayesian Deep Learning workshop. He also presented his works in form
of posters at relevant machine learning conferences, such as NeurIPS
2020 and AISTATS 2021 (upcoming).

**Maurizio Filippone** has been an Associate Professor at EURECOM since
2015. Prior to that, he carried out some postdoctoral experience in
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
environmental sciences. He gave several talks at leading research
institutes, such as University of Oxford (2019, 2015), Imperial College
(2018), Google Research NYC (2017), Yandex Moscow (2017) and he was
invited speaker at international events, such as the Deep Bayes summer
school in Moscow, Russia (2018, 2019), the MLCC summer school in Genoa,
Italy (2019) and the Northern Lights Deep Learning Workshop in Tromsø,
Norway (2019). In 2019 he also delivered a tutorial on Gaussian
processes at the IJCNN conference.

