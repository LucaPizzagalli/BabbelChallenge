Title:        Babbel Challenge
Author:       Luca Pizzagalli
Date:         2021-12-14
Language:     en
CSS:          retro.css
HTML header:  <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
              <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

# Babbel Challenge

## Bayesian Inference

Bayesian inference is a technique for incorporating new information in a belief corpus.

Most (all) of our belief are not absolute certainties, as we always have a certain degree of uncertainty. In bayesian inference we quantify this uncertainty as a probability, for example the probability that the belief $A$ is true. This probability is called "prior" and it comes from a theoretical model, assumptions, or past experimental data.

As we collect new evidence (let's call it $B$), we want to update our probability to reflect all the information available, past and new.
For doing so we use the Bayes' Theorem:

$$P(A|B) = \frac{P(B|A) P(A)}{P(B)}$$

$P(A|B)$ is our new degree of confidence in $A$ given the new evidence $B$, and it's called "posterior".
Bayesian inference is an iterative process, as newer information can always become available. If this happens we can use our old posterior as new prior for the next step in our quest for building a belief system that better end better maps the real world.

## Multi-Armed Bandit problem

Multi-Armed Bandit problem is a theoretical problem in which we have to balance exploration with exploitation.

In the problem we have $N$ slots machine, each giving a reward coming from an unknown probability distribution when played.
We have in total $T$ rounds and in each round we can choose which slot to play. Our goal is to maximize the total reward.

What's interesting about this problem is the necessary trade-off between spending many rounds trying to understand which slot is the most fruitful, and having many rounds left to squeeze the most out of the best slot.

## Beta-Bernoulli model

Let's say we have a random variable $X$, we know $X$ can be $1$ with probability $P$ and $0$ with probability $1-P$, but we don't know $P$, and we want to find it.

If we sample $X$ multiple times we collect more and more data, with which it's possible to update our belief about what is the true value of $P$, a perfect context for bayesian inference!

$$beta(p;\alpha,\beta) \propto p^{\alpha-1} (1-p)^{\beta-1}$$

This up here is the beta distribution, a probability distribution depending on the two parameters $\alpha$ and $\beta$.
We can see that the shape of the formula is the same of the binomial distribution, with $\alpha-1$ equals to the number of successes and $\beta -1$ equals to the number of failures. The difference is that here the unknown variable to estimate is $p$, not the number of successes.

Therefore the beta distribution naturally expresses the probability distribution for $p$ and we can use it as prior.

The other very convenient thing about the beta distribution is that when we update it with new data the posterior is also a beta distributions, just with the parameters $\alpha$ and $\beta$ updated.


## Thompson Sampling

In the context of Multi-Armed Bandit problem with slots' rewards following a bernulli distribution, many strategies have been proposed. Thompson Sampling is one of those.

In Thompson Sampling we use the beta distributions to model our priors and we update the $\alpha$ s and $\beta$ s as we sample the slots.
For deciding which slot to play we sample all the priors and we choose the slot whose prior provided the highest value. In this way we'll focus more on the slots that we think have highest $P$, but still always giving a chance to the lower performing ones.
