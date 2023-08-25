# PART I - The Yang

## 01 - Introduction To Causality [ COMPLETED ]

**Comment le biais peut apparaitre lorsqu'on fait A/B test**


$E[Y|T=1] - E[Y|T=0] = E[Y_1 - Y_0 | T=1] - (E[Y_0|T=1] - E[Y_0|T=0])$

- La difference observée entre les 2 groupes est égale à la moyenne du 
traitement sur le groupe traité moins la différence entre les 2 groupes 
au début de l'expérience (le biais)

- Puisqu'on assigne chaque groupe to either has the treatment or not, on 
  ne peut pas savoir comment le groupe réagit avec l'option alternative, 
  alors il y a aura toujours un biais.
  * Ex: groupe A: sans traitement; on ne peut pas tester le groupe A avec traitement aussi pour voir si on a les mêmes résultats qu'avec le groupe B

- Dans les tests A/B, on assume que les buckets ont des gens similaires, mais 
  ce n'est pas toujours le cas, ce qui créé des biais


**Comment déterminer si la corrélation est une causation**

- Si on est capable de montrer que les 2 groupes du début son similaire, c-à-d
  $E[Y_0|T=0] = E[Y_0|T=1]$, alors c'est une causation. La différence des 
  moyennes devient la relation causale
- Correlation = causation iif there is no bias

## 02 - Randomised Experiments [ COMPLETED ]

**How does randomized experiment negate potential bias between samples**

- The bias that exists between the two group doesn't exist, so the difference 
  observed is the difference between treatment or not

**What to do if we cannot perform a randomized controlled trial (RCT)?**

- Conditionnal Randomization


## 03 - Stats Review: The Most Dangerous Equation [ COMPLETED ]

**How can we know that the difference between samples is not due to variance?**

- Standard Error, Confidence Intervals and Hypothesis Testing

## 04 - Graphical Causal Models: Understanding Cofounding variable and selection bias [ COMPLETED ]

**How can conditional probability solve bias in samples?**

- Instead of giving treatment to people that are very sick and not giving 
  treatment to people that are not that sick, we choose randomly to give 
  or not the treatment only on people that or sick/not sick.

**How can two independant events become dependant when knowing the outcome of a third event?**

- If two events A and B influences C, and we know that C occurred and not B, 
  then it surely means that A likely occurred.

**How to use graphical model to diagnose which bias we are dealing with?**

- Help us find COLLIDER

**What is a cofounding variable**

- Occurs when treatment and the outcome share the same cause

**What is selection bias**

- We control for more variable than we should


## 05 - The Unreasonable Effectiveness of Linear Regression [ COMPLETED ]

**How to Construct Linear Regression**

- With a single regressor variable ie $y=\beta _0 + \beta _1 x$
    * $\beta _1 = cov(X, Y) / Var(Y)$
- With multiple regressor variables ie $y=\beta _0 + \beta _1 x_1 + \beta _2 x_2 + ...$
    * $\beta = (X'X)^{-1}X'Y$

**How to interpret linear regression variables**

- coefficient: by how much do we expect y to increase/decrease when x changes by 1
- pvalue: 

**How to determine if the variable is causal or not**


**How can we use Linear Regression to perform A/B Testing** [TODO]


## 06 - Grouped and Dummy Regression [ COMPLETED ]

**Why are not all data points created equal?**

- Ex: big school have less variance than small school
- Heteroskedasticity
- Aggregate values occurs when we need data privacy
- We can use weights to combat this effect

- The sample with higher sample size and lower variance should be given more weight


## 07 - Beyond Confounders [ IN PROGRESS ]

## 08 - Instrumental Variables

## 09 - Non Compliance and LATE

## 10 - Matching

## 11 - Propensity Score

## 12 - Doubly Robust Estimation

## 13 - Difference-in-Differences

## 14 - Panel Data and Fixed Effects

## 15 - Synthetic Control

## 16 - Regression Discontinuity Design

