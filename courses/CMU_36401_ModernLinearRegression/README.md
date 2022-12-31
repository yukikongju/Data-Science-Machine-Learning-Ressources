# Homeworks for Modern Linear Regression

[Larry website](https://www.stat.cmu.edu/~larry/=stat401/)
[Cosma website](https://www.stat.cmu.edu/~cshalizi/mreg/15/)

**Homework**

- [X] HW1 - Expectancy and covariance properties for random variables and noises
- [X] HW2 - Linear regression estimators properties
- [X] HW3 - Linear Regression with gaussian noises/errors
- [X] HW4 - Using the residuals to verify linear regression assumptions (homodescadicity, residuals mean is zero); computing pvalues and confidence interval on estimated parameters
- [X] HW5 - Confidence Intervals and pvalue on Multiple Linear RegressionCoefficients; Computing beta coefficients and variance using matrix formula
- [ ] HW6 - fitting a polynomial with factor variables; multiple linear regression assumptions (what makes X^TX invertible)
- [ ] HW7 - Building Rectangle and Ellipsoid Confidence Intervals; Comparing Models using the F test
- [ ] HW8 - 
- [ ] HW9 - 
- [ ] HW10 - 

**Projects**

- [ ] Project 1 - 
- [ ] Project 2 - 
- [ ] Project 3 - 


## Concepts to check out

**HW5**

- [ ] Confidence interval for predicted values (Q1d)
- [X] Using anova F statistic to perform hypothesis testing 
      + H0: model F statistic is 0 (model doesn't predict well);
      + H1: model F statistic is not 0
      + F = ESS/MSE
- [ ] Why do we need to substract 1 when performing `lm(Y~X-1)` (Q3a)

**HW6**

- [ ] What do we plot residual against? y_predicted, y_real?

**HW7**

- [ ] Why does ChatGPT refers as features matrix as `design matrix`, but the solution suggest that it is $X^TX$ (find design matrix Q1d)
- [ ] Why use Bonferonni correction when computing rectangle 
      confidence interval? Why 90 percent confidence interval rectangle correspond 
      to 99 percent 
- [ ] Why don't we use intercept when constructing rectangle interval?
- [ ] Why anova sometimes compute F statistic as NA?


**HW8**



**HW9**



**HW10**


## Ressources

- [Bookdown - R](https://bookdown.org/dli/rguide/inference-on-two-independent-sample-means.html#one-sided-hypothesis-test-1)
- [R Cookbook 2nd edition](https://rc2e.com/)
- [How to interpret qqplot](https://stats.stackexchange.com/questions/101274/how-to-interpret-a-qq-plot)
- [Common Box-Cox Transformation](https://sixsigmastudyguide.com/box-cox-transformation/)
- [Spatial Statistics & Analysis](https://www.css.cornell.edu/faculty/dgr2/ref/index.html)
- [faculty dgr2 R - Cornell](https://www.css.cornell.edu/faculty/dgr2/_static/files/R_html/)
