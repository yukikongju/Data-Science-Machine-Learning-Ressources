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

- [X] How can we measure the influence of individual point in the fit? 
      + jacknife (studentized) residuals 
      + cook's distance
- [ ] How to plot residuals against covariate


**HW9**



**HW10**


## Ressources

- [Bookdown - R](https://bookdown.org/dli/rguide/inference-on-two-independent-sample-means.html#one-sided-hypothesis-test-1)
- [Cosma Shalizi - The Truth About Linear Regression](https://www.stat.cmu.edu/~cshalizi/TALR/)
- [R Cookbook 2nd edition](https://rc2e.com/)
- [How to interpret qqplot](https://stats.stackexchange.com/questions/101274/how-to-interpret-a-qq-plot)
- [Common Box-Cox Transformation](https://sixsigmastudyguide.com/box-cox-transformation/)
- [Spatial Statistics & Analysis](https://www.css.cornell.edu/faculty/dgr2/ref/index.html)
- [faculty dgr2 R - Cornell](https://www.css.cornell.edu/faculty/dgr2/_static/files/R_html/)

