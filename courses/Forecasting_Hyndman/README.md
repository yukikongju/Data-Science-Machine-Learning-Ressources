# Forecasting: Principles and Principles by Rob Hyndman

All about time series forecasting. Course can be found [here](https://otexts.com/fpp3/)


# Chapters Exercices

- [X] Chapter 1 - Getting Started
- [.] Chapter 2 - Time Series Graphics
- [X] Chapter 3 - The Forecaster's toolbox
- [ ] Chapter 4 - Judgmental Forecasts
- [.] Chapter 5 - Time Series Regression Models
- [X] Chapter 6 - Time Series Decomposition
- [X] Chapter 7 - Exponenetial Smoothing
- [ ] Chapter 8 - ARIMA models
- [ ] Chapter 9 - Dynamic Regression Models
- [ ] Chapter 10 - Forecasting Hierarchical or grouped time series
- [ ] Chapter 11 - Advanced Forecasting Methods
- [ ] Chapter 12 -

# What we learned

**Modelisation**

- Regression Models
    * Metrics: `accuracy()`, `CV()`
- Decomposition: 
    * Classical: Additive and Multiplicative Models
    * Advanced: X11, SEATS, STL Decomposition
    * Metrics: measuring trend and seasonality with $F_t$ and $F_s$
- Exponential Smoothing
    * Methods: Simple Exponential Models, Holt (dampened), Holt-Winter (additive vs multiple method) (also called double/triple exponential method), 
    * Dampened Trend: decreasing trend rather than constant or increasing trend
    * ETS: (error, trend, seasonality) [A: additive; M: multiplicative; N: None]
	+ Error: Additive (A), Multiplicative (M)
	+ Trend: (N), (A), Additive dampened (A_d)
	+ Seasonal: (N), (A), (M)
- ARIMA models

**Forecasting**

- Classical Forecasting: naive, seasonal naive, trend, drift

**Model Selection**

- Information Criterion



## Ressources

- [Cornell Time Series Exercices in R](https://www.css.cornell.edu/faculty/dgr2/_static/files/R_PDF/exTSA.pdf)
- [Solutions - Forecasting: Principles and Practice](https://robjhyndman.com/forecasting/solutions.pdf)
- [Notes for fpp3](https://qiushiyan.github.io/fpp/)
- [Element of Statistical Learning](https://hastie.su.domains/Papers/ESLII.pdf)
- [Penn State - Stat 462 - Applied Regression Analysis](https://online.stat.psu.edu/stat462/node/78/)
- [Intro to Time Series Analysis - Tejendra Pratap Singh](https://bookdown.org/singh_pratap_tejendra/intro_time_series_r/)

