# Q3: Simulation Problems

# a) generate data following linear regression and plot it

n <- 100
b0 <- 5
b1 <- 3

x <- runif(n, 0, 1)
y <- b0 + b1 * x + rnorm(n, 0, 1)

reg <- lm(y~x)
plot(x, y)
abline(reg$coefficients[1], reg$coefficients[2], col='green')


# b) compute beta1 chapeau and plot histogram

n <- 100
num_iters = 1000
betas1 <- rep(NA, 1, num_iters)

for (i in 1:num_iters) {
  # generate the linear regression
  x <- runif(n, 0, 1)
  y <- b0 + b1 * x + rnorm(n, 0, 1)
  model <- lm(y~x)
  betas1[i] <- model$coefficients[2]
}

# the mean for betas1 is
mean(betas1)

# the histogram
hist(betas1, xlab = expression(hat(beta)[1]), probability = FALSE, 
     breaks = 50, title="histogram of betas 1 with normal errors")


# c) compute beta1 chapeau and plot histogram using cauchy distribution to generate epsilons

betas1_cauchy <- rep(NA, 1, num_iters)
for(i in 1:num_iters) {
  x <- runif(n, 0, 1)
  y <- b0 + b1 * x + rcauchy(n)
  model <- lm(y~x)
  betas1_cauchy[i] <- model$coefficients[2]
}

hist(betas1, xlab = expression(hat(beta)[1]), probability = FALSE, 
     breaks = 50, title="histogram of betas 1 following cauchy errors")

# the histogram is the same

# d) plotting yi and wi

betas1_q3d <- rep(NA, 1, num_iters)
for(i in 1:1000) {
  x <- runif(n, 0, 1)
  w <- x + rnorm(n, 0, 2)
  y <- b0 + b1 * x + rnorm(n, 0, 1)
  model <- lm(y~w)
  betas1_q3d[i] <- model$coefficients[2]
}

# plotting
model <- lm(y~w)
plot(w, y)
abline(model$coefficients[1], model$coefficients[2], col='blue')

# doesn't form a line

# expectancy of betas 1 chapeau
mean(betas1_q3d)

# histogram
hist(betas1_q3d, xlab = expression(hat(beta)[1]),  probability = FALSE, breaks = 50)

# TODO: the effect of having the errors in X: 



# Q4: airquality

data("airquality")

# a) using summary and pairs functions
summary(airquality)
pairs(airquality)

# b) 
plot(airquality$Solar.R, airquality$Ozone)

# it seems that that the two variables are positively correlated


# c) fit a least square regression
model <- lm(Ozone~Solar.R, data=airquality)
abline(model$coefficients[1], model$coefficients[2], col='red')
summary(model)$coefficients
summary(model)


# d) compute the residuals
res <- airquality$Ozone - predict(model, newdata = 
          data.frame(Solar.R = airquality$Solar.R))
plot(airquality$Solar.R, res, ylab="residuals", xlab="ozone")
abline(h=0)

# No, the linear regression assumptions doesn't hold because 
# the graph does not follow homoskedacity: the errors is not constant




