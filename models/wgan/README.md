# WGAN Notes

Generative Adversial Networks (GANs) are comprised of:
- discriminator: binary classifier which determine whether an image is 
  fake or not
- generator: a model that generate an image from a latent vector

More generally, we have two distribution:
- Pr: Probability distribution for real images
- Pg: Probability distribution for generated images

The goal of the generator is to minimize the distance between Pr and Pg. 
There are several distances used to measure that distance:
- Kullback-Leibler (KL) divergence
- Jensen-Shannon (JS) divergence
- Wasserstein Distance <- The paper introduce this new statistic measure

Why:
- JS has some issue around gradient leading to unstable training, which 
  the Wasserstein distance aims to solve

