# ResNet Notes

- [paper](https://arxiv.org/pdf/1512.03385)
- [video](https://www.youtube.com/watch?v=DkNIBBBvcPs&list=PLhhyoLH6IjfxeoooqP9rhU3HJIAVAJ3Vz&index=20)

- When adding residual to output in `forward()`, we need to make 
  the identity and the output the same shape using a `identity_downsampling` 
  module
- 34-layer has worse training and validation performance compared to 18-layer model. This degradation is not due to vanishing gradient (norm are healthy)
    * Later: why 34-layer has lower convergence rate


