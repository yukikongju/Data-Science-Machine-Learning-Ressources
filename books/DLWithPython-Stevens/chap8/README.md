# Chap 8 - Using Convolutions to Generalize

[Code](https://github.com/deep-learning-with-pytorch/dlwpt-code/tree/master/p1ch8)

- [O] Convolutional Neural Network (CNN)
    - [X] With Conv2D only
    - [X] With MaxPool2D
    - [X] With AdaptiveMaxPool2d
    - [X] With Dropout2D
    - [X] With BatchNorm2D
    - [ ] With Blocking
- [X] Using regularization/weight decay
    - [X] computing weight decay inside training loop
    - [X] initialize weight_decay inside optimizer
- [X] Comparing our models
- [ ] Plotting history and accuracy
- [ ] Saving our model


**Notes**

- if we have a tensor of shape [3, 32, 32], we can use `t.unsqueeze(0)` to reshape 
  it to [1, 3, 32, 32]
- Conv2D is of size [out_ch x in_ch x 3 x 3] (ie Batch x Channels x Height x Width)
    * image size : [1, 3, 32, 32]
- To figure out how many neurons our Linear layer has after the Conv2D layer:
    * ` nn.Linear( last_out_conv2d x 8 * 8 )`, where 8 is `32/(2**num_conv2d_layers)`
    * `nn.Linear (last_out x last_out x num_channels // 2)`
    * We can use `nn.AdaptiveMaxPool2d((5,7))` to fix image to 5x7 

