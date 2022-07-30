# Chap 8 - Using Convolutions to Generalize

- [o] Convolutional Neural Network (CNN)
    - [X] With Conv2D only
    - [X] With MaxPool2D
    - [X] With AdaptiveMaxPool2d
    - [ ] With Dropout2D
    - [ ] With BatchNorm2D
    - [ ] With Regularization and weight penalties/decay
- [ ] Comparing our models
- [ ] Saving our model


**Notes**

- if we have a tensor of shape [3, 32, 32], we can use `t.unsqueeze(0)` to reshape 
  it to [1, 3, 32, 32]
- Conv2D is of size [out_ch x in_ch x 3 x 3] (ie Batch x Channels x Height x Width)
    * image size : [1, 3, 32, 32]
- To figure out how many neurons our Linear layer has after the Conv2D layer:
    * ` nn.Linear( last_out_conv2d x 8 * 8 )`, where 8 is 32/(2**num_conv2d_layers)
    * We can use `nn.AdaptiveMaxPool2d((5,7))` to fix image to 5x7 

