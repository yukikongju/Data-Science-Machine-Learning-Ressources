# LeNet Notes

- To connect the last convolutional layer with the fully connected layer, 
  we need to flatten the tensor with `t.flatten(start_dim=1)`
- The formula to compute the feature map size is `O=(I+K-2P)/S + 1`
    * To keep the feature map the same size, remember the following:
	+ `kernel_size=1, padding=0`
	+ `kernel_size=3, padding=1`
	+ `kernel_size=5, padding=2`
	+ `kernel_size=7, padding=3`
    * Having a stride of 2 will half the feature map (eg 28x28 -> 14x14)
- The `out_channels` argument defines the number of filter we want
    * Ex: If input is (B, C1, H1, W1) and we want output (B, C2, H2, W2), 
      `nn.Conv2d(in_channels=C1, out_channels=C2, kernel_size=K, stride=S, padding=P)`. We don't define H2, W2 explicitly, they can be calculated with the above formula
