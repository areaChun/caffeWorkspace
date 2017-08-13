# ShuffleNet
This is caffe implementation of ShuffleNet, For details, please read the original paper:  
["ShuffleNet: An Extremely Efficient Convolutional
Neural Network for Mobile Devices" by Xiangyu Zhang et. al. 2017](https://arxiv.org/pdf/1707.01083.pdf).
This code is based on camel007's implementation(https://github.com/camel007/Caffe-ShuffleNet), but I recode the cuda file for acceleration.

## How to use?
#### caffe.proto:
```
message LayerParameter {
...
optional ShuffleChannelParameter shuffle_channel_param = 164;
...
}
...
message ShuffleChannelParameter {
  optional uint32 group = 1[default = 1]; // The number of group
}
```
compare with shuffle-layer-camel007 

batch=70,size=224*224 

GPU:

shuffle-layer-farmingyard Benchmark star:
shuffle11	forward: 0.998134 ms.
shuffle11	backward: 1.00431 ms.
Average Forward pass: 325.699 ms.
Average Backward pass: 1011.38 ms.
Average Forward-Backward: 1337.36 ms.
Total Time: 26747.3 ms.

shuffle-layer-camel007:
shuffle11 forward: 45.2015 ms.
shuffle11 backward: 45.1394 ms.
Average Forward pass: 6586.59 ms.
Average Backward pass: 7746.23 ms.
Average Forward-Backward: 14333 ms.
Total Time: 286660 ms.

GPU:

shuffle-layer-farmingyard Benchmark star:
shuffle11	forward: 0.50535 ms.
shuffle11	backward: 0.5231 ms.
Average Forward pass: 6689.27 ms.
Average Backward pass: 7707.68 ms.
Average Forward-Backward: 14397.3 ms.
Total Time: 287947 ms.

shuffle-layer-camel007:
shuffle11	forward: 2.93615 ms.
shuffle11	backward: 2.7464 ms.
Average Forward pass: 6586.59 ms.
Average Backward pass: 7746.23 ms.
Average Forward-Backward: 14333 ms.
Total Time: 286660 ms.



