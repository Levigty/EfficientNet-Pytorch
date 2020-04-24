# EfficientNet-Pytorch
A demo for train your own dataset on EfficientNet

Thanks for the >[A PyTorch implementation of EfficientNet](https://github.com/lukemelas/EfficientNet-PyTorch), I just simply demonstrate how to train your own dataset based on the EfficientNet-Pytorch.

## Step 1：Prepare your own classification dataset
---
Then the data directory should looks like:   
```
-dataset\
    -model\
    -train\
        -1\
        -2\
        ...
    -test\
        -1\
        -2\
        ...
```

## Step 2: train and test 
(1)You can choose to download the pre-trained model automatically or not by modify the ```line 169```.

The pre-trained model is available on >[release](https://github.com/lukemelas/EfficientNet-PyTorch/releases). 

You can download them under the folder ```eff_weights```.

(2)Change some settings to match your dataset.
i.e. ```line13-22```
 ```
    run efficientnet_sample.py to start train and test
 ```
(3)You can get the final results and the best model on ```dataset/model/```.
