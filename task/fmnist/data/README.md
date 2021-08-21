We follow exactly the same data processing procedures described in [the paper](https://arxiv.org/abs/1902.00146) we are comparing with. See ```create_dataset.py``` for the details.

First download raw data:

```
[currently in the under the same directory as this readme file]
cd ./raw_data/fashion
wget http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz
wget http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz
wget http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz
wget http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz
cd ../../
```
Then go to the `task/fmnist/data` dir, and runpreprocess data:

```
python create_dataset.py
```

The testing data and training data will be in the ```data/test``` and ```data/train``` folders respectively. These (standard) training and testing samples are exactly the same as those used in the [AFL paper](https://arxiv.org/abs/1902.00146).