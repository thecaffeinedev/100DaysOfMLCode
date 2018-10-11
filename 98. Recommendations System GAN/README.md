# Recommendations-GAN
Experiments using GANs for recommendations in TensorFlow

## To run:
To run and develop on the various tests, do the following:
```
python3 setup.py develop
```
This will install the necessary libraries into your environment, including the test scripts. The scripts available are the following:
```
ganrecs_mnist_test: Test the GAN architecture with MNIST data set
svd_test: Run recommendation tests using the MovieLens data set and SVD
run_ml_recs: Run recommendation tests using the MovieLens data set and GAN
```
