# Federated-Dimensionality-Reduction
Master Project at UZH

This projects goal is to meassure the effect of dimensionality reduction on a classification task. Dimensionality reduction  simplifies the datastructure and thus can lead to faster training time. This is more relevant for large models and datasets or if the device used for training is not strong enough and the training time would be very big. But in exchange for faster training time the accuracy for example will have to pay for that because if the dimensions are being reduced, then there is a good chance that some features that are actually important are getting lost in the process. To find out if the trade-off between training time and performance is worth it, this project uses 2 popular dimensionality reduction methods, mainly PCA and Autoencoder for its purposes.
The datasets used are Mnist, Fashionmnist and Cifar 10.

WIP