
## Neural Networks

> In other words, to recap, a neural network is a particular kind of machine learning model, which fits right in to Samuel's original conception. 
> Neural networks are special because they are highly flexible, which means they can solve an unusually wide range of problems just by finding the right weights. This is powerful, because **stochastic gradient descent** provides us a way to find those weight values automatically.


- Our inputs are the images. Our weights are the weights in the neural net. 
- Our model is a neural net. Our results are the values that are calculated by the neural net, like "dog" or "cat."
- SGD is our mechanism for updating the weight assignments

> Nonlinear activation functions are among the main reasons for which neural networks are so powerful and can practically approximate any function.
> _So this thresholding is what contributes to the property of “Universal Approximation” - that says simple neural nets can approximate continous functions._

A simple (2 hidden layer) neural network without thresholding cannot approximate this function:

![image](https://github.com/jeyabalajis/deep_learning_fastai/assets/15995686/fc26ca7a-98d9-4d48-b0ea-9ee17eefdb25)

However, if we introduce the thresholding (ReLU) between the 2 layers, the same network can magically learn to predict (approximately) as per the sine wave function:

![image](https://github.com/jeyabalajis/deep_learning_fastai/assets/15995686/48f45276-eacc-4886-a478-454aacf10e33)

## References

[How a Neural Network approximates any given function](https://www.kaggle.com/code/jhoward/how-does-a-neural-net-really-work#How-a-neural-network-approximates-any-given-function)
[NumPy Basics](https://wesmckinney.com/book/numpy-basics.html)

