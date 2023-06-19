
## Neural Networks

A neural network is [infinitely expressive](https://en.wikipedia.org/wiki/Universal_approximation_theorem). A neural network can approximate any computable function, given enough parameters. A "computable function" can cover just about anything you can imagine: understand and translate human speech; paint a picture; diagnose a disease from medical imaging; write an essay; etc...

The way a neural network approximates a function actually turns out to be very simple. The key trick is to combine two extremely basic steps:

- Matrix multiplication, which is just multiplying things together and then adding them up
- The function  𝑚𝑎𝑥(𝑥,0), which simply replaces all negative numbers with zero.
- Compute error through a loss function (example, mae)
- Adjust weights through Gradient Descent (Derivative) that traverses towards minimizing loss
- Repeat from the beginning, stop when loss is minimized.

### Parallel Distributed Processing (PDP)

In fact, the approach laid out in PDP is very similar to the approach used in today's neural networks. The book defined parallel distributed processing as requiring:

- A set of processing units
- A state of activation
- An output function for each unit
- A pattern of connectivity among units
- A propagation rule for propagating patterns of activities through the network of connectivities
- An activation rule for combining the inputs impinging on a unit with the current state of that unit to produce an output for the unit
- A learning rule whereby patterns of connectivity are modified by experience
- An environment within which the system must operate
 
> We have created a nifty little example showing that we can draw squiggly lines that go through some points. So what?
> Well... the truth is that actually drawing squiggly lines (or planes, or high-dimensional hyperplanes...) through some points is literally *all that deep learning does*!
> If your data points are, say, the RGB values of pixels in photos of owls, then you can create an owl-recognizer model by following the exact steps above.

> In other words, to recap, a neural network is a particular kind of machine learning model, which fits right into Samuel's original conception. 
> Neural networks are unique because they are highly flexible, which means they can solve an unusually wide range of problems just by finding the right weights. This is powerful because **stochastic gradient descent** provides us a way to find those weight values automatically.


- Our inputs are the images. Our weights are the weights in the neural net. 
- Our model is a neural net. Our results are the values that are calculated by the neural net, like "dog" or "cat."
- SGD is our mechanism for updating the weight assignments

> Nonlinear activation functions are among the main reasons why neural networks are so powerful and can practically approximate any function.
> _So this thresholding is what contributes to the property of “Universal Approximation” - that says simple neural nets can approximate continuous functions._
> `m` changes the slope, and `b` changes where the "hook" appears. This function doesn't do much on its own, but look what happens when we add two of them together:

A simple (2 hidden layer) neural network without thresholding cannot approximate this function:

![image](https://github.com/jeyabalajis/deep_learning_fastai/assets/15995686/fc26ca7a-98d9-4d48-b0ea-9ee17eefdb25)

However, if we introduce the thresholding (ReLU) between the 2 layers, the same network can magically learn to predict (approximately) as per the sine wave function:

![image](https://github.com/jeyabalajis/deep_learning_fastai/assets/15995686/48f45276-eacc-4886-a478-454aacf10e33)

### Sensing how good is our fit (vis-a-vis the actual label)

- An easy metric we could use is _mean absolute error (MAE)_, which is the distance from each data point to the curve.

> In a modern neural network we'll often have tens of millions of parameters to fit, or more, and thousands or millions of data points to fit them to. 
> We're not going to be able to do that by moving sliders around! We'll need to automate this process.
> Thankfully, that turns out to be pretty straightforward. We can use calculus to figure out, for each parameter, whether we should increase or decrease it.

### Learning Rate

> The "small number" we multiply is the learning rate, the most important hyper-parameter to set when training a neural network.
> We need to _decrease our learning rate as we train_. This is done using a learning rate schedule and can be automated in most deep learning frameworks, such as fastai and PyTorch.

## References

[How a Neural Network approximates any given function](https://www.kaggle.com/code/jhoward/how-does-a-neural-net-really-work#How-a-neural-network-approximates-any-given-function)
[NumPy Basics](https://wesmckinney.com/book/numpy-basics.html)
[Calculus Basics](https://www.youtube.com/playlist?list=PLybg94GvOJ9ELZEe9s2NXTKr41Yedbw7M)
