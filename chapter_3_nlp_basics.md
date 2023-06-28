The most widely practical application of NLP is classification, i.e. classifying a document automatically into some category.
Examples:  
- Sentiment analysis (e.g are people saying *positive* or *negative* things about your product)
- Author identification (what author most likely wrote some document)
- Legal discovery (which documents are in scope for a trial)
- Organizing documents by topic
- Triaging inbound emails

### Stochastic Gradient Algorithm (SGD)

> SGD, or stochastic gradient descent, is an optimization algorithm. Specifically, SGD is an algorithm that will update the parameters of a model in order to minimize a given loss function that was evaluated on the predictions and target. The key idea behind SGD (and many optimization algorithms, for that matter) is that the gradient of the loss function provides an indication of how that loss function changes in the parameter space, which we can use to determine how best to update the parameters in order to minimize the loss function. This is what SGD does.

#### Mini-Batch

As a compromise, we calculate the average loss for a small subset of the dataset at a time. This subset is called a mini-batch. Using mini-batches are also more computationally efficient than single items on a GPU.

#### SGD steps in Machine Learning

1. Initialize the parameters – Random values often work best.
2. Calculate the predictions – This is done on the training set, one mini-batch at a time.
3. Calculate the loss – The average loss over the minibatch is calculated
4. Calculate the gradients – this is an approximation of how the parameters need to change in order to minimize the loss function
5. Step the weights – update the parameters based on the calculated weights
6. Repeat the process
7. Stop – In practice, this is either based on time constraints or usually based on when the training/validation losses and metrics stop improving.

#### Gradient

The gradients tell us how much we have to change each weight to make our model better. It is essentially a measure of how the loss function changes with changes of the weights of the model (the derivative).

#### ReLU

ReLU just means “replace any negative numbers with zero”. It is a commonly used activation function.

> The activation function is another function that is part of the neural network, which has the purpose of providing non-linearity to the model. The idea is that without an activation function, we just have multiple linear functions of the form y=mx+b. However, a series of linear layers is equivalent to a single linear layer, so our model can only fit a line to the data. By introducing a non-linearity in between the linear layers, this is no longer true. Each layer is somewhat decoupled from the rest of the layers, and the model can now fit much more complex functions. In fact, it can be mathematically proven that such a model can solve any computable problem to an arbitrarily high accuracy, if the model is large enough with the correct weights. This is known as the universal approximation theorem.