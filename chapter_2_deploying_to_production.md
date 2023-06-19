## High level Steps

![image](https://github.com/jeyabalajis/deep_learning_fastai/assets/15995686/a43f9a14-8a4f-41b5-b071-8c322009d1a5)

### Training before Cleaning

Before cleaning, pass it through a model, train and find out the specific images where the **loss** is high (either the model is very confident but it is predicting wrong (or) the model is not confident but it is predicting correctly)

### Data Augmentation

> `RandomResizedCrop`: Out of the same picture, a different portion of the image is picked up. Useful for _Data Augmentation_.


### Confusion Matrix

- Useful metric when the labels are categories.
- plot top losses: Tells us the places where the loss is the highest. The model is very confident that it is a grizzy, but it is a black (or) it has predicted correctly, but it is not not confident.

![image](https://github.com/jeyabalajis/deep_learning_fastai/assets/15995686/b745e7f8-1887-489f-9b9a-d0c40d29fe69)
