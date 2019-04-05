# Dog Breed Classifier

In this project, breed of dog is classified from the given dog image. There are 133 dog breed category used for classification. Dataset
for training, Validation and Testing is provide by udacity with bottleneck feature extracted. I have used transfer learning approach using
pre-trained CNN models to classify images. 

Model can also accept Human images and correctly classifies it as Human or Dog. If image is of human then resembling dog breed is predicted. 
If image is neither human nor dog then model displays correct message to indicate this.     

## Project Dependencies 

This project requires following libraries to be imported. 

	- Numpy
	- keras
	- matplotlib
	- glob
	- sklearn
	- random
	- tqdm, PIL, cv2 for image processing

## Implementation

Algorithm accepts a file path to an image and first determines whether the image contains a human, dog, or neither.  Then,
- if a __dog__ is detected in the image, return the predicted breed.
- if a __human__ is detected in the image, return the resembling dog breed.
- if __neither__ is detected in the image, provide output that indicates an error.

I have used bottleneck features from pre-trained model. These features are passed to GlobalAveragePooling2D layer as input to reduce overfitting in 
training model. If we instead just flatten output from last convolution layer then we will end up training million of weights in final layer which will 
overfit the model. What GAP does is basically it reduces the trainable features by taking average val from each filter in last convolution layer. 
This way each convolution filter output single number. Pixels with high intensity have large impact on content representation of image which 
enchances object localization.

In final layer I have used Dense fully connected output layer with 133 categories for dog breed classification applying softmax function 
to output score to calculate probablities for each dog breed.

In this project, I have also tried building the CNN model from scratch to train model to classify dog breed. I could easily achieve accuracy of 
18.89% in 30 epochs. 

## Project Observations:
Using transfer learning approach, I have evaluated several pre-trained CNN model such as InceptionV3, Xception, Resnet50 and Vgg19. 
Experimenting with all model VGG19, InceptionV3, Xception and Resnet50, I got best testing accuracy with Xception around 83.97%. 
Resnet50 score next with 82.59% followed by Xception with 79.19% with each model training running for 20 epochs. 
VGG19 gave low accuracy of 47.61% running for 40 epochs. Now performance for Xception is 20 seconds slower then Resnet50 but given the improvement 
in accuracy of more than 1% its worth the tradeoff. Based on these stats I decide to go with Xception model with highest test accuracy.

For testing the trained model, I downloaded 6 dogs images from internet(https://www.akc.org/dog-breeds/). Ran predictions on all 6 dog images. 
Result are great as 5 off 6 dog breed prediction matches with true label. Dog image 6 did not matched with prediction
(Truth - Icelandic face, Predicton - Belgian_tervuren).

Then I input two human images (first image is me and second is my uncle). Algorithm successfully classified us as Humans and showed interesting 
prediction to resembling dog breed.

At last I input two cat images. Algorithm outputs "Error: Neither dog nor human detected in image" for both the cat images as expected. 
Algorithm classifies it as neither humans nor dogs and goes into Error condition as outlined in Step 6 to display correct output message. 


### Below are some of the further improvements I think can be done.

	- We can use data augmentation to further improve accuracy of predictions. Through data augmentation model can learn to predict images 
	  irrespective of Scale Invariance, Rotational Invariance and Translation Invariance.  
	- We can change Optimizer function and try with different optimizer to further improve performance and accuracy of algorithm. 
	- We can add additional layers or nodes to FC layer to improve accuracy of predictions. As we saw feature representation of images is 
	  well done by pretrained CNN Xception. Experimenting with different layers in fully connected network with Dropout probablities can lead 
	  us to improved accuracy.
	- We can try changing different hyperparameter like Epochs, Batch Size etc to further improve accuracy of model. 
