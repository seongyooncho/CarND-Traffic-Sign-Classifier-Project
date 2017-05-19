# **Traffic Sign Recognition**

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.png "Visualization"
[image2]: ./examples/grayscale.png "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/0.jpg "Traffic Sign 1"
[image5]: ./examples/1.jpg "Traffic Sign 2"
[image6]: ./examples/2.jpg "Traffic Sign 3"
[image7]: ./examples/3.jpg "Traffic Sign 4"
[image8]: ./examples/4.jpg "Traffic Sign 5"
[image9]: ./examples/5.jpg "Traffic Sign 6"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook.  

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the fourth code cell of the IPython notebook.  

Here is an exploratory visualization of the data set. It is a bar chart showing number of examples per label. It shows the most label, label 2, is almost 10 times more than the least label, label 0. It must be adjusted to prevent overfitting for certain labels.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the fifth code cell of the IPython notebook.

As a first step, I decided to convert the images to grayscale because colors can be varied by lighting conditions and can corrupt training.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data because the original data is from zero to 255. I have normalized it from zero to one.

#### 2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

I replicated original training set to normalize number of images for each labels.

I used original validation set and testing set as is.

The training set had 34799 number of images. The validation set and test set had 4410 and 12630 number of images.

The sixth code cell of the IPython notebook contains the code for replicating the data set. I decided to replicate data because the differences in number of labels are quite dramatic, as you can see in the previous figure *Number of examples per label*. So it needs to be adjusted to prevent overfitting on certain labels.

#### 3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the seventh cell of the ipython notebook.

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x1 Grayscale image   							|
| Convolution 5x5     	| 1x1 stride, same padding, outputs 28x28x24 |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x24 				|
| Convolution 5x5	    | 1x1 stride, same padding, outputs 10x10x64 |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x64 				|
| Flatten					|	1600 |
| Fully connected		| outputs 480 |
| RELU					|												|
| Fully connected		| outputs 336 |
| RELU					|												|
| Fully connected		| outputs 43 |
| Softmax				|  |

#### 4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the ninth cell of the ipython notebook.

To train the model, I followed most of the settings used for LeNet example.
- Optimizer: AdamOptimizer
- Batch size: 128
- Epochs: 30
- Learning rate: 0.001

#### 5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the ninth cell of the Ipython notebook.

My final model results were:
* training set accuracy of 1.000
* validation set accuracy of 0.966
* test set accuracy of 0.952

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen? __LeNet__
* Why did you believe it would be relevant to the traffic sign application? __Traffic sign is basically signs and it would be relevant to classifying letters as well.__
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well? __Training accuracy show 1.0 which means it's overfitted and it can result in bad performances. But the validation and test set show high accuracy and it seems okay.__

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are six German traffic signs that I found on the web:

|No.|Image|No.|Image|No.|Image|
|---|:---:|---|:---:|---|:---:|
|1|![alt text][image4]|2|![alt text][image5]|3|![alt text][image6]|
|4|![alt text][image7]|5|![alt text][image8]|6|![alt text][image9]|

The fourth, fifth, and sixth images might be difficult to classify because it has two signs in one image and it's not centered.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Speed limit (50km/h)  | Speed limit (50km/h)   									|
| General caution 			| General caution								|
| Right-of-way at the next intersection| Right-of-way at the next intersection |
| General caution | Right-of-way at the next intersection |
| Go straight or right| Priority road |
|Roundabout mandatory|Road work|


The model was able to correctly guess 3 of the 6 traffic signs, which gives an accuracy of 50%. First half of the images is somewhat resembles to test set. So we can say the accuracy is favorable. Second half of the images is very mush different to test set. I think I need to augment training set with translation and scaling to make classification works.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 13th cell of the Ipython notebook.

*I couldn't figure out why my results are relatively big numbers, not below 1.*

For the first image, the model is relatively sure that this is a Speed limit (50km/h) sign (probability of 85), and the image does contain a gSpeed limit (50km/h) sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 85.45128632         	|Speed limit (50km/h)|
| 30.60815048     			|Wild animals crossing|
| 28.33760452				    |Speed limit (30km/h)|
| 14.92870998	      		|Speed limit (80km/h)|
| -0.17537288				    |Speed limit (60km/h)|

For the second image, the model is relatively sure that this is a general caution sign (probability of 111), and the image does contain a general caution sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
|111.94968414| General caution|
|52.3482132| Pedestrians|
|40.1158905| Right-of-way at the next intersection|
|27.73794937| Road work|
|22.26135063| Keep right|

For the third image, the model is relatively sure that this is a Right-of-way at the next intersection sign (probability of 65), and the image does contain a Right-of-way at the next intersection sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
|65.38359833|Right-of-way at the next intersection|
| 8.47878456|Roundabout mandatory|
|3.60774255|Traffic signals|
|3.5506916|Beware of ice/snow|
|1.94609737|Slippery road|

For the fourth image, the model is relatively not sure that this is a Right-of-way at the next intersection sign (probability of 22), and the image does contain a Right-of-way at the next intersection sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
|22.4647522|Right-of-way at the next intersection|
|10.27308369|Ahead only|
|7.05428839|Roundabout mandatory|
|3.20513678|Children crossing|
|1.49114037|Speed limit (100km/h)|

For the fifth image, the model is relatively not sure that this is a Priority road sign (probability of 36), and the image does contain a Priority road sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
|36.53054428|Priority road|
|26.76468468|Ahead only|
|25.35012245|No passing|
|14.84806919|Roundabout mandatory|
|13.13069916|No entry|

For the sixth image, the model is relatively not sure that this is a Road work sign (probability of 42), and the image does contain a Road work sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
|42.82881927|Road work|
|28.9650116|Stop|
|8.89744282|Yield|
|6.45120382|Keep right|
|5.39219475|Beware of ice/snow|
