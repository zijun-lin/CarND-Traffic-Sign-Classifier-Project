# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./output_images/random_image_10.png "Visualization"
[image2]: ./output_images/image_process.png "before and after image process"
[image3]: ./output_images/new_images.png



## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it!

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I calculate summary statistics of the traffic signs data set:

* The size of training set is **34799**
* The size of the validation set is **4410**
* The size of test set is **12630**
* The shape of a traffic sign image is **(32, 32, 3)**
* The number of unique classes/labels in the data set is **43**

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is show image of 10 random data points.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I use `cv2.cvtColor()` function to convert the images to grayscale, than I use `(pixel - 128)/ 128` to approximately normalize the data, so that the data has mean zero and equal variance. Finally, I add the single image data to a data set.

##### process function:
```python
def process_image(dataset):
    n_imgs, img_height, img_width, _ = dataset.shape
    processed_dataset = np.zeros((n_imgs, img_height, img_width, 1))
    for idx in range(len(dataset)):
        img = dataset[idx]
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        processed_dataset[idx, :, :, 0] = (gray - 128.0) / 128.0
    return processed_dataset
```
##### Image processing results:
* Before process:
Dataset shape:  (34799, 32, 32, 3)
Date type:  uint8
* After process:
Dataset shape:  (34799, 32, 32, 1)
Date type:  float64

Here is an example of an original image and an augmented image, I display 5 random images:
![alt text][image2]


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 gray image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x12 	|
| RELU					|												|
| Dropout				|	keep_dropout: 0.5							|
| Max pooling	      	| 2x2 stride,  outputs 14x14x12 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x24	|
| RELU					|												|
| Dropout				|	keep_dropout: 0.5							|
| Max pooling	      	| 2x2 stride,  outputs 5x5x24   				|
| Flatten	        	| inputs 5x5x24  outputs 600        			|
| Fully connected 		| inputs 600  outputs 400           			|
| RELU					|												|
| Dropout				|	keep_dropout: 0.5							|
| Fully connected 		| inputs 400  outputs 120           			|
| RELU					|												|
| Dropout				|	keep_dropout: 0.5							|
| Fully connected 		| inputs 120  outputs 84             			|
| RELU					|												|
| Dropout				|	keep_dropout: 0.5							|
| Fully connected 		| inputs 84  outputs 43             			|
| Softmax				|           									|
| Loss					| cross_entropy									|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I train the model for 50 epochs and the batch size is 128. I use the cross entropy to describe the loss function. The initial learning rate is set at 0.001 and the keep dropout set as 0.5. I use the Adam Optimizer to train the model.


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* The accuracy of training set is **0.995**
* validation set accuracy of ? 
* The accuracy of validation set is **0.958**
* test set accuracy of ?
* The accuracy of test set is **0.944**

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
> I chose the conventional LeNet model which used in the previous lesson.

* What were some problems with the initial architecture?
> The conventional LeNet model is easy to over fit.

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
> Firstly, I convert the images to grayscale, than I normalize the image data, so that the data has mean zero and equal variance. Finally, I add the dropout layer and set the keep_prob at 0.5.

* Which parameters were tuned? How were they adjusted and why?
> To speedup computation, I adjusted the epochs, batch size and learning_rate parameters. I add the keep_prob and normalize the image data for avoids over fitting.

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
> In order to avoid over fitting, I add the dropout layer and normalize the image data.


If a well known architecture was chosen:
* What architecture was chosen?
> The conventional LeNet model

* Why did you believe it would be relevant to the traffic sign application?
> The conventional LeNet model is work well in MNIST dataset, the current project is a classification problem which is very similar to the MNIST, so we can use a similar neural network model to solve this problem.

* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
> The accuracy of validation and test dataset are 95.8% and 94.6%, respectively, and these data prove that the model is working well.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are eight German traffic signs that I found on the web:

![alt text][image3]

The thied image (label: 42) might be difficult to classify because the unclear outline.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

Label | Image                 |     Prediction        |
-     | -                     | -                     |
0     | 20km/h                | 20km/h                |
1     | 30km/h                | 30km/h                |
2     | 50km/h                | 50km/h                |
12    | Priority road         | Priority road         |
17    | No entry              | No entryd             |
35    | Ahead only            | Ahead only            |
40    | Roundabout mandatory  | Roundabout mandatory  |
42    | End of no passing by vehicles over 3.5 metric tons | End of no passing by vehicles over 3.5 metric tons            |


The model was able to correctly guess 8 traffic signs, which gives an accuracy of 100%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The  top 5 softmax probabilities for each image are calculated.
```
TopKV2(values=array(
      [[ 19.25990868,   2.16702557,  -0.05496739,  -0.57856375, -0.76799887],
       [ 12.4054966 ,   6.67153358,   3.64173031,   2.48373556,  2.43848419],
       [  6.44235706,   2.22815633,   1.95378363,   0.59966576,  -0.7436313 ],
       [ 14.34313869,   9.63320923,   7.3887701 ,   2.70913577, 1.93182623],
       [  8.01793289,   6.10177708,   2.16000271,  -0.72416687, -0.81832039],
       [ 10.81099415,   6.10091209,   4.03429842,   3.97748399, -1.03867114],
       [ 29.55239296,  10.65698433,   5.80331326,   4.42131662, 1.80545092],
       [ 14.88151646,   6.96163654,   5.08525419,   2.43104386, 1.98111999]], dtype=float32), indices=array([[12, 40,  1,  7, 15],
       [ 2,  1,  5,  4, 15],
       [40,  7, 12, 37, 30],
       [42,  6, 41, 32, 12],
       [ 0,  1,  4, 40, 17],
       [ 1,  0,  4,  2,  8],
       [17,  0, 33, 34, 14],
       [35,  3,  9, 37, 13]], dtype=int32))

new_image_labels:  [12, 2, 40, 42, 0, 1, 17, 35]
```
The indices (class ids) of the top predictions of images exactly match the label of images, which proves our model work well in traffic sign classification problem. 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


