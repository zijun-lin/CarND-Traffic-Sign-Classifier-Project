import pickle
import random
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from tensorflow.contrib.layers import flatten
import tensorflow as tf
import numpy as np
import cv2
import os
import glob
import matplotlib.image as mpimg

# 忽略读取mnist数据时的warning
old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)
# 忽略cpu的warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Load pickled data
# TODO Step 0: Load The Data
# TODO: Fill this in based on where you saved the training and testing data
training_file = 'traffic_signs_data/train.p'
validation_file = 'traffic_signs_data/valid.p'
testing_file = 'traffic_signs_data/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

assert(len(X_train) == len(y_train))
assert(len(X_valid) == len(y_valid))
assert(len(X_test) == len(y_test))

print("Data Shape:     {}".format(X_train.shape))
print("Image Shape:    {}".format(X_train[0].shape))
print("Training Set:   {} samples".format(len(X_train)))
print("Validation Set: {} samples".format(len(X_valid)))
print("Test Set:       {} samples".format(len(X_test)))

# # Visualize Data
# f, ax = plt.subplots(2, 5, figsize=(12, 6))
# f.subplots_adjust(hspace = .2, wspace=.1)
# ax = ax.ravel()
# for i in range(10):
#     idx = random.randint(0, len(X_train))
#     image = X_train[idx].squeeze()
#     ax[i].imshow(image)
#     ax[i].set_title(y_train[idx])
# # plt.show()


# TODO Step 1: Dataset Summary & Exploration
### Replace each question mark with the appropriate value.
### Use python, pandas or numpy methods rather than hard coding the results
# TODO: Number of training examples
n_train = len(X_train)
# TODO: Number of validation examples
n_validation = len(X_valid)
# TODO: Number of testing examples.
n_test = len(X_test)
# TODO: What's the shape of an traffic sign image?
image_shape = X_train[0].shape
# TODO: How many unique classes/labels there are in the dataset.
n_classes = len(set(y_train))

print("Number of training examples =", n_train)
print("Number of validation examples =", n_validation)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)


# TODO Step 2: Design and Test a Model Architecture
def process_image(dataset):
    n_imgs, img_height, img_width, _ = dataset.shape
    processed_dataset = np.zeros((n_imgs, img_height, img_width, 1))
    for idx in range(len(dataset)):
        img = dataset[idx]
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        processed_dataset[idx, :, :, 0] = (gray - 128.0) / 128.0
    return processed_dataset


def process_image2(dataset):
    images = []
    for idx in range(len(dataset)):
        img_gray = cv2.cvtColor(dataset[idx], cv2.COLOR_RGB2GRAY)
        img = (img_gray - 128.0) / 128.0
        images.append(img)
    images = np.expand_dims(images, 4)
    return images


# Preprocess the dataset
X_train, y_train = shuffle(X_train, y_train)
print('Dataset shape before process: ', X_train.shape)
print('Dta type before process: ', X_train.dtype)
X_train_origin = X_train
X_train = process_image(X_train)
X_valid = process_image(X_valid)
X_test = process_image(X_test)
print('Dataset shape after process: ', X_train.shape)
print('Dta type before process: ', X_train.dtype)

# # I display 5 random images before and after data processing
# f, axs = plt.subplots(2, 5, figsize=(12, 6))
# f.subplots_adjust(hspace=.2, wspace=.1)
# axs = axs.ravel()
# for i in range(5):
#     idx = random.randint(0, len(X_train))
#     image_origin = X_train_origin[idx]
#     image_process = X_train[idx].squeeze()
#     axs[i].axis('off')
#     axs[i].imshow(image_origin)
#     axs[i].set_title(y_train[idx])
#     axs[i+5].axis('off')
#     axs[i+5].imshow(image_process, cmap='gray')
#     axs[i+5].set_title(y_train[idx])
# # plt.show()


# Arguments used for tf.truncated_normal, randomly defines
# variables for the weights and biases for each layer
mu = 0
sigma = 0.1
conv1_w = tf.Variable(tf.truncated_normal([5, 5, 1, 12], mean=mu, stddev=sigma))
conv1_b = tf.Variable(tf.zeros([12]))
conv2_w = tf.Variable(tf.truncated_normal([5, 5, 12, 24], mean=mu, stddev=sigma))
conv2_b = tf.Variable(tf.zeros([24]))


def LeNet(x):
    # TODO: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x12.
    conv1_layer = tf.nn.bias_add(tf.nn.conv2d(x, conv1_w, strides=[1, 1, 1, 1], padding='VALID'), conv1_b)
    conv1_layer = tf.nn.relu(conv1_layer)
    conv1_layer = tf.nn.dropout(conv1_layer, keep_prob)
    # TODO: Pooling. Input = 28x28x12. Output = 14x14x12.
    conv1_layer = tf.nn.max_pool(conv1_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # TODO: Layer 2: Convolutional. Iutput = 14x14x12 Output = 10x10x24.
    conv2_layer = tf.nn.bias_add(tf.nn.conv2d(conv1_layer, conv2_w, strides=[1, 1, 1, 1], padding='VALID'), conv2_b)
    conv2_layer = tf.nn.relu(conv2_layer)
    conv2_layer = tf.nn.dropout(conv2_layer, keep_prob)
    # TODO: Pooling. Input = 10x10x24. Output = 5x5x24.
    conv2_layer = tf.nn.max_pool(conv2_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # TODO: Flatten. Input = 5x5x24. Output = 600.
    fc_layer = flatten(conv2_layer)

    # TODO: Layer 0: Fully Connected. Input = 600. Output = 400.
    fc0_w = tf.Variable(tf.truncated_normal([600, 400], mean=mu, stddev=sigma))
    fc0_b = tf.Variable(tf.truncated_normal([400]))
    fc0_layer = tf.add(tf.matmul(fc_layer, fc0_w), fc0_b)
    fc0_layer = tf.nn.relu(fc0_layer)
    fc0_layer = tf.nn.dropout(fc0_layer, keep_prob)

    # TODO: Layer 1: Fully Connected. Input = 400. Output = 120.
    fc1_w = tf.Variable(tf.truncated_normal([400, 120], mean=mu, stddev=sigma))
    fc1_b = tf.Variable(tf.truncated_normal([120]))
    fc1_layer = tf.add(tf.matmul(fc0_layer, fc1_w), fc1_b)
    fc1_layer = tf.nn.relu(fc1_layer)
    fc1_layer = tf.nn.dropout(fc1_layer, keep_prob)

    # TODO: Layer 2: Fully Connected. Input = 120. Output = 84.
    fc2_w = tf.Variable(tf.truncated_normal([120, 84], mean=mu, stddev=sigma))
    fc2_b = tf.Variable(tf.truncated_normal([84]))
    fc2_layer = tf.add(tf.matmul(fc1_layer, fc2_w), fc2_b)
    fc2_layer = tf.nn.relu(fc2_layer)
    fc2_layer = tf.nn.dropout(fc2_layer, keep_prob)

    # TODO: Layer 3: Fully Connected. Input = 84. Output = 43.
    fc3_w = tf.Variable(tf.truncated_normal([84, 43], mean=mu, stddev=sigma))
    fc3_b = tf.Variable(tf.truncated_normal([43]))
    fc3_layer = tf.add(tf.matmul(fc2_layer, fc3_w), fc3_b)

    logits = fc3_layer
    return logits


# Features and Labels
x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
keep_prob = tf.placeholder(tf.float32)
one_hot_y = tf.one_hot(y, 43)

# Training Parameters
epochs = 40
batch_size = 128
prob = 0.5

# Training Pipeline
rate = 0.001
logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
# optimizer = tf.train.AdamOptimizer(learning_rate=rate)
# training_operation = optimizer.minimize(loss_operation)
training_operation = tf.train.AdamOptimizer(learning_rate=rate).minimize(loss_operation)

# Model Evaluation
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()


def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, batch_size):
        batch_x, batch_y = X_data[offset:offset+batch_size], y_data[offset:offset+batch_size]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 1})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples


# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.75  # 注意该值不要取过大，防止GPU不够；或者不设该值；最好不设置
# # TODO Train the Model
# # with tf.Session(config=config) as sess:
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     num_examples = len(X_train)
#
#     print("Training...")
#     print()
#     for i in range(epochs):
#         X_train, y_train = shuffle(X_train, y_train)
#         for offset in range(0, num_examples, batch_size):
#             end = offset + batch_size
#             batch_x, batch_y = X_train[offset:end], y_train[offset:end]
#             sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: prob})
#         validation_accuracy = evaluate(X_valid, y_valid)
#         print("EPOCH {} ...".format(i + 1))
#         print("Validation Accuracy = {:.3f}".format(validation_accuracy))
#         print()
#
#     saver.save(sess, './lenet_template')
#     print("Model saved")
#     tf.summary.FileWriter('./my_graph', sess.graph)
#     print("File writer")


with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))

    train_accuracy = evaluate(X_train, y_train)
    print("Test Accuracy = {:.3f}".format(train_accuracy))
    valid_accuracy = evaluate(X_valid, y_valid)
    print("Test Accuracy = {:.3f}".format(valid_accuracy))
    test_accuracy = evaluate(X_test, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))


### Visualize your network's feature maps here. Feel free to use as many code cells as needed.
# image_input: the test image being fed into the network to produce the feature maps
# tf_activation: should be a tf variable name used during your training procedure that represents the calculated state of a specific weight layer
# activation_min/max: can be used to view the activation contrast in more detail, by default matplot sets min and max to the actual min and max values of the output
# plt_num: used to plot out multiple different weight feature map sets on the same block, just extend the plt number for each new feature map entry
def outputFeatureMap(image_input, sess, tf_activation, activation_min=-1, activation_max=-1, plt_num=1):
    # Here make sure to preprocess your image_input in a way your network expects
    # with size, normalization, ect if needed
    # image_input =
    # Note: x should be the same name as your network's tensorflow data placeholder variable
    # If you get an error tf_activation is not defined it maybe having trouble accessing the variable from inside a function
    activation = tf_activation.eval(session=sess, feed_dict={x: image_input})
    featuremaps = activation.shape[3]
    print('featuremaps: ', featuremaps)
    plt.figure(plt_num, figsize=(15, 15))
    for featuremap in range(featuremaps):
        plt.subplot(4, 6, featuremap+1)  # sets the number of feature maps to show on each row and column
        plt.title('FeatureMap ' + str(featuremap))  # displays the feature map number
        if activation_min != -1 & activation_max != -1:
            plt.imshow(activation[0, :, :, featuremap], interpolation="nearest", vmin =activation_min, vmax=activation_max, cmap="gray")
        elif activation_max != -1:
            plt.imshow(activation[0, :, :, featuremap], interpolation="nearest", vmax=activation_max, cmap="gray")
        elif activation_min != -1:
            plt.imshow(activation[0, :, :, featuremap], interpolation="nearest", vmin=activation_min, cmap="gray")
        else:
            plt.imshow(activation[0, :, :, featuremap], interpolation="nearest", cmap="gray")


with tf.Session() as sess:
    # sess.run(tf.global_variables_initializer())
    saver2 = tf.train.import_meta_graph('./lenet_template.meta')
    saver2.restore(sess, "./lenet_template")
    image = np.float32(X_train[0])
    print("Image shape: ", image.shape)

    plt.imshow(image.squeeze(), cmap='gray')

    image_input = np.reshape(image, [1, 32, 32, 1])
    print("Image shape: ", image_input.shape)

    # TODO: Layer 1: Convolutional. Input = 32x32x1. ConvOutput = 28x28x12. PoolOutput = 14x14x12.
    conv1_layer = tf.nn.bias_add(tf.nn.conv2d(image_input, conv1_w, strides=[1, 1, 1, 1], padding='VALID'), conv1_b)
    conv1_layer = tf.nn.relu(conv1_layer)
    conv1_layer = tf.nn.dropout(conv1_layer, 1)
    conv1_layer = tf.nn.max_pool(conv1_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    # TODO: Layer 2: Convolutional. Iutput = 14x14x12 ConvOutput = 10x10x24. PoolOutput = 5x5x24.
    conv2_layer = tf.nn.bias_add(tf.nn.conv2d(conv1_layer, conv2_w, strides=[1, 1, 1, 1], padding='VALID'), conv2_b)
    conv2_layer = tf.nn.relu(conv2_layer)
    conv2_layer = tf.nn.dropout(conv2_layer, 1)
    conv2_layer = tf.nn.max_pool(conv2_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    outputFeatureMap(image_input, sess, conv2_layer)
    plt.show()