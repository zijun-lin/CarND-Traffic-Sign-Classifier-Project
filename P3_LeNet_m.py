import pickle
import random
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from tensorflow.contrib.layers import flatten
import tensorflow as tf
import numpy as np
import cv2
import os

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

# Visualize Data
index = random.randint(0, len(X_train))
image = X_train[index].squeeze()
print('label: ', y_train[index])
# plt.figure(figsize=(4, 4))
# plt.imshow(image, cmap='gray')
# plt.show()

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

X_train, y_train = shuffle(X_train, y_train)
X_train = process_image(X_train)
X_valid = process_image(X_valid)
X_test = process_image(X_test)

epochs = 40
batch_size = 128
prob = 0.5


def LeNet(x):
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1

    # TODO: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x12.
    w1 = tf.Variable(tf.truncated_normal([5, 5, 1, 12], mean=mu, stddev=sigma))
    b1 = tf.Variable(tf.zeros([12]))
    st1 = [1, 1, 1, 1]
    layer1 = tf.nn.bias_add(tf.nn.conv2d(x, w1, st1, padding='VALID'), b1)
    # TODO: Activation.
    layer1 = tf.nn.relu(layer1)
    layer1 = tf.nn.dropout(layer1, keep_prob)
    # TODO: Pooling. Input = 28x28x12. Output = 14x14x12.
    p_ks1 = [1, 2, 2, 1]
    p_st1 = [1, 2, 2, 1]
    layer1 = tf.nn.max_pool(layer1, p_ks1, p_st1, padding='VALID')

    # TODO: Layer 2: Convolutional. Iutput = 14x14x12 Output = 10x10x24.
    w2 = tf.Variable(tf.truncated_normal([5, 5, 12, 24], mean=mu, stddev=sigma))
    b2 = tf.Variable(tf.zeros([24]))
    st2 = [1, 1, 1, 1]
    layer2 = tf.nn.bias_add(tf.nn.conv2d(layer1, w2, st2, padding='VALID'), b2)
    # TODO: Activation.
    layer2 = tf.nn.relu(layer2)
    layer2 = tf.nn.dropout(layer2, keep_prob)
    # TODO: Pooling. Input = 10x10x24. Output = 5x5x24.
    p_ks2 = [1, 2, 2, 1]
    p_st2 = [1, 2, 2, 1]
    layer2 = tf.nn.max_pool(layer2, p_ks2, p_st2, padding='VALID')

    ##############################################################
    # TODO: Flatten. Input = 5x5x24. Output = 600.
    f_layer = flatten(layer2)
    ##############################################################

    # TODO: Layer 0: Fully Connected. Input = 600. Output = 400.
    f0_w = tf.Variable(tf.truncated_normal([600, 400], mean=mu, stddev=sigma))
    f0_b = tf.Variable(tf.truncated_normal([400]))
    f0 = tf.add(tf.matmul(f_layer, f0_w), f0_b)
    # TODO: Activation.
    f0 = tf.nn.relu(f0)
    f0 = tf.nn.dropout(f0, keep_prob)

    # TODO: Layer 1: Fully Connected. Input = 400. Output = 120.
    f1_w = tf.Variable(tf.truncated_normal([400, 120], mean=mu, stddev=sigma))
    f1_b = tf.Variable(tf.truncated_normal([120]))
    f1 = tf.add(tf.matmul(f0, f1_w), f1_b)
    # TODO: Activation.
    f1 = tf.nn.relu(f1)
    f1 = tf.nn.dropout(f1, keep_prob)

    # TODO: Layer 2: Fully Connected. Input = 120. Output = 84.
    f2_w = tf.Variable(tf.truncated_normal([120, 84], mean=mu, stddev=sigma))
    f2_b = tf.Variable(tf.truncated_normal([84]))
    f2 = tf.add(tf.matmul(f1, f2_w), f2_b)
    # TODO: Activation.
    f2 = tf.nn.relu(f2)
    f2 = tf.nn.dropout(f2, keep_prob)

    # TODO: Layer 3: Fully Connected. Input = 84. Output = 43.
    f3_w = tf.Variable(tf.truncated_normal([84, 43], mean=mu, stddev=sigma))
    f3_b = tf.Variable(tf.truncated_normal([43]))
    f3 = tf.add(tf.matmul(f2, f3_w), f3_b)

    logits = f3
    return logits


# Features and Labels
x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
keep_prob = tf.placeholder(tf.float32)
one_hot_y = tf.one_hot(y, 43)

# Training Pipeline
rate = 0.001
logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=rate)
training_operation = optimizer.minimize(loss_operation)

# Model Evaluation
corrext_prediction = tf.equal(tf.argmax(logits, 1),
                              tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(corrext_prediction, tf.float32))
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


# TODO Train the Model
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)

    print("Training...")
    print()
    for i in range(epochs):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, batch_size):
            end = offset + batch_size
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: prob})

        validation_accuracy = evaluate(X_valid, y_valid)
        print("EPOCH {} ...".format(i + 1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()

    saver.save(sess, './lenet')
    print("Model saved")


with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))

    test_accuracy = evaluate(X_test, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))
