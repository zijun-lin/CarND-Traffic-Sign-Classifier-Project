import pickle
import random
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from tensorflow.contrib.layers import flatten
import tensorflow as tf
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


# TODO Step 2: Design and Test a Model Architecture¶
X_train, y_train = shuffle(X_train, y_train)
epochs = 20
batch_size = 128


def LeNet(x):
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1

    # TODO: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    w1 = tf.Variable(tf.truncated_normal([5, 5, 3, 6], mean=mu, stddev=sigma))
    b1 = tf.Variable(tf.truncated_normal([6]))
    st1 = [1, 1, 1, 1]
    layer1 = tf.nn.bias_add(tf.nn.conv2d(x, w1, st1, padding='VALID'), b1)
    # TODO: Activation.
    layer1 = tf.nn.relu(layer1)
    # TODO: Pooling. Input = 28x28x6. Output = 14x14x6.
    p_ks1 = [1, 2, 2, 1]
    p_st1 = [1, 2, 2, 1]
    layer1 = tf.nn.max_pool(layer1, p_ks1, p_st1, padding='VALID')

    # TODO: Layer 2: Convolutional. Output = 10x10x16.
    w2 = tf.Variable(tf.truncated_normal([5, 5, 6, 16], mean=mu, stddev=sigma))
    b2 = tf.Variable(tf.truncated_normal([16]))
    st2 = [1, 1, 1, 1]
    layer2 = tf.nn.bias_add(tf.nn.conv2d(layer1, w2, st2, padding='VALID'), b2)
    # TODO: Activation.
    layer2 = tf.nn.relu(layer2)
    # TODO: Pooling. Input = 10x10x16. Output = 5x5x16.
    p_ks2 = [1, 2, 2, 1]
    p_st2 = [1, 2, 2, 1]
    layer2 = tf.nn.max_pool(layer2, p_ks2, p_st2, padding='VALID')

    # TODO: Flatten. Input = 5x5x16. Output = 400.
    f0 = flatten(layer2)

    # TODO: Layer 3: Fully Connected. Input = 400. Output = 120.
    f1_w = tf.Variable(tf.truncated_normal([400, 120], mean=mu, stddev=sigma))
    f1_b = tf.Variable(tf.truncated_normal([120]))
    f1 = tf.add(tf.matmul(f0, f1_w), f1_b)
    # TODO: Activation.
    f1 = tf.nn.relu(f1)

    # TODO: Layer 4: Fully Connected. Input = 120. Output = 84.
    f2_w = tf.Variable(tf.truncated_normal([120, 84], mean=mu, stddev=sigma))
    f2_b = tf.Variable(tf.truncated_normal([84]))
    f2 = tf.add(tf.matmul(f1, f2_w), f2_b)
    # TODO: Activation.
    f2 = tf.nn.relu(f2)

    # TODO: Layer 5: Fully Connected. Input = 84. Output = 43.
    f3_w = tf.Variable(tf.truncated_normal([84, 43], mean=mu, stddev=sigma))
    f3_b = tf.Variable(tf.truncated_normal([43]))
    f4 = tf.add(tf.matmul(f2, f3_w), f3_b)

    logits = f4
    return logits


# Features and Labels
x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int32, (None))
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
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples


# TODO Train the Model
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
#             sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
#
#         validation_accuracy = evaluate(X_valid, y_valid)
#         print("EPOCH {} ...".format(i + 1))
#         print("Validation Accuracy = {:.3f}".format(validation_accuracy))
#         print()
#
#     saver.save(sess, './lenet')
#     print("Model saved")


with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))

    test_accuracy = evaluate(X_test, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))





