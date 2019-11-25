import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.contrib.layers import flatten
import tensorflow as tf
import numpy as np
import cv2
import os
import glob

# 忽略读取mnist数据时的warning
old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)
# 忽略cpu的warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# TODO Step 2: Design and Test a Model Architecture
def process_image(dataset):
    n_imgs, img_height, img_width, _ = dataset.shape
    processed_dataset = np.zeros((n_imgs, img_height, img_width, 1))
    for idx in range(len(dataset)):
        img = dataset[idx]
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        processed_dataset[idx, :, :, 0] = (gray - 128.0) / 128.0
    return processed_dataset


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


# Parameters
epochs = 50
batch_size = 128

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


fig, axs = plt.subplots(2, 4, figsize=(4, 2))
fig.subplots_adjust(hspace=.2, wspace=.001)
axs = axs.ravel()

# TODO: The name of images represent the traffic sign labels
new_images = []
new_images_labels = []
for i, fimg in enumerate(glob.glob('new_images/*.ppm')):
    image = mpimg.imread(fimg)
    axs[i].axis('off')
    axs[i].imshow(image)
    print('Image Name: ', fimg)
    new_images.append(image)
    img_label = (fimg.strip('new_images/')).strip('.ppm')
    new_images_labels.append(int(img_label))

new_images = np.asarray(new_images)
new_images_normalized = process_image(new_images)
print('new_images: ', new_images_normalized.shape)
plt.show()


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.import_meta_graph('./lenet.meta')
    saver.restore(sess, "./lenet")
    new_images_accuracy = evaluate(new_images_normalized, new_images_labels)
    print("New Test Set Accuracy = {:.3f}".format(new_images_accuracy))
