
# coding: utf-8

# ## Import Modules
import cv2
import os
import numpy as np
import random
from datetime import datetime
import argparse
import time

# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf
import tensorflow.contrib.slim as slim

# custom functions
from dataset import Dataset
from tools import *

# ## Parse Arguments
# arguments for training model
parser = argparse.ArgumentParser(description='Hyperparameter for training a Mobilenet V2')
parser.add_argument('--input_width',      help='Rescale the image in x-axis.', type=int, default=192)
parser.add_argument('--input_height',     help='Rescale the image in y-axis.', type=int, default=192)
parser.add_argument('--batchsize',        help='Size of the batches.',         type=int, default=16)
parser.add_argument('--gpu',              help='Id of the GPU to use (as reported by nvidia-smi).', type=str, default="0")
parser.add_argument('--epoch',            help='Number of epochs to train.',   type=int, default=30)
parser.add_argument('--save_feq',         help='Frequency of saving checkpoints in epochs.', type=int, default=10)
parser.add_argument('--checkpoint_path',  help='Path to sotre snapshots of models during training.', type=str, default='./checkpoints')
parser.add_argument('--lr',               help='Learning rate.',                   type=float, default=1e-3)
parser.add_argument('--lr_decay_rate',    help='Decay rate of the learning rate.', type=float, default=0.95)
parser.add_argument('--random_transform', help='Able/Disable data augmentation.',  type=bool, default=False)
args = parser.parse_known_args()[0]

# use this environment flag to change which GPU to use
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


# ## Preprocess the Image Data
def pad_and_reshape(img, coord, target_shape=(args.input_width, args.input_height)):
    # compute how much padding need.
    h,w,_ = img.shape
    max_dim = max(h, w)
    delta_w = max_dim - w
    delta_h = max_dim - h
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    # pad the image to match ratio of target shape
    resized_img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0,0,0])
    draw = resized_img.copy()
    scale = max_dim / target_shape[0]

    # resize the image with target shape
    resized_img = cv2.resize(resized_img, target_shape)
    
    # pad the coordinate to match ratio of target shape
    for i in range(coord.shape[0]):
        coord[i, 0] += left 
        coord[i, 1] += top
    
    # resize the coordinate with the computed scale
    resized_coord = coord.copy()
    for i in range(resized_coord.shape[0]):
        resized_coord[i, 0] = int(resized_coord[i, 0] / scale)
        resized_coord[i, 1] = int(resized_coord[i, 1] / scale)
    
    return resized_img, resized_coord


def render_gaussian_heatmap(coord):
    sigmas = 3
    
    input_shape  = [args.input_height, args.input_width]
    output_shape = [args.input_height//2, args.input_width//2]
    num_kps = coord.shape[1]
    
    x = [i for i in range(output_shape[1])]
    y = [i for i in range(output_shape[0])]
    xx, yy = tf.meshgrid(x, y)
    xx = tf.reshape(tf.to_float(xx), (1, output_shape[0], output_shape[1], 1))
    yy = tf.reshape(tf.to_float(yy), (1, output_shape[0], output_shape[1], 1))
    
    x = tf.floor(tf.reshape(coord[:, :, 0], [-1, 1, 1, num_kps]) / input_shape[1] * output_shape[1] + 0.5)
    y = tf.floor(tf.reshape(coord[:, :, 1], [-1, 1, 1, num_kps]) / input_shape[0] * output_shape[0] + 0.5)
    
    heatmap = tf.exp(-(((xx-x)/tf.to_float(sigmas))**2)/tf.to_float(2)
                     -(((yy-y)/tf.to_float(sigmas))**2)/tf.to_float(2))
    
    return heatmap
    
    
def generate_batch(train_data, train_indices, step, batchsize=args.batchsize):
    image_lst = []
    coord_lst = []
    flags_lst = []
    for j in train_indices[step * batchsize:(step+1) * batchsize]:
        im2read = cv2.imread(train_data[j]['image_path'])
        joints = np.array(train_data[j]['joints']).reshape(14, 3)
        keypoints = joints[:, :2].astype(np.float32)
        flags = joints[:, 2].astype(np.float32)
        reshaped_image, reshaped_keypoint = pad_and_reshape(im2read, keypoints)            
                
        image_lst.append(reshaped_image)
        coord_lst.append(reshaped_keypoint)
        flags_lst.append(flags)
                
    np_image = np.array(image_lst, dtype=np.float32)
    np_coord = np.array(coord_lst, dtype=np.float32)
    np_flags = np.array(flags_lst, dtype=np.float32)
            
    feed_dict = dict()
    feed_dict[input_image] = np_image
    feed_dict[input_coord] = np_coord
    feed_dict[input_flags] = np_flags
    
    return feed_dict
    


# ## Call Dataset
d = Dataset()
train_data = d.load_frame_data(task='train')


# ## Base Function of Mobilenet V2
_init_xavier = tf.contrib.layers.xavier_initializer()
_init_norm = tf.truncated_normal_initializer(stddev=0.01)
_init_zero = slim.init_ops.zeros_initializer()
_l2_regularizer_00004 = tf.contrib.layers.l2_regularizer(0.00004)
_trainable = True


def is_trainable(trainable=True):
    global _trainable
    _trainable = trainable
    

def max_pool(inputs, k_h, k_w, s_h, s_w, name, padding="SAME"):
    return tf.nn.max_pool(inputs,
                          ksize=[1, k_h, k_w, 1],
                          strides=[1, s_h, s_w, 1],
                          padding=padding,
                          name=name)


def upsample(inputs, factor, name):
    return tf.image.resize_bilinear(inputs, [int(inputs.get_shape()[1]) * factor, int(inputs.get_shape()[2]) * factor],
                                    name=name)


def convb(input, k_h, k_w, c_o, stride, name, relu=True):
    with slim.arg_scope([slim.batch_norm], decay=0.999, fused=True, is_training=_trainable):
        output = slim.convolution2d(
            inputs=input,
            num_outputs=c_o,
            kernel_size=[k_h, k_w],
            stride=stride,
            normalizer_fn=slim.batch_norm,
            weights_regularizer=_l2_regularizer_00004,
            weights_initializer=_init_xavier,
            biases_initializer=_init_zero,
            activation_fn=tf.nn.relu if relu else None,
            scope=name,
            trainable=_trainable)
    return output


def separable_conv(input, c_o, k_s, stride, scope):
    with slim.arg_scope([slim.batch_norm],
                        decay=0.999,
                        fused=True,
                        is_training=_trainable,
                        activation_fn=tf.nn.relu6):
        output = slim.separable_convolution2d(input,
                                              num_outputs=None,
                                              stride=stride,
                                              trainable=_trainable,
                                              depth_multiplier=1.0,
                                              kernel_size=[k_s, k_s],
                                              weights_initializer=_init_xavier,
                                              weights_regularizer=_l2_regularizer_00004,
                                              biases_initializer=None,
                                              scope=scope + '_depthwise')

        output = slim.convolution2d(output,
                                    c_o,
                                    stride=1,
                                    kernel_size=[1, 1],
                                    weights_initializer=_init_xavier,
                                    biases_initializer=_init_zero,
                                    normalizer_fn=slim.batch_norm,
                                    trainable=_trainable,
                                    weights_regularizer=None,
                                    scope=scope + '_pointwise')

    return output


def inverted_bottleneck(inputs, up_channel_rate, channels, subsample, k_s=3, scope=""):
    with tf.variable_scope("inverted_bottleneck_%s" % scope):
        with slim.arg_scope([slim.batch_norm],
                            decay=0.999,
                            fused=True,
                            is_training=_trainable,
                            activation_fn=tf.nn.relu6):
            stride = 2 if subsample else 1

            output = slim.convolution2d(inputs,
                                        up_channel_rate * inputs.get_shape().as_list()[-1],
                                        stride=1,
                                        kernel_size=[1, 1],
                                        weights_initializer=_init_xavier,
                                        biases_initializer=_init_zero,
                                        normalizer_fn=slim.batch_norm,
                                        weights_regularizer=None,
                                        scope=scope + '_up_pointwise',
                                        trainable=_trainable)

            output = slim.separable_convolution2d(output,
                                                  num_outputs=None,
                                                  stride=stride,
                                                  depth_multiplier=1.0,
                                                  kernel_size=k_s,
                                                  weights_initializer=_init_xavier,
                                                  weights_regularizer=_l2_regularizer_00004,
                                                  biases_initializer=None,
                                                  padding="SAME",
                                                  scope=scope + '_depthwise',
                                                  trainable=_trainable)

            output = slim.convolution2d(output,
                                        channels,
                                        stride=1,
                                        kernel_size=[1, 1],
                                        activation_fn=None,
                                        weights_initializer=_init_xavier,
                                        biases_initializer=_init_zero,
                                        normalizer_fn=slim.batch_norm,
                                        weights_regularizer=None,
                                        scope=scope + '_pointwise',
                                        trainable=_trainable)
            if inputs.get_shape().as_list()[-1] == channels:
                output = tf.add(inputs, output)

    return output


# ## MobileNet V2 Network

N_KPOINTS = 14
STAGE_NUM = 6
out_channel_ratio = lambda d: max(int(d * 0.75), 8)
up_channel_ratio = lambda d: int(d * 1.)
out_channel_cpm = lambda d: max(int(d * 0.75), 8)


def cpm_mobilenet_v2(input, trainable):
    is_trainable(trainable)
    
    net = convb(input, 3, 3, out_channel_ratio(32), 2, name="Conv2d_0")
    with tf.variable_scope('MobilenetV2'):

        # 128, 112
        mv2_branch_0 = slim.stack(net, inverted_bottleneck,
                                  [
                                      (1, out_channel_ratio(16), 0, 3),
                                      (1, out_channel_ratio(16), 0, 3)
                                  ], scope="MobilenetV2_part_0")

        # 64, 56
        mv2_branch_1 = slim.stack(mv2_branch_0, inverted_bottleneck,
                                  [
                                      (up_channel_ratio(6), out_channel_ratio(24), 1, 3),
                                      (up_channel_ratio(6), out_channel_ratio(24), 0, 3),
                                      (up_channel_ratio(6), out_channel_ratio(24), 0, 3),
                                      (up_channel_ratio(6), out_channel_ratio(24), 0, 3),
                                      (up_channel_ratio(6), out_channel_ratio(24), 0, 3),
                                  ], scope="MobilenetV2_part_1")

        # 32, 28
        mv2_branch_2 = slim.stack(mv2_branch_1, inverted_bottleneck,
                                  [
                                      (up_channel_ratio(6), out_channel_ratio(32), 1, 3),
                                      (up_channel_ratio(6), out_channel_ratio(32), 0, 3),
                                      (up_channel_ratio(6), out_channel_ratio(32), 0, 3),
                                      (up_channel_ratio(6), out_channel_ratio(32), 0, 3),
                                      (up_channel_ratio(6), out_channel_ratio(32), 0, 3),
                                  ], scope="MobilenetV2_part_2")

        # 16, 14
        mv2_branch_3 = slim.stack(mv2_branch_2, inverted_bottleneck,
                                  [
                                      (up_channel_ratio(6), out_channel_ratio(64), 1, 3),
                                      (up_channel_ratio(6), out_channel_ratio(64), 0, 3),
                                      (up_channel_ratio(6), out_channel_ratio(64), 0, 3),
                                      (up_channel_ratio(6), out_channel_ratio(64), 0, 3),
                                      (up_channel_ratio(6), out_channel_ratio(64), 0, 3),
                                  ], scope="MobilenetV2_part_3")

        # 8, 7
        mv2_branch_4 = slim.stack(mv2_branch_3, inverted_bottleneck,
                                  [
                                      (up_channel_ratio(6), out_channel_ratio(96), 1, 3),
                                      (up_channel_ratio(6), out_channel_ratio(96), 0, 3),
                                      (up_channel_ratio(6), out_channel_ratio(96), 0, 3),
                                      (up_channel_ratio(6), out_channel_ratio(96), 0, 3),
                                      (up_channel_ratio(6), out_channel_ratio(96), 0, 3)
                                  ], scope="MobilenetV2_part_4")

        cancat_mv2 = tf.concat(
            [
                max_pool(mv2_branch_0, 4, 4, 4, 4, name="mv2_0_max_pool"),
                max_pool(mv2_branch_1, 2, 2, 2, 2, name="mv2_1_max_pool"),
                mv2_branch_2,
                upsample(mv2_branch_3, 2, name="mv2_3_upsample"),
                upsample(mv2_branch_4, 4, name="mv2_4_upsample")
            ]
            , axis=3)

    with tf.variable_scope("Convolutional_Pose_Machine"):
        l2s = []
        prev = None
        for stage_number in range(STAGE_NUM):
            if prev is not None:
                inputs = tf.concat([cancat_mv2, prev], axis=3)
            else:
                inputs = cancat_mv2

            kernel_size = 7
            lastest_channel_size = 128
            if stage_number == 0:
                kernel_size = 3
                lastest_channel_size = 512

            _ = slim.stack(inputs, inverted_bottleneck,
                           [
                               (2, out_channel_cpm(32), 0, kernel_size),
                               (up_channel_ratio(4), out_channel_cpm(32), 0, kernel_size),
                               (up_channel_ratio(4), out_channel_cpm(32), 0, kernel_size),
                           ], scope="stage_%d_mv2" % stage_number)

            _ = slim.stack(_, separable_conv,
                           [
                               (out_channel_ratio(lastest_channel_size), 1, 1),
                               (N_KPOINTS, 1, 1)
                           ], scope="stage_%d_mv1" % stage_number)

            prev = _
            cpm_out = upsample(_, 4, "stage_%d_out" % stage_number)
            l2s.append(cpm_out)

    return cpm_out, l2s


# ## Placeholder
input_image = tf.placeholder(tf.float32, [None, args.input_height, args.input_width, 3])    # [batchsize, H, W, C]
input_coord = tf.placeholder(tf.float32, [None, 14, 2])         # [batchsize, num_kps, 2]
input_flags = tf.placeholder(tf.float32, [None, 14])            # [batchsize, num_kps]
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(float(args.lr), global_step,
        decay_steps=10000, decay_rate=float(args.lr_decay_rate), staircase=True)


# ## Output
input_heatmaps = render_gaussian_heatmap(input_coord)
with tf.variable_scope(tf.get_variable_scope(), reuse=False):
    _, pred_heatmaps_all = cpm_mobilenet_v2(input_image, True)


# ## Loss
losses = []
for idx, pred_heat in enumerate(pred_heatmaps_all):
    reshaped_flags = tf.reshape(input_flags, [-1, 1, 1, 14])
    loss_l2 = tf.nn.l2_loss((tf.concat(pred_heat, axis=0) - input_heatmaps) * reshaped_flags, name='loss_heatmap_stage%d' % idx)
    losses.append(loss_l2)

total_loss = tf.reduce_sum(losses) / args.batchsize
total_loss_ll_heat = tf.reduce_sum(loss_l2) / args.batchsize


# ## Optimization
optim = tf.train.AdamOptimizer(learning_rate, epsilon = 1e-8)
grads = optim.compute_gradients(total_loss)
apply_gradients_op = optim.apply_gradients(grads, global_step=global_step)

MOVING_AVERAGE_DECAY = 0.99
variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
variable_to_average = (tf.trainable_variables() + tf.moving_average_variables())
variables_averages_op = variable_averages.apply(variable_to_average)

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_op = tf.group(apply_gradients_op, variables_averages_op)


# ## Create session before training
sess = get_session()

# initialize variables
sess.run(tf.global_variables_initializer())

# prepare to train
train_indices = np.arange(len(train_data))
num_steps = len(train_indices) // args.batchsize


# ## Start Training

for epoch in range(args.epoch):
    # shuffle all elements per epoch
    shuffled_indices = train_indices.copy()
    get_rng().shuffle(shuffled_indices)
        
    for i in range(num_steps):
        feed_dict = generate_batch(train_data, shuffled_indices, i)
        start = time.time()
        _, loss, loss_lastlayer_heat = sess.run([train_op, total_loss, total_loss_ll_heat], 
                                            feed_dict = feed_dict)
        duration = time.time() - start
    
    # print train info per epoch
    print('epoch: %d, loss = %.2f, last_heat_loss = %.2f, duration = %.4f' % (epoch, loss, loss_lastlayer_heat, duration))      

    # visualize ground truth heatmap and output heatmap
    background = feed_dict[input_image][0]
    flags_info = feed_dict[input_flags][0]
    heatmaps, outputs = sess.run([input_heatmaps, pred_heat], feed_dict = feed_dict)  
    visualize_gt_and_output(background, heatmaps[0], outputs[0], flags_info)

cv2.destroyAllWindows()

