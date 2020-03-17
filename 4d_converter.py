import tensorflow as tf
tf.enable_eager_execution() 
import argparse
import numpy as np
from tensorflow.keras.layers import Add, Conv2D, Input, Lambda
from tensorflow.keras.models import Model
import os
from Super_Reslution_util import resolve_single
from PIL import Image
import cv2







parser = argparse.ArgumentParser()


parser.add_argument('--videoDir', type=str, default='')

args = parser.parse_args()


if args.videoDir == '':
    print("please set video directory")


weights_dir = './Super_Reslution_util/weight'
new_im = np.zeros((512,512,3)).astype(np.uint8)
videoDir = args.videoDir

DIV2K_RGB_MEAN = np.array([0.4488, 0.4371, 0.4040]) * 255


# model definetion

def edsr(scale, num_filters=64, num_res_blocks=8, res_block_scaling=None):
    """Creates an EDSR model."""
    x_in = Input(shape=(None, None, 3))
    x = Lambda(normalize)(x_in)

    x = b = Conv2D(num_filters, 3, padding='same')(x)
    for i in range(num_res_blocks):
        b = res_block(b, num_filters, res_block_scaling)
    b = Conv2D(num_filters, 3, padding='same')(b)
    x = Add()([x, b])

    x = upsample(x, scale, num_filters)
    x = Conv2D(3, 3, padding='same')(x)

    x = Lambda(denormalize)(x)
    return Model(x_in, x, name="edsr")


def res_block(x_in, filters, scaling):
    """Creates an EDSR residual block."""
    x = Conv2D(filters, 3, padding='same', activation='relu')(x_in)
    x = Conv2D(filters, 3, padding='same')(x)
    if scaling:
        x = Lambda(lambda t: t * scaling)(x)
    x = Add()([x_in, x])
    return x


def upsample(x, scale, num_filters):
    def upsample_1(x, factor, **kwargs):
        """Sub-pixel convolution."""
        x = Conv2D(num_filters * (factor ** 2), 3, padding='same', **kwargs)(x)
        return Lambda(pixel_shuffle(scale=factor))(x)

    if scale == 2:
        x = upsample_1(x, 2, name='conv2d_1_scale_2')
    elif scale == 3:
        x = upsample_1(x, 3, name='conv2d_1_scale_3')
    elif scale == 4:
        x = upsample_1(x, 2, name='conv2d_1_scale_2')
        x = upsample_1(x, 2, name='conv2d_2_scale_2')

    return x


def pixel_shuffle(scale):
    return lambda x: tf.nn.depth_to_space(x, scale)


def normalize(x):
    return (x - DIV2K_RGB_MEAN) / 127.5


def denormalize(x):
    return x * 127.5 + DIV2K_RGB_MEAN

# load pre trained weights
edsr_fine_tuned = edsr(scale=4, num_res_blocks=16)
edsr_fine_tuned.load_weights(os.path.join(weights_dir, 'weights-edsr-16-x4-fine-tuned.h5'))


# convert video to frames

vidcap = cv2.VideoCapture(videoDir)
success,image = vidcap.read()
images = [] 
while success:
  images.append(image)
  success,image = vidcap.read()

# increase reslution for each frame to 2d

HDimages = [] 
height = 0 
width = 0 
for i  in range(len(images)) :
  shape = images[i].shape 
  height , width = shape[:-1]
  new_im[:height,:width] = images[i]
  sr_ft = resolve_single(edsr_fine_tuned, new_im)
  im2a = sr_ft.numpy()
  cropped = im2a[:height*4,:width*4]
  #croppedIm = Image.fromarray(cropped)
  HDimages.append(cropped)


# convert frames to video

size = (width*4,height*4)
out = cv2.VideoWriter('HDvideo.mp4',cv2.VideoWriter_fourcc(*'DIVX'),1,size)
for i in range(len(HDimages)):
    out.write(HDimages[i])
out.release()





