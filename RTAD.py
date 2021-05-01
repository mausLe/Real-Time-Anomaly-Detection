import tensorflow as tf
import os
import time
import numpy as np
import pickle
import scipy.misc
from PIL import Image
import cv2
import glob
import random
from models import generator
from utils import DataLoader, load, save, psnr_error
from constant import const
import evaluate
import matplotlib.pyplot as plt
import darknet


slim = tf.contrib.slim

os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = const.GPU

dataset_name = const.DATASET
test_folder = const.TEST_FOLDER

num_his = const.NUM_HIS
height, width = 256, 256

snapshot_dir = const.SNAPSHOT_DIR
psnr_dir = const.PSNR_DIR
evaluate_name = const.EVALUATE
batch = []
print(const)


# define dataset
with tf.name_scope('dataset'):
    test_video_clips_tensor = tf.placeholder(shape=[1, height, width, 3 * (num_his + 1)],
                                             dtype=tf.float32)
    test_inputs = test_video_clips_tensor[..., 0:num_his*3]
    test_gt = test_video_clips_tensor[..., -3:]
    print('test inputs = {}'.format(test_inputs))
    print('test prediction gt = {}'.format(test_gt))

# define testing generator function and
# in testing, only generator networks, there is no discriminator networks and flownet.
with tf.variable_scope('generator', reuse=None):
    print('testing = {}'.format(tf.get_variable_scope().name))
    test_outputs = generator(test_inputs, layers=4, output_channel=3)
    test_psnr_error,shape = psnr_error(gen_frames=test_outputs, gt_frames=test_gt)
    print(shape,[shape])


config = tf.ConfigProto()
config.gpu_options.allow_growth = True

def np_load_frame(filename, resize_height, resize_width):
    """
    Load image path and convert it to numpy.ndarray. Notes that the color channels are BGR and the color space
    is normalized from [0, 255] to [-1, 1].

    :param filename: the full path of image
    :param resize_height: resized height
    :param resize_width: resized width
    :return: numpy.ndarray
    """
    image_decoded = cv2.imread(filename)
    image_resized = cv2.resize(image_decoded, (resize_width, resize_height))
    image_resized = image_resized.astype(dtype=np.float32)
    image_resized = (image_resized / 127.5) - 1.0
    return image_resized



def generatePSNR(frame, count):
    print("\n\n" + "-"*40)
    # length = video['length']
    # print("\nNum_his {}, length {}".format(num_his, length))
    # video_clip = data_loader.get_video_clips(count, i - num_his, i + 1)
    video_clip = []
    # print("\n\nBATCH: ", batch)
    print("Batch size", len(batch))
    if len(batch) < 4:
        image = np_load_frame(frame, resize_height=256, resize_width=256)
        batch.append(image)
        return None
    elif len(batch) == 4:
        image = np_load_frame(frame, resize_height=256, resize_width=256)
        batch.append(image)
    elif len(batch) == 5:
        pass
    else:
        for i in range(5, len(batch)):
            batch.pop(0)

    print("\n\nBATCH: ", len(batch))
    video_clip = np.concatenate(batch, axis=2)
    print("\n\nVideo Clip: ", video_clip.shape)
    # print("\n\nVideo Clip 2: ", video_clip[np.newaxis, ...])
    
    batch.pop(0)
    psnr = sess.run(test_psnr_error, feed_dict={test_video_clips_tensor: video_clip[np.newaxis, ...]})
    print("psnr: ", psnr)
    sh = sess.run(test_outputs,
                      feed_dict={test_video_clips_tensor: video_clip[np.newaxis, ...]})
    # if not os.path.isdir("Train_Data/" + count):
    #     os.mkdir("Train_Data/" + count)
    im = Image.fromarray((sh[0,:,:,:]*255).astype(np.uint8))
    im.save("Train_Data/" + str(count).zfill(6) + ".jpg")
    im2 = Image.fromarray((video_clip[...,-3:]*255).astype(np.uint8))
    im2.save("Train_Data/" + str(count).zfill(6) + "_gt"+".jpg")
    print("count: {} psnr {}".format(count, psnr))
    
    print("PSNR : ", psnr)
    return psnr

def image_detection(image_path, network, class_names, class_colors, thresh):
    # Darknet doesn't accept numpy images.
    # Create one with image we reuse for each detect
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    darknet_image = darknet.make_image(width, height, 3)

    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (width, height),
                               interpolation=cv2.INTER_LINEAR)

    darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
    detections = darknet.detect_image(network, class_names, darknet_image, thresh=thresh)
    darknet.free_image(darknet_image)
    image = darknet.draw_boxes(detections, image_resized, class_colors)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), detections

def convert2relative(image, bbox):
    """
    YOLO format use relative coordinates for annotation
    """
    x, y, w, h = bbox
    height, width, _ = image.shape
    return x/width, y/height, w/width, h/height

def save_annotations(name, image, detections, class_names):
    """
    Files saved with image_name.txt and relative coordinates
    """
    file_name = os.path.splitext(name)[0] + ".txt"
    with open(file_name, "w") as f:
        for label, confidence, bbox in detections:
            x, y, w, h = convert2relative(image, bbox)
            label = class_names.index(label)
            f.write("{} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}\n".format(label, x, y, w, h, float(confidence)))

sess = tf.compat.v1.Session(config=config)

# Initialize for Tensorflow
restore_var = [v for v in tf.global_variables()]
loader = tf.train.Saver(var_list=restore_var)
ckpt = const.SNAPSHOT_DIR
load(loader, sess, ckpt)


# Yolo Initializer
config_file = "/content/test/cfg/yolov4.cfg"
weights = "/content/test/yolov4.weights"
data_file = "/content/test/cfg/coco.data"
batch_size = 1
thresh = 0.25 # remove object with confidence score < 0.25

save_labels = False # do not export output to .txt file

random.seed(3)  # deterministic bbox colors
network, class_names, class_colors = darknet.load_network(
    config_file,
    data_file,
    weights,
    batch_size=batch_size
)

print("-"*40)
# images = sorted(glob.glob("/content/Data/ped2/testing/frames/01/*.jpg"))
images = sorted(glob.glob("/content/Data/test/*.jpg"))

count = 0
# print("Images: ", images)

for imageDir in images:
    prev_time = time.time()

    print("dir ", imageDir)
    # Run GAN to calc MSE
    generatePSNR(imageDir, count)
    """
    # Run yolo to detect object and export infos
    image, detections = image_detection(imageDir, network, 
    class_names, class_colors, thresh)

    if save_labels:
        save_annotations(imageDir, image, detections, class_names)
    """
    fps = int(1/(time.time() - prev_time))
    print("FPS: {}".format(fps))
    count += 1

sess.close()
