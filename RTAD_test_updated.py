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
    # print("\n\n" + "-"*40)
    # length = video['length']
    # print("\nNum_his {}, length {}".format(num_his, length))
    # video_clip = data_loader.get_video_clips(count, i - num_his, i + 1)
    video_clip = []
    # print("\n\nBATCH: ", batch)
    # print("Batch size", len(batch))
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

    # print("\n\nBATCH: ", len(batch))
    video_clip = np.concatenate(batch, axis=2)
    # print("\n\nVideo Clip: ", video_clip.shape)
    # print("\n\nVideo Clip 2: ", video_clip[np.newaxis, ...])
    
    batch.pop(0)
    psnr = sess.run(test_psnr_error, feed_dict={test_video_clips_tensor: video_clip[np.newaxis, ...]})
    # print("psnr: ", psnr)
    sh = sess.run(test_outputs,
                      feed_dict={test_video_clips_tensor: video_clip[np.newaxis, ...]})
    # if not os.path.isdir("Train_Data/" + count):
    #     os.mkdir("Train_Data/" + count)
    im = Image.fromarray((sh[0,:,:,:]*255).astype(np.uint8))
    im.save("Train_Data/" + str(count).zfill(6) + ".jpg")
    im2 = Image.fromarray((video_clip[...,-3:]*255).astype(np.uint8))
    im2.save("Train_Data/" + str(count).zfill(6) + "_gt"+".jpg")
    # print("count: {} psnr {}".format(count, psnr))
    
    # print("PSNR : ", psnr)
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


def createTrainFS (yoloData, psnrError):
    global Train_FS, trainingClasses
        
    # YOlo input ratio
    ratio = 256
    for i in range (4, len(yoloData)): # Pass the first 4 images
        mse =  1/psnrError[i] # Image frame error
        isObjectIn = False

        if len(yoloData[i]) != 0: # if yolo detect >= 1 object
            for obj in yoloData[i]:
                if obj['confidence'] >= 0.6: # Filter Class that it's confidence score < 0.6
                    if obj['name'] not in trainingClasses:
                        continue
                        trainingClasses.append(obj['name'])
                        print("New object at frame {} name {}".format(i, obj['name']))

                    # Filter outline objects
                    
                    
                    y = obj['relative_coordinates']['center_y']*ratio  # top left corner
                    x = obj['relative_coordinates']['center_x']*ratio # top left corner
                    h = obj['relative_coordinates']['height']*ratio # height
                    w = obj['relative_coordinates']['width']*ratio # width

                    cx = x + w/2
                    cy = y + h/2
                    area = w*h

                    probs = np.zeros((1,15))
                    probs[0,trainingClasses.index(obj['name'])] = obj['confidence']
                    fs = [mse,cx,cy,area] + list(probs[0,:])
                    Train_FS.append(fs)

                    isObjectIn = True

        if isObjectIn == False:
            cx = cy = area = 0
            probs = np.zeros((1,15))
            fs = [mse,cx,cy,area] + list(probs[0,:])
            Train_FS.append(fs)

    # return Train_FS


def createTestFS (yoloFrameData, psnrFrameError):
    """
    yoloFrameData :
    psnrFrameError : 
    """
    global Test_FS, testingClasses

    Frame_FS = [] # Contrain feature space of that frame

    mse = 1/psnrFrameError

    # filter bounding box size
    boundFilter = {'person' : [], 'skateboard' : [], 'bicycle' : [], 'skis' : []}
    
    # Yolo input ratio
    ratio = 357
    isObjectIn = False
    if len(yoloFrameData) != 0: # if yolo detect >= 1 object
        for obj in yoloFrameData:
            # print("my Object: ", obj)
            if float(obj[1]) >= 45: # Filter Class that it's confidence score < 0.6
                if obj[0] not in testingClasses:
                    # continue
                    testingClasses.append(obj[0])
                    print("New object at frame name {}".format(obj[0]))

                
                y = obj[2][1]  # top left corner
                x = obj[2][0] # top left corner
                h = obj[2][3] # height
                w = obj[2][2] # width
                # print("Class {} -  Score {} ".format(obj[0], obj[1]))

                x = x*256//608
                y = y*256//608
                h = h*256//608
                w = w*256//608
                
                cx = x + w/2
                cy = y + h/2
                area = w*h

                probs = np.zeros((1,15))
                probs[0,testingClasses.index(obj[0])] = float(obj[1])
                fs = [mse,cx,cy,area] + list(probs[0,:])
                Test_FS.append(fs)
                Frame_FS.append(fs)
                
                isObjectIn = True

    if isObjectIn == False:
        cx = cy = area = 0
        probs = np.zeros((1,15))
        fs = [mse,cx,cy,area] + list(probs[0,:])
        Test_FS.append(fs)
        Frame_FS.append(fs)

    return Frame_FS

def normalizeTrainFS(g_max, g_min):
    Train_FS_N = Train_FS*1
    for i in range(0,4):
        Train_FS_N[:,i] = (Train_FS[:,i] - g_min[i])/(g_max[i]-g_min[i])

    for i in range(1,4):                            
        Train_FS_N[:,i] = 0.2*(Train_FS_N[:,i])

    for i in range(4,5):                            
        Train_FS_N[:,i] = 0*(Train_FS_N[:,i])

    # print("Train_FS_N[10]: ", Train_FS_N[10])
    return Train_FS_N

def normalizeTestFS(Test_FrameFS, g_max, g_min):
    # print("Test_FrameFS: ", Test_FrameFS)
    # print("g_max", g_max)
    # print("g_min", g_min)


    for i in range (len(Test_FrameFS)):
        for m in range(0,4):
            # print("i m", i, m)
            Test_FrameFS[i][m] = (Test_FrameFS[i][m] - g_min[m])/(g_max[m]-g_min[m])

        for m in range(1,4):
            Test_FrameFS[i][m] = 0.2*(Test_FrameFS[i][m])

        for m in range(4,5):
            Test_FrameFS[i][m] = 0*(Test_FrameFS[i][m])

        for m in range(5,18):
            if Test_FrameFS[i][m] != 0:
                (Test_FrameFS[i][m]) = 0.9

    # print("Test_FS_N: ", Test_FrameFS)
    return Test_FrameFS

# Train the Training Feature Space with KNN distance
def knndis_tr(t,X_M):
    Mg = X_M.shape[1]
    
    dist = np.sum((np.transpose(np.matlib.repmat(t,Mg,1)) - X_M)**2,0) 
    dist = np.sort(dist)
    return sum(dist[1:11])

def knndis(t,X_M):
    Mg = X_M.shape[1]

    
    dist = np.sum((np.transpose(np.matlib.repmat(t,Mg,1)) - X_M)**2,0) 
    dist = np.sort(dist)
    return sum(dist[0:10])

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


count = 0
# print("Images: ", images)

# Initialize MONAD
# Create Training Feature Space
Train_FS = list()
trainingClasses = ["person"]

import json
with open('result_train_ucsd.json', 'r') as f:
    D = json.load(f)

train_objects = []
for i in range(len(D)):
    train_objects.append(D[i]["objects"])


psnr_train = np.load('ped2_train',allow_pickle=True)['psnr']
psnrError = []
for vid in psnr_train:
    for ps  in vid:
        psnrError.append(ps)


# Call this func to create Training Feature Space
createTrainFS(train_objects, psnrError)
Train_FS = np.array(Train_FS)
print("Training classes: ", trainingClasses)

g_min = np.min(Train_FS,0)
g_max = np.max(Train_FS,0)
print("g_max: ", g_max)
print("g_min: ", g_min)

# Normalize Feature Space
# Check the w1, w2, w3 on the paper
# But these number quite different from what on KevalDoshi notebook :/ 
Train_FS_N = normalizeTrainFS(g_max, g_min) 
print("Train_FS_N[250]: ", Train_FS_N[250])

np.random.shuffle(Train_FS_N)
print("Train_FS_N: ", Train_FS_N.shape)

# Create Testing Feature Space
Test_FS = list()
testingClasses = ['person', 'skateboard', 'bicycle', 'skis']


import numpy.matlib
errors = list()
np.random.shuffle(Train_FS_N)
Ng = 5000
Mg = Train_FS_N.shape[0] - 5000

X_N = Train_FS_N[0:Ng]
X_M = Train_FS_N[Ng:-1]

for i in range(Ng):
    
    e = (X_N[i,0]) + knndis_tr(np.transpose(X_N[i,1:-1]),np.transpose(X_M[:,1:-1]))
    errors.append(e)
    if i% 500 == 0:
        print(i,e)

Base_lm = np.sort(errors)[int(len(errors)*0.9)]

print("-"*40)
print("Training Train_FS succesful!")
print("Base_lm: ", Base_lm)

plt.plot(np.sort(errors)[1:Ng])
plt.savefig("/content/Train_FS_N.png")
plt.close()
count = 4

print("-"*40)
images = sorted(glob.glob("/content/test/ped2/testing/frames/04/*.jpg"))
# images = sorted(glob.glob("/content/Data/test/*.jpg"))


# Export video output
myImage = cv2.imread(images[0])
myHeight, myWidth, _ = myImage.shape

resultVid = cv2.VideoWriter('/content/output/vid/out04.avi', fourcc=cv2.VideoWriter_fourcc(*'XVID'), fps=25, frameSize=(myWidth, myHeight))

psnr = None
figData = [0]
ind = 0
for imageDir in images: # Apple Store stolen images[600:1000]
    prev_time = time.time()
    # print("dir ", imageDir)
    # Run GAN to calc MSE
    psnr = generatePSNR(imageDir, ind)

    # Run yolo to detect object and export infos
    image, detections = image_detection(imageDir, 
    network, class_names, class_colors, thresh)
    
    if ind > 64 and ind < 70:
        cv2.imwrite("/content/output/{}.jpg".format(ind), image)

    # image.save("/content/test/yolo_output/" + str(count).zfill(6) + ".jpg")
    # print("detection: ", detections)

    if psnr is None:
        fps = int(1/(time.time() - prev_time))
        print("FPS: {}".format(fps))
        continue

    frame_FS = createTestFS(detections, psnr)
    
    
    # print("Test FS: ", Test_FS[len(Test_FS) - 1])
    # print("Testing classes: ", testingClasses)
    # Normalization
    # normalized_FS = normalizeTestFS(Test_FS[len(Test_FS) - 1], g_max, g_min)
    
    
    t = list()
    normalized_FS = normalizeTestFS(frame_FS, g_max, g_min)
    # Test_FS[len(Test_FS) - 1].append(normalized_FS)
    
    # Calc testing score
    # frame_FS = normalized_FS

    for obj in normalized_FS:
        t.append(obj[0] + knndis(np.transpose(obj[1:]),np.transpose(Train_FS_N[Ng:-1, 1:])))

    dis = (np.max(t)) - 0.75  # IDK what 0.8 means :(
    # print("Test_FS Error: ", dis)
    figData.append(np.max((0,figData[ind] + dis)))
    
    if ind > 64 and ind < 70:

        print("\n\nframe_FS {}: ".format(ind+1))
        for fr in frame_FS:
          print(fr)

        print("\nDetection: ")
        for det in detections:
          if float(det[1]) > 40:
              print(det)
        print("\nfigData[{}] before: ".format(ind+1), figData[ind+1])

        # print("figData[{}] before: ".format(ind+1), figData[ind+1])


    if ind > 5:
        if figData[ind+1] - figData[ind] <=0:
            if figData[ind] - figData[ind-1] <=0:
                if figData[ind-1] - figData[ind-2] <=0:
                    figData[ind+1] = 0

    image = cv2.imread(imageDir)
    
    rgb_red = [0, 0, 255]
    color = rgb_green = [60, 179, 0]

    thickness=40*image.shape[0]//600
    starting_point  = (0, 0)
    ending_point  = (image.shape[1], image.shape[0])
    # if images.index(imageDir) > 65 and images.index(imageDir) < 69:
    if ind > 64 and ind < 70:
        print("figData[{}] after: ".format(ind+1), figData[ind+1])
    if figData[ind+1] > 0.8:
        color = rgb_red


    cv2.rectangle(image, (0, 0), ending_point, color, thickness)


    
    
    resultVid.write(image)
    
    

    ind += 1
    # if save_labels:
    #     save_annotations(imageDir, image, detections, class_names)
    fps = float(1/(time.time() - prev_time))
    # print("Image size: ", image.shape)
    # print("FPS: {}".format(fps))

print("\n\nind: ", ind)
resultVid.release()

figData.pop(0)
idx = np.where(np.array(figData)>0)[0]
for id1 in range(0,len(idx)-6):
    if idx[id1+6] - idx[id1] >= 10:
        figData[idx[id1]] = 0

for id2 in range(0,len(idx)-50):
    if idx[id2+50] - idx[id2] >= 54:
        figData[idx[id2]] = 0


#
sess.close()


sc = (figData-np.min(figData))/(np.max(figData)-np.min(figData))
labels = np.load('labels.npy')
test_label = labels[170*2 + 150:170*2 + 150+180]
plt.plot((figData-np.min(figData))/(np.max(figData)-np.min(figData)), label = "Detection")
plt.plot(test_label, label = "GT")
plt.legend()

plt.savefig("/content/Test_FS_N 04 with GT.png")
plt.close()

import random


sc = np.array(sc)


from sklearn import metrics
import scipy.io as scio

sc = np.where(sc > 0, 1, 0)

print("score: ", sc)
print("test_label: ", test_label)


fpr2, tpr2, thresholds = metrics.roc_curve(test_label, sc ,1)
# fpr2 = np.sort(np.append(fpr2,(0.45))) # We extrapolate a point so as to complete the ROC curve
# tpr2 = np.sort(np.append(tpr2,(1)))
plt.plot(fpr2, tpr2)
plt.savefig("/content/AUC Ped2 test frame 04 ODIT score - .png")
plt.close()
print('ODIT AUC:', metrics.auc(fpr2, tpr2))

print("auc: ", np.trapz(tpr2, fpr2))