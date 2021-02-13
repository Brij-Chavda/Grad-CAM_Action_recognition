from tensorflow.keras.layers import Lambda
from tensorflow.keras.models import Sequential
from tensorflow.python.framework import ops
import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np
from tensorflow import keras
import sys
import cv2


from tensorflow.keras.layers import Lambda
from tensorflow.keras.models import Sequential
from tensorflow.python.framework import ops
import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np
from tensorflow import keras
import sys
import cv2

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Lambda
from tensorflow.keras.layers import Conv3D, MaxPooling3D, ZeroPadding3D
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K



from tensorflow.keras.layers import Input

import numpy as np
import cv2

C3D_MEAN_PATH = '/content/drive/MyDrive/txt_c3d/train01_16_128_171_mean.npy'
SPORTS1M_CLASSES_PATH = '/content/drive/MyDrive/txt_c3d/labels.txt'

def preprocess_input(video):
    """Resize and subtract mean from video input
    
    Keyword arguments:
    video -- video frames to preprocess. Expected shape 
        (frames, rows, columns, channels). If the input has more than 16 frames
        then only 16 evenly samples frames will be selected to process.
    
    Returns:
    A numpy array.
    
    """
    intervals = np.ceil(np.linspace(0, len(video)-1, 16)).astype(int)
    print(intervals)
    frames = [video[i] for i in intervals]
    shape  = np.shape(frames)
    
    # Reshape to 128x171
    reshape_frames = np.zeros((shape[0], 128, 171, shape[3]))
    for i, img in enumerate(frames):
        #img = imresize(img, (128,171), 'bicubic')
        img = cv2.resize(img, (171,128), interpolation = cv2.INTER_AREA)
        print(np.shape(img))
        reshape_frames[i,:,:,:] = img
        
    
    
    # Subtract mean
    mean = np.load(C3D_MEAN_PATH)
    mean = np.reshape(mean, np.shape(reshape_frames))
    reshape_frames -= mean
    # Crop to 112x112
    reshape_frames = reshape_frames[:,8:120,30:142,:]
    # Add extra dimension for samples
    reshape_frames = np.expand_dims(reshape_frames, axis=0)
    
    return reshape_frames

def decode_predictions(preds):
    """Returns class label and confidence of top predicted answer
    
    Keyword arguments:
    preds -- numpy array of class probability
    
    Returns:
    A list of tuples.
    
    """
    class_pred = []
    for x in range(preds.shape[0]):
        class_pred.append(np.argmax(preds[x]))
    
    
    
    with open(SPORTS1M_CLASSES_PATH, 'r') as f:
        labels = [lines.strip() for lines in f]
        
    decoded = [(labels[x],preds[i,x]) for i,x in enumerate(class_pred)]
    
    return decoded

def target_category_loss(x, category_index, nb_classes):
    return tf.multiply(x, K.one_hot([category_index], nb_classes))

def target_category_loss_output_shape(input_shape):
    return input_shape

def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)



def grad_cam(model, image, category_index, layer_name):
    '''model = Sequential()
    model.add(input_model)'''

    nb_classes = 487
    target_layer = lambda x: target_category_loss(x, category_index, nb_classes)
    #model.add(Lambda(target_layer,output_shape = target_category_loss_output_shape))

    n_model = tf.keras.Model(inputs= model.input, outputs= model.layers[11].output)
    

    with tf.GradientTape() as tape:
      output, conv_output =m4(image)
      class_output=tf.reduce_sum(target_category_loss(output,category_index,nb_classes))
      #conv_output = n_model(image)
      #loss = K.sum([model.layers[-1].output])
      #conv_output =  model.layers[-9].output
    variables = model.trainable_variables
    grads = (tape.gradient(class_output, conv_output))
    weights = grads[0]
    w = normalize(weights)
    
    print(np.shape(grads))
    
    
    output = conv_output[0]
    print(np.shape(output))
    
    #weights = np.mean(weights, axis = (1, 2))
    w = np.mean(w, axis = (0, 1, 2))
    print(np.shape(weights))
    cam = np.ones(output.shape[0 : 3], dtype = np.float32)
    #cam2 = np.ones(output.shape[1 : 3], dtype = np.float32)
    print(np.shape(w))
    print(np.shape(output))
    for i in range(512):
        cam += w[i] * output[: ,:, :, i]


    cam1 = cv2.resize(np.float32(cam[0]), (112, 112), cv2.INTER_AREA)
    cam1 = np.maximum(cam1, 0)
    heatmap1 = cam1 / np.max(cam1)

    cam2 = cv2.resize(np.float32(cam[1]), (112, 112), cv2.INTER_AREA)
    cam2 = np.maximum(cam2, 0)
    heatmap2 = cam2 / np.max(cam2)

    cam3 = cv2.resize(np.float32(cam[2]), (112, 112), cv2.INTER_AREA)
    cam3 = np.maximum(cam3, 0)
    heatmap3 = cam3 / np.max(cam3)

    cam4 = cv2.resize(np.float32(cam[3]), (112, 112), cv2.INTER_AREA)
    cam4 = np.maximum(cam4, 0)
    heatmap4 = cam4 / np.max(cam4)

    print(np.shape(image))
    #Return to BGR [0..255] from the preprocessed image
    image = image[0]
    img1, img2 = image[1], image[4]
    img1 -= np.min(img1)
    img1 = np.minimum(img1, 255)
    img2 -= np.min(img2)
    img2 = np.minimum(img2, 255)
    img3, img4 = image[8], image[11]
    img3 -= np.min(img3)
    img3 = np.minimum(img3, 255)
    img4 -= np.min(img4)
    img4 = np.minimum(img4, 255)

    cam1 = cv2.applyColorMap(np.uint8(255*heatmap1), cv2.COLORMAP_JET)
    cam1 = np.float32(cam1) + np.float32(img1)
    cam1 = 255 * cam1 / np.max(cam1)

    cam2 = cv2.applyColorMap(np.uint8(255*heatmap2), cv2.COLORMAP_JET)
    cam2 = np.float32(cam2) + np.float32(img2)
    cam2 = 255 * cam2 / np.max(cam2)

    cam3 = cv2.applyColorMap(np.uint8(255*heatmap3), cv2.COLORMAP_JET)
    cam3 = np.float32(cam3) + np.float32(img3)
    cam3 = 255 * cam3 / np.max(cam3)

    cam4 = cv2.applyColorMap(np.uint8(255*heatmap4), cv2.COLORMAP_JET)
    cam4 = np.float32(cam4) + np.float32(img4)
    cam4 = 255 * cam4 / np.max(cam4)
    cam1, cam2, cam3, cam4 = cv2.cvtColor(cam1, cv2.COLOR_BGR2RGB), cv2.cvtColor(cam2, cv2.COLOR_BGR2RGB), cv2.cvtColor(cam3, cv2.COLOR_BGR2RGB), cv2.cvtColor(cam4, cv2.COLOR_BGR2RGB)
    return np.uint8(cam1), heatmap1, np.uint8(cam2), heatmap2, np.uint8(cam3), heatmap3, np.uint8(cam4), heatmap4
#predictions = c3d_model.predict(preprocessed_input)
#top_1 = decode_predictions(predictions)[0][0]
#print('Predicted class:')
#print('%s (%s) with probability %.2f' % (top_1[1], top_1[0], top_1[2]))
predictions = m2.predict(inp1)
predicted_class = np.argmax(predictions)
cam1, heatmap1, cam2, heatmap2,  cam3, heatmap3, cam4, heatmap4 = grad_cam(m1, inp1, predicted_class, "conv5b")
fol = 'gradcam_4_avg' + str(folder_name)
import matplotlib.pyplot as plt
import skimage
import skimage.io as imshow
plt.figure(figsize = (8,8))
plt.subplot(1,4,1)
plt.imshow(cam1)
plt.subplot(1,4,2)
plt.imshow(cam2)
plt.subplot(1,4,3)
plt.imshow(cam3)
plt.subplot(1,4,4)
plt.imshow(cam4)
f_name = '/content/' + str(fol) + '.png'
plt.savefig(f_name) 
FILE.download(f_name)
#!rm -r {unzip_name}
