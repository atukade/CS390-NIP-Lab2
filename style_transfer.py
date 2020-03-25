#!/usr/bin/env python
# coding: utf-8

# In[76]:


import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
import random
from scipy.misc import imsave, imresize
from scipy.optimize import fmin_l_bfgs_b   # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_l_bfgs_b.html
from tensorflow.keras.applications import vgg19
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import warnings


# In[8]:


random.seed(1618)
np.random.seed(1618)
tf.set_random_seed(1618)

tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# In[135]:


CONTENT_IMG_PATH = "nature_resize.jpg"           #TODO: Add this.
STYLE_IMG_PATH = "starry_resize.jpg"             #TODO: Add this.


CONTENT_IMG_H = 500
CONTENT_IMG_W = 500

STYLE_IMG_H = 500
STYLE_IMG_W = 500

CONTENT_WEIGHT = 0.005    # Alpha weight.
STYLE_WEIGHT = 100.0      # Beta weight.
TOTAL_WEIGHT = 100.0

TRANSFER_ROUNDS = 10

f_outputs = None
# width, height = load_img(CONTENT_IMG_PATH).size
# img_nrows = 500
# img_ncols = int(width * img_nrows / height)


# In[136]:


#=============================<Helper Fuctions>=================================
'''
TODO: implement this.
This function should take the tensor and re-convert it to an image.
'''
def deprocessImage(img):
    if K.image_data_format() == 'channels_first':
        img = img.reshape((3, CONTENT_IMG_H, CONTENT_IMG_W))
        img = img.transpose((1, 2, 0))
    else:
        img = img.reshape((CONTENT_IMG_H, CONTENT_IMG_W, 3))
    # Remove zero-center by mean pixel
    img[:, :, 0] += 103.939
    img[:, :, 1] += 116.779
    img[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    img = img[:, :, ::-1]
    img = np.clip(img, 0, 255).astype('uint8')
    return img


def gramMatrix(x):
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram


# In[137]:


#========================<Loss Function Builder Functions>======================

def styleLoss(style, gen):
    return K.sum(K.square(gramMatrix(style) - gramMatrix(gen))) / (4.0 * (3**2) * ((STYLE_IMG_H * STYLE_IMG_W)**2))  #TODO: implement.


def contentLoss(content, gen):
    return K.sum(K.square(gen - content))


def totalLoss(x):
    assert K.ndim(x) == 4
    if K.image_data_format() == 'channels_first':
        a = K.square(
            x[:, :, :CONTENT_IMG_H - 1, :CONTENT_IMG_H - 1] - x[:, :, 1:, :CONTENT_IMG_H - 1])
        b = K.square(
            x[:, :, :CONTENT_IMG_H - 1, :CONTENT_IMG_H - 1] - x[:, :, :CONTENT_IMG_H - 1, 1:])
    else:
        a = K.square(
            x[:, :CONTENT_IMG_H - 1, :CONTENT_IMG_H - 1, :] - x[:, 1:, :CONTENT_IMG_H - 1, :])
        b = K.square(
            x[:, :CONTENT_IMG_H - 1, :CONTENT_IMG_H - 1, :] - x[:, :CONTENT_IMG_H - 1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))


# In[138]:


#=========================<Pipeline Functions>==================================

def getRawData():
    print("   Loading images.")
    print("      Content image URL:  \"%s\"." % CONTENT_IMG_PATH)
    print("      Style image URL:    \"%s\"." % STYLE_IMG_PATH)
    cImg = load_img(CONTENT_IMG_PATH)
    tImg = cImg.copy()
    sImg = load_img(STYLE_IMG_PATH)
    print("      Images have been loaded.")
    return ((cImg, CONTENT_IMG_H, CONTENT_IMG_W), (sImg, STYLE_IMG_H, STYLE_IMG_W), (tImg, CONTENT_IMG_H, CONTENT_IMG_W))



def preprocessData(raw):
    img, ih, iw = raw
    img = img_to_array(img)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        img = imresize(img, (ih, iw, 3))
    img = img.astype("float64")
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return img


# In[139]:


'''
TODO: Allot of stuff needs to be implemented in this function.
First, make sure the model is set up properly.
Then construct the loss function (from content and style loss).
Gradient functions will also need to be created, or you can use K.Gradients().
Finally, do the style transfer with gradient descent.
Save the newly generated and deprocessed images.
'''
def styleTransfer(cData, sData, tData):
    
    global f_outputs
    print("   Building transfer model.")
    
    contentTensor = K.variable(cData)
    styleTensor = K.variable(sData)
    
    genTensor = K.placeholder((1, CONTENT_IMG_H, CONTENT_IMG_W, 3))
    inputTensor = K.concatenate([contentTensor, styleTensor, genTensor], axis=0)
    
    model = vgg19.VGG19(include_top=False, weights='imagenet', input_tensor=inputTensor)   #TODO: implement.
    
    outputDict = dict([(layer.name, layer.output) for layer in model.layers])
    print("   VGG19 model loaded.")
    loss = K.variable(0.0)
    styleLayerNames = ["block1_conv1", "block2_conv1", "block3_conv1", "block4_conv1", "block5_conv1"]
    contentLayerName = "block5_conv2"
    print("   Calculating content loss.")
    contentLayer = outputDict[contentLayerName]
    contentOutput = contentLayer[0, :, :, :]
    genOutput = contentLayer[2, :, :, :]
    
    loss = loss + CONTENT_WEIGHT * contentLoss(contentOutput, genOutput)
    
    print("   Calculating style loss.")
    for layerName in styleLayerNames:
        layer_features = outputDict[layerName]
        style_features = layer_features[1, :, :, :]
        gen_features = layer_features[2, :, :, :]
        s = styleLoss(style_features, gen_features)
        loss = loss + (STYLE_WEIGHT / len(styleLayerNames)) * s
    loss = loss + TOTAL_WEIGHT * totalLoss(genTensor)
    
    # TODO: Setup gradients or use K.gradients().
    
    grads = K.gradients(loss, genTensor)[0]
    #grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
    
    outputs = [loss]
    
    if isinstance(grads, (list, tuple)):
        outputs += grads
    else:
        outputs.append(grads)

    f_outputs = K.function([genTensor], outputs)

    
    #iterate = K.function([genTensor], [loss, grads])    
    
    print("   Beginning transfer.")
    
    x = tData
    
    for i in range(TRANSFER_ROUNDS):
        print("   Step %d." % i)
        #TODO: perform gradient descent using fmin_l_bfgs_b.
#         outs = f_outputs([cData])
        
#         loss_value = outs[0]
#         if len(outs[1:]) == 1:
#             print("In here")
#             grad_values = outs[1].flatten().astype('float64')
#         else:
#             grad_values = np.array(outs[1:]).flatten().astype('float64')
        cData, min_val, info = fmin_l_bfgs_b(evaluator.loss, cData.flatten(), fprime=evaluator.grads, maxfun=20)
        print('Current loss value:', min_val)
        img = deprocessImage(cData.copy())
        saveFile = 'my_result' + '_at_iteration_%d.png' % i
        imsave(saveFile, img)
        print("      Image saved to \"%s\"." % saveFile)
    print("   Transfer complete.")

class Evaluator(object):

    def __init__(self):
        self.loss_value = None
        self.grads_values = None

    def loss(self, x):
        assert self.loss_value is None
        loss_value, grad_values = eval_loss_and_grads(x)
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values


evaluator = Evaluator()

def eval_loss_and_grads(x):
    if K.image_data_format() == 'channels_first':
        x = x.reshape((1, 3, CONTENT_IMG_H, CONTENT_IMG_H))
    else:
        x = x.reshape((1, CONTENT_IMG_H, CONTENT_IMG_H, 3))
    outs = f_outputs([x])
    loss_value = outs[0]
    if len(outs[1:]) == 1:
        grad_values = outs[1].flatten().astype('float64')
    else:
        grad_values = np.array(outs[1:]).flatten().astype('float64')
    return loss_value, grad_values


# In[ ]:


#=========================<Main>================================================

def main():
    print("Starting style transfer program.")
    raw = getRawData()
    cData = preprocessData(raw[0])   # Content image.
    sData = preprocessData(raw[1])   # Style image.
    tData = preprocessData(raw[2])   # Transfer image.
    styleTransfer(cData, sData, tData)
    print("Done. Goodbye.")



if __name__ == "__main__":
    main()


# In[ ]:





# In[ ]:




