from PIL import Image
import os, glob
import numpy as np
import random
from PIL import ImageFile
import json

from numpy.lib.function_base import select
from numpy.lib.function_base import append
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import datetime,os
from keras.preprocessing.image import array_to_img, img_to_array, load_img, save_img

# モデル読み込み用
from keras.models import load_model
#混合行列計算用
from sklearn.metrics import confusion_matrix,f1_score,recall_score,precision_score

# IOError: image file is truncated (0 bytes not processed)回避のため
ImageFile.LOAD_TRUNCATED_IMAGES = True

classes = ["空あり", "空なし"] #0:clear 1:rain
num_classes = len(classes)
test_data_dir = './dataset/train'

img_width, img_height = 224, 224

train_batch_size = 64
val_batch_size = 1600

test_datagen = ImageDataGenerator(
  # rescale=1. / 255
  # rotation_range=15,
  width_shift_range=0.5,
  zoom_range= 0.5,
  # horizontal_flip=True,
)

test_generator = test_datagen.flow_from_directory(
  test_data_dir,
  target_size=(img_height, img_width),
  color_mode='rgb',
  classes=classes,
  class_mode='categorical',
  batch_size=val_batch_size # 1回のバッチ生成で作る画像数
  )

batch = test_generator.next()
x, y_test = batch
y_test=np.argmax(y_test,axis=1)

x= np.asarray(x,dtype='uint8')#flot32→uint8 変換
y_test = np.asarray(y_test)


y = y_test
np.savez("./dataset/sky_nonsky_aug", x,y)