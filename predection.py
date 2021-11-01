from tensorflow import keras

import PIL.Image
import os, glob,json
import numpy as np

import shutil

imsize = 224
keras_param = "./model/ResNet50.h5"
testpic = "/home/student/e18/e185701/sky_nonsky_ver2/sky_nonsky/b1e9ee0e-67e26f2e.jpg"

def load_image(path):
    img = PIL.Image.open(path)
    img = img.convert('RGB')
    # 学習時に、(64, 64, 3)で学習したので、画像の縦・横は今回 変数imsizeの(64, 64)にリサイズします。
    img = img.resize((imsize,imsize))
    # 画像データをnumpy配列の形式に変更
    img = np.asarray(img)
    img = img / 255.0
    return img


model = keras.models.load_model(keras_param)

img = load_image(testpic)
prd = model.predict(np.array([img]))
print(prd) # 精度の表示
prelabel = np.argmax(prd, axis=1)

if prelabel == 0:
    print(">>> 空あり")
elif prelabel == 1:
    print(">>> 空なし")