from tensorflow import keras

import PIL.Image
import os, glob,json
import numpy as np

import shutil

img_width, img_height = 224, 224
keras_param = "./model/vgg16.h5"
testpic = "/home/student/e18/e185701/sky_nonsky_ver2/sky_nonsky/FILE210928-071156-M 4.jpg"
files =[]
types=["jpg"]

def load_image(path):
    img = PIL.Image.open(path)
    img = img.convert('RGB')
    # 学習時に、(64, 64, 3)で学習したので、画像の縦・横は今回 変数imsizeの(64, 64)にリサイズします。
    img = img.resize((img_width,img_height))
    # 画像データをnumpy配列の形式に変更
    img = np.asarray(img)
    img = img / 255.0
    return img


model = keras.models.load_model(keras_param)

# img = load_image(testpic)
# prd = model.predict(np.array([img]))
# print(prd) # 精度の表示
# prelabel = np.argmax(prd, axis=1)

# if prelabel == 0:
#     print(">>> 空あり")
# elif prelabel == 1:
#     print(">>> 空なし")

photos_dir = "/home/student/e18/e185701/sky_nonsky_ver2/sky_nonsky/210929"
for ext in types:
  file_path = os.path.join(photos_dir, '*.{}'.format(ext))
  files.extend(glob.glob(file_path))

for i in files:
    img = load_image(i)
    prd = model.predict(np.array([img]))
    print(prd) # 精度の表示
    prelabel = np.argmax(prd, axis=1)

    if prelabel == 0:
        shutil.move(i,"/home/student/e18/e185701/sky_nonsky_ver2/sky_nonsky/空あり")
        print(">>> 空あり")
    elif prelabel == 1:
        shutil.move(i,"/home/student/e18/e185701/sky_nonsky_ver2/sky_nonsky/空なし")
        print(">>> 空なし")