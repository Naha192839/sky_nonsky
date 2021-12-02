from PIL import Image
import os, glob
import numpy as np
from PIL import ImageFile
import json
from sklearn.model_selection import train_test_split

# IOError: image file is truncated (0 bytes not processed)回避のため
ImageFile.LOAD_TRUNCATED_IMAGES = True

classes = ["空あり", "空なし"] #0:clear 1:rain
num_classes = len(classes)
image_size = 224


X_train = []
y_train = []


files = []
for index, classlabel in enumerate(classes):
    photos_dir = "./dataset_kfold/train/" + classlabel
    files = glob.glob(photos_dir + "/*.jpg")
    for i, file in enumerate(files):
        print(i)
        image = Image.open(file)
        image = image.convert("RGB")
        image = image.resize((image_size, image_size))
        data = np.asarray(image)
        X_train.append(data)
        y_train.append(index)

    
print("空あり:"+str(y_train.count(0)))
print("空なし:"+str(y_train.count(1)))
#validation_splitで均等にとるためにtrainをシャッフル
shuffl_num = np.random.randint(0, 100)
np.random.seed(shuffl_num)
np.random.shuffle(X_train)
np.random.seed(shuffl_num)
np.random.shuffle(y_train)

# X_train,X_test,y_train,y_test =train_test_split(X_train,y_train,test_size=0.2)
# print(y_train)
# # #テストデータの確保 20%
# X_train = X_train[:int(len(X_train) * 0.8)]
# y_train = y_train[:int(len(y_train) * 0.8)]

# X_test = X_train[int(len(X_train) * 0.8):]
# y_test = y_train[int(len(y_train) * 0.8):]

print("訓練データ:"+str(len(y_train)))
# print("テストデータ:"+str(len(y_test)))

X_train = np.array(X_train)
# X_test  = np.array(X_test)
y_train = np.array(y_train)
# y_test  = np.array(y_test)

x = X_train
y = y_train
np.savez("./dataset_kfold/sky_nonsky", x,y)

# xy = (X_train, X_test, y_train, y_test)
# np.save("./dataset/sky_nonsky.npy", xy)