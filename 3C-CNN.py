import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import models, layers,optimizers,regularizers
from tensorflow.keras.applications import VGG16,ResNet50
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Flatten, Dense,Dropout,GlobalAveragePooling2D,AveragePooling2D,Cropping2D,ZeroPadding2D,Conv2D,MaxPooling2D,concatenate
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ReduceLROnPlateau,EarlyStopping, ModelCheckpoint
import os , datetime
classes = ['空あり','空なし'] #分類するクラス
nb_classes = len(classes)

train_data_dir = './dataset/train'
validation_data_dir = './dataset/val'
test_data_dir = './dataset/test2'
model_dir = "./model"

nb_train_samples = 1400
nb_validation_samples = 200
img_width, img_height = 224, 224

train_batch_size = 16
val_batch_size = 16

# train用
train_datagen = ImageDataGenerator(rescale=1. / 255,
# rotation_range=15,
# width_shift_range=10,
# horizontal_flip=True,
)

# validation
val_datagen = ImageDataGenerator(rescale=1. / 255)

# test 用
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
  train_data_dir,
  target_size=(img_height, img_width),
  color_mode='rgb',
  classes=classes,
  class_mode='categorical',
  batch_size= train_batch_size,# 1回のバッチ生成で作る画像数
  shuffle=True,
  )
 
validation_generator = val_datagen.flow_from_directory(
  validation_data_dir,
  target_size=(img_height,img_width),
  color_mode='rgb',
  classes=classes,
  class_mode='categorical',
  batch_size=val_batch_size,# 1回のバッチ生成で作る画像数
  )

test_generator = test_datagen.flow_from_directory(
  test_data_dir,
  target_size=(img_height,img_width),
  color_mode='rgb',
  classes=classes,
  class_mode='categorical',
  batch_size=val_batch_size # 1回のバッチ生成で作る画像数
  )

def generator_two_img():
    genX1 = train_generator
    genX2 = train_generator
    while True:
        X1i = genX1.next()
        X2i = genX2.next()
        yield [X1i[0], X2i[0]], X1i[1]

global_input_tensor = Input(shape=(img_height,img_width,3))
top_input_tensor = Input(shape=(img_height,img_width,3))
bottom_input_tensor = Input(shape=(img_height,img_width,3))

# ----------------------------------------------------
vgg16 = VGG16(include_top=False, weights='imagenet',input_tensor=global_input_tensor)
vgg16.trainable = False

# resnet50 = ResNet50(include_top=False, weights='imagenet',input_tensor=global_input_tensor)
# resnet50.trainable = False

# x=vgg16.output
# x = GlobalAveragePooling2D()(x)
# # let's add a fully-connected layer
# x = Dense(1024, activation='relu')(x)
# # and a logistic layer -- let's say we have 200 classes
# predictions = Dense(nb_classes, activation='softmax')(x)
global_model = Model(global_input_tensor, vgg16.output)
# global_model = Model(resnet50.input, resnet50.output)

# 上領域---------------------------------------------------
top_model = Cropping2D(cropping=((0,112), (0,0)))(top_input_tensor)
top_model = ZeroPadding2D(padding=(3, 3))(top_model)
top_model = Conv2D(32, (7, 7),strides=(1, 2),activation='relu')(top_model)
top_model = ZeroPadding2D(padding=(1,1))(top_model)
top_model = MaxPooling2D((3, 3),strides=(2, 2))(top_model)

top_model =Conv2D(128, (3, 3),padding='same',activation='relu')(top_model)

top_model = ZeroPadding2D(padding=(1,1))(top_model)
top_model = Conv2D(256,(3, 3),strides=(2, 2), activation='relu')(top_model)

top_model = ZeroPadding2D(padding=(1,1))(top_model)
top_model = Conv2D(512,(3, 3),strides=(2, 2), activation='relu')(top_model)

top_model = ZeroPadding2D(padding=(1,1))(top_model)
top_model = Conv2D(1024,(3, 3),strides=(2, 2), activation='relu')(top_model)
top_model = Model(top_input_tensor,top_model)

# -----------------------------------------------------

#下領域 ---------------------------------------------------
bottom_model = Cropping2D(cropping=((112,0), (0,0)))(bottom_input_tensor)
bottom_model = ZeroPadding2D(padding=(3, 3))(bottom_model)
bottom_model = Conv2D(32, (7, 7),strides=(1, 2),activation='relu')(bottom_model)
bottom_model = ZeroPadding2D(padding=(1,1))(bottom_model)
bottom_model = MaxPooling2D((3, 3),strides=(2, 2))(bottom_model)

bottom_model =Conv2D(128, (3, 3),padding='same',activation='relu')(bottom_model)

bottom_model = ZeroPadding2D(padding=(1,1))(bottom_model)
bottom_model = Conv2D(256,(3, 3),strides=(2, 2), activation='relu')(bottom_model)

bottom_model = ZeroPadding2D(padding=(1,1))(bottom_model)
bottom_model = Conv2D(512,(3, 3),strides=(2, 2), activation='relu')(bottom_model)

bottom_model = ZeroPadding2D(padding=(1,1))(bottom_model)
bottom_model = Conv2D(1024,(3, 3),strides=(2, 2), activation='relu')(bottom_model)
bottom_model = Model(bottom_input_tensor,bottom_model)

# -----------------------------------------------------

model = concatenate([bottom_model.output,top_model.output],axis=3)
model = AveragePooling2D((7,7))(model)
model = Flatten()(model)
prediction = Dense(nb_classes, activation='softmax')(model)

model = Model([bottom_model.input,top_model.input],prediction)
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=0.001, momentum=0.9,decay=0.0002),
              metrics=['accuracy'])
model.summary()

plot_model(model, show_shapes=True,show_layer_names=False,to_file='model.png')

history = model.fit_generator(
  generator = generator_two_img(), 
  steps_per_epoch = nb_train_samples // train_batch_size,
#   validation_data = [validation_generator,validation_generator],
#   validation_steps = nb_validation_samples // val_batch_size,
  epochs=1,
#   callbacks=[early_stopping,reduce_lr]
)