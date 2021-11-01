import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import models, layers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Flatten, Dense
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ReduceLROnPlateau,EarlyStopping

classes = ['空あり','空なし'] #分類するクラス
nb_classes = len(classes)
train_data_dir = './dataset/train'
validation_data_dir = './dataset/val'
test_data_dir = './dataset/test'
nb_train_samples = 1400
nb_validation_samples = 201 
img_width, img_height = 224, 224

train_batch_size = 16
val_batch_size = 16
# train, validation 用
train_datagen = ImageDataGenerator(rescale=1.0 / 255)

# validation
val_datagen = ImageDataGenerator(rescale=1.0 / 255)

# test 用
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
  train_data_dir,
  target_size=(img_width, img_height),
  color_mode='rgb',
  classes=classes,
  class_mode='categorical',
  batch_size= train_batch_size# 1回のバッチ生成で作る画像数
  )
 
validation_generator = val_datagen.flow_from_directory(
  validation_data_dir,
  target_size=(img_width, img_height),
  color_mode='rgb',
  classes=classes,
  class_mode='categorical',
  batch_size=val_batch_size # 1回のバッチ生成で作る画像数
  )

test_generator = test_datagen.flow_from_directory(
  test_data_dir,
  target_size=(img_width, img_height),
  color_mode='rgb',
  classes=classes,
  class_mode='categorical',
  batch_size=val_batch_size # 1回のバッチ生成で作る画像数
  )

input_tensor = Input(shape=(img_width, img_height, 3))
ResNet50 = ResNet50(include_top=False, weights='imagenet',input_tensor=input_tensor)
 
top_model = Sequential()
top_model.add(Flatten(input_shape=ResNet50.output_shape[1:]))
top_model.add(Dense(nb_classes, activation='softmax'))
 
model = Model(ResNet50.input, top_model(ResNet50.output))
 
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=0.001, momentum=0.9,decay=0.0002),
              metrics=['accuracy'])

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=0.0001)

history = model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples/16,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=None,
    callbacks=[reduce_lr]
    )

# Evaluate the model on the test data using `evaluate`
print("Evaluate on test data")
results = model.evaluate_generator(test_generator,steps=None)
print("test loss, test acc:", results)

model.save('./model/ResNet50.h5')    