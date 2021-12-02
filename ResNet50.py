import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import models, layers
from tensorflow.keras.applications import ResNet50V2,ResNet50
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Flatten, Dense,Dropout,GlobalAveragePooling2D,AveragePooling2D
from tensorflow.keras import optimizers,regularizers
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ReduceLROnPlateau,EarlyStopping
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

train_batch_size = 64
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


input_tensor = Input(shape=(img_height,img_width, 3))
resnet50 = ResNet50(include_top=False, weights='imagenet',input_tensor=input_tensor,pooling='avg')

# 重みパラメータの凍結
resnet50.trainable = False

x=resnet50.output
predictions = Dense(nb_classes, activation='softmax')(x)

model = Model(resnet50.input, predictions)

# model.compile(loss='categorical_crossentropy',
#               optimizer=optimizers.SGD(lr=0.001 ,momentum=0.9,decay=0.0002),
#               metrics=['accuracy'])
# model.summary()

# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
#                               patience=3, min_lr=0.001)

# #TO early stopping
# early_stopping = EarlyStopping(monitor='val_loss',patience=5,verbose=0,mode='auto')

# # steps_per_epoch: 1エポックを宣言してから次のエポックの開始前までにgeneratorから生成されるサンプル (サンプルのバッチ) の総数．
# # 典型的には，データにおけるユニークなサンプル数をバッチサイズで割った値です． 

# history = model.fit(
#   train_generator, 
#   steps_per_epoch = train_generator.n // train_batch_size,
#   validation_data = validation_generator,
#   validation_steps = validation_generator.n // val_batch_size,
#   epochs=30,
#   # callbacks=[reduce_lr,early_stopping]
# )
# # Evaluate the model on the test data using `evaluate`
# print("Evaluate on test data")
# results = model.evaluate_generator(
#   test_generator,
#   steps=test_generator.n // val_batch_size)
# print("test loss, test acc:", results)


# block5の重みパラメーターを解凍
for layer in model.layers[:154]:
   layer.trainable = False
for layer in model.layers[154:]:
   layer.trainable = True

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=0.001, momentum=0.9,decay=0.0002),
              metrics=['accuracy'])
model.summary()
plot_model(model, show_shapes=True,show_layer_names=False,to_file='model.png')

history = model.fit(
  train_generator, 
  steps_per_epoch = train_generator.n // train_batch_size,
  validation_data = validation_generator,
  validation_steps = validation_generator.n // val_batch_size,
  epochs=30,
  # callbacks=[reduce_lr,early_stopping]
)

# Evaluate the model on the test data using `evaluate`
print("Evaluate on test data")
results = model.evaluate(
  test_generator,
  steps=test_generator.n // val_batch_size)
print("test loss, test acc:", results)

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.savefig(os.path.join("./fig/acc_fig/",str(datetime.datetime.today())+"acc.jpg"))
plt.clf()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.savefig(os.path.join("./fig/loss_fig/",str(datetime.datetime.today())+"loss.jpg"))

model.save('./model/ResNet50.h5')    