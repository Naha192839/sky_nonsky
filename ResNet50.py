import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import models, layers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Flatten, Dense
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ReduceLROnPlateau,EarlyStopping
import os , datetime
classes = ['空あり','空なし'] #分類するクラス
nb_classes = len(classes)
train_data_dir = './dataset/train'
validation_data_dir = './dataset/val'
test_data_dir = './dataset/test'
nb_train_samples = 1400
nb_validation_samples = 200 
img_width, img_height = 224, 224

train_batch_size = 32
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
  batch_size= train_batch_size,# 1回のバッチ生成で作る画像数
  )
 
validation_generator = val_datagen.flow_from_directory(
  validation_data_dir,
  target_size=(img_width, img_height),
  color_mode='rgb',
  classes=classes,
  class_mode='categorical',
  batch_size=val_batch_size,# 1回のバッチ生成で作る画像数
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
              optimizer=optimizers.SGD(lr=0.0009, momentum=0.9,decay=0.0002),
              metrics=['accuracy'])

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                              patience=5, min_lr=0.0001)
# stop = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=5, verbose=0, mode='auto')

# steps_per_epoch: 1エポックを宣言してから次のエポックの開始前までにgeneratorから生成されるサンプル (サンプルのバッチ) の総数．
# 典型的には，データにおけるユニークなサンプル数をバッチサイズで割った値です． 
# history = model.fit_generator(
#     train_generator,
#     steps_per_epoch=  1400//32,
#     epochs=3,
#     validation_data=validation_generator,
#     validation_steps= 201//16,
#     callbacks=[reduce_lr]
#     )

history = model.fit(
  train_generator, 
  steps_per_epoch = nb_train_samples // train_batch_size,
  validation_data = validation_generator,
  validation_steps = nb_validation_samples // val_batch_size,
  epochs=30,
  callbacks=[reduce_lr]
)
# Evaluate the model on the test data using `evaluate`
print("Evaluate on test data")
results = model.evaluate(test_generator)
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