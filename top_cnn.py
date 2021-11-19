import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import models, layers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Flatten, Dense,Dropout,Conv2D,MaxPooling2D,ZeroPadding2D,BatchNormalization,Activation,Cropping2D
from tensorflow.keras import optimizers,regularizers
from tensorflow.keras.callbacks import ReduceLROnPlateau,EarlyStopping, ModelCheckpoint
import os , datetime

from tensorflow.python.keras.layers.pooling import AveragePooling2D
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
  batch_size = train_batch_size,# 1回のバッチ生成で作る画像数
  )
 
validation_generator = val_datagen.flow_from_directory(
  validation_data_dir,
  target_size=(img_height, img_width),
  color_mode='rgb',
  classes=classes,
  class_mode='categorical',
  batch_size=val_batch_size,# 1回のバッチ生成で作る画像数
  )

test_generator = test_datagen.flow_from_directory(
  test_data_dir,
  target_size=(img_height, img_width),
  color_mode='rgb',
  classes=classes,
  class_mode='categorical',
  batch_size=val_batch_size # 1回のバッチ生成で作る画像数
  )

 
model = Sequential()
model.add(Cropping2D(cropping=((0,112), (0,0)),input_shape=(img_height, img_width, 3)))
model.add(ZeroPadding2D(padding=(3, 3)))
model.add(Conv2D(32, (7, 7),strides=(1, 2)))
model.add(Activation('relu'))
model.add(ZeroPadding2D(padding=(1,1)))
model.add(MaxPooling2D((3, 3),strides=(2, 2)))

model.add(Conv2D(128, (3, 3),padding='same',activation='relu'))

model.add(ZeroPadding2D(padding=(1,1)))
model.add(Conv2D(256,(3, 3),strides=(2, 2), activation='relu'))

model.add(ZeroPadding2D(padding=(1,1)))
model.add(Conv2D(512, (3, 3),strides=(2, 2), activation='relu'))

model.add(ZeroPadding2D(padding=(1,1)))
model.add(Conv2D(1024, (3, 3),strides=(2, 2), activation='relu'))

model.add(AveragePooling2D((1,1)))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(nb_classes, activation='softmax'))
model.summary()


reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=0.001)

# checkpoint = ModelCheckpoint(
#     filepath = os.path.join(
#         model_dir,
#         'model_{epoch:02d}.hdf5'
#     ),
#     save_best_only=True
# )

#TO early stopping
early_stopping = EarlyStopping(monitor='val_loss',patience=5,verbose=0,mode='auto')

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=0.001, momentum=0.9,decay=0.0002),
              metrics=['accuracy'])
# steps_per_epoch: 1エポックを宣言してから次のエポックの開始前までにgeneratorから生成されるサンプル (サンプルのバッチ) の総数．
# 典型的には，データにおけるユニークなサンプル数をバッチサイズで割った値です． 

history = model.fit(
  train_generator, 
  steps_per_epoch = nb_train_samples // train_batch_size,
  validation_data = validation_generator,
  validation_steps = nb_validation_samples // val_batch_size,
  epochs=50,
  callbacks=[early_stopping,reduce_lr]
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

model.save('./model/top_cnn_nopadding.h5')    