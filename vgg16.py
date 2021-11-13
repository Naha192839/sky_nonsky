import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import models, layers
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Flatten, Dense,Dropout,GlobalAveragePooling2D
from tensorflow.keras import optimizers,regularizers
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

train_batch_size = 32
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


input_tensor = Input(shape=(img_height,img_width,3))
vgg16 = VGG16(include_top=False, weights='imagenet',input_tensor=input_tensor)
vgg16.trainable = False

x=vgg16.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(nb_classes, activation='softmax')(x)

model = Model(vgg16.input, predictions)

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=0.001, momentum=0.9,decay=0.0002),
              metrics=['accuracy'])
model.summary()

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=0.001)

checkpoint = ModelCheckpoint(
    filepath = os.path.join(
        model_dir,
        'model_{epoch:02d}.hdf5'
    ),
    save_best_only=True
)

#TO early stopping
early_stopping = EarlyStopping(monitor='val_loss',patience=5,verbose=0,mode='auto')

# steps_per_epoch: 1エポックを宣言してから次のエポックの開始前までにgeneratorから生成されるサンプル (サンプルのバッチ) の総数．
# 典型的には，データにおけるユニークなサンプル数をバッチサイズで割った値です． 

history = model.fit(
  train_generator, 
  steps_per_epoch = train_generator.n // train_batch_size,
  validation_data = validation_generator,
  validation_steps = validation_generator.n // val_batch_size,
  epochs=20,
  callbacks=[reduce_lr,early_stopping]
)
# Evaluate the model on the test data using `evaluate`
print("Evaluate on test data")
results = model.evaluate(test_generator)
print("test loss, test acc:", results)

for layer in model.layers[:15]:
   layer.trainable = False
for layer in model.layers[15:]:
   layer.trainable = True

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=0.001, momentum=0.9,decay=0.0002),
              metrics=['accuracy'])
model.summary()

history = model.fit(
  train_generator, 
  steps_per_epoch = train_generator.n // train_batch_size,
  validation_data = validation_generator,
  validation_steps = validation_generator.n // val_batch_size,
  epochs=50,
  # callbacks=[checkpoint,early_stopping]
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

model.save('./model/vgg16.h5')    