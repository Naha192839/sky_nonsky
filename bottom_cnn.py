import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import models, layers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Flatten, Dense,Dropout,Conv2D,MaxPooling2D,ZeroPadding2D,BatchNormalization,Activation,Cropping2D,GlobalAveragePooling2D
from tensorflow.keras import optimizers,regularizers
from tensorflow.keras.callbacks import ReduceLROnPlateau,EarlyStopping, ModelCheckpoint
import os , datetime,shutil,glob,numpy as np

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
rotation_range=15,
width_shift_range=0.5,
zoom_range=0.5,
horizontal_flip=True,
)

# validation
val_datagen = ImageDataGenerator(rescale=1. / 255)

# test 用
test_datagen = ImageDataGenerator(rescale=1. / 255)

num = 0
val_num = 0 
val_count = 0
ep = 50

all_loss=[]
all_val_loss=[]
all_test_loss=[]

all_acc=[]
all_val_acc=[]
all_test_acc=[]
#交差検証
for i in range(8):
  print(val_count)
  for index, classlabel in enumerate(classes):
    val_dir = "./dataset/val/" + classlabel
    train_dir = "./dataset/train/" + classlabel
    for p in os.listdir(val_dir):
      shutil.move(os.path.join(val_dir, p), train_dir)#検証データを訓練データに移動
    files = glob.glob(train_dir + "/*.jpg")#全データを取得
    val_list = files[val_count:val_count+100]#検証データの確保
    for i in val_list:
      shutil.move(i,val_dir)#検証ディレクトに移動
    
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

  model = Sequential()
  model.add(Cropping2D(cropping=((112,0), (0,0)),input_shape=(img_height, img_width, 3)))
  model.add(ZeroPadding2D(padding=(3, 3)))
  model.add(Conv2D(32, (7, 7),strides=(1, 2)))
  model.add(Activation('relu'))
  model.add(ZeroPadding2D(padding=(1,1)))
  model.add(MaxPooling2D((3, 3),strides=(2, 2)))

  model.add(ZeroPadding2D(padding=(1,1)))
  model.add(Conv2D(128, (3, 3),strides=(2, 2),activation='relu'))

  model.add(ZeroPadding2D(padding=(1,1)))
  model.add(Conv2D(256,(3, 3),strides=(2, 2), activation='relu'))

  model.add(ZeroPadding2D(padding=(1,1)))
  model.add(Conv2D(512, (3, 3),strides=(2, 2), activation='relu'))


  model.add(Conv2D(1024,(3, 3),padding='same',activation='relu'))

  model.add(GlobalAveragePooling2D())
  model.add(Dense(nb_classes, activation='softmax'))

  model.compile(loss='categorical_crossentropy',
                optimizer=optimizers.SGD(lr=0.001, momentum=0.9,decay=0.0002),
                metrics=['accuracy'])
                
  history = model.fit(
    train_generator, 
    steps_per_epoch = train_generator.n // train_batch_size,
    validation_data = validation_generator,
    validation_steps = validation_generator.n // val_batch_size,
    epochs=ep,
    # callbacks=[reduce_lr]
  )

  # Evaluate the model on the test data using `evaluate`
  print("Evaluate on test data")
  results = model.evaluate(
    test_generator,
    steps=test_generator.n // val_batch_size)
  print("test loss, test acc:", results)
  
  loss=history.history['loss']
  val_loss=history.history['val_loss']
  

  acc=history.history['accuracy']
  val_acc=history.history['val_accuracy']

  all_loss.append(loss)
  all_val_loss.append(val_loss)
  all_test_loss.append(results[0])

  all_acc.append(acc)
  all_val_acc.append(val_acc)
  all_test_acc.append(results[1])



  # Plot training & validation accuracy values
  plt.plot(acc)
  plt.plot(val_acc)
  plt.title('Model accuracy')
  plt.ylabel('Accuracy')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Val'], loc='upper left')
  plt.savefig(os.path.join("./fig/acc_fig/",str(datetime.datetime.today())+"acc.jpg"))
  plt.clf()

  # Plot training & validation loss values
  plt.plot(loss)
  plt.plot(val_loss)
  plt.title('Model loss')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Val'], loc='upper left')
  plt.savefig(os.path.join("./fig/loss_fig/",str(datetime.datetime.today())+"loss.jpg"))
  plt.clf()

  model.save('./model/bottom_cnn_'+str(num)+'_aug.h5')    
  num += 1
  val_count = val_count + 100

ave_all_loss=[
    np.mean([x[i] for x in all_loss]) for i in range(ep)]
ave_all_val_loss=[
    np.mean([x[i] for x in all_val_loss]) for i in range(ep)]

ave_all_acc=[
    np.mean([x[i] for x in all_acc]) for i in range(ep)]
ave_all_val_acc=[
    np.mean([x[i] for x in all_val_acc]) for i in range(ep)]

ave_all_test_loss=[
    np.mean(all_test_loss)]
ave_all_test_acc=[
    np.mean( all_test_acc)]

print("ave_all_loss"+str(ave_all_loss))
print("ave_all_acc"+str(ave_all_acc))
print("ave_all_val_loss"+str(ave_all_val_loss))
print("ave_all_val_acc"+str(ave_all_val_acc))
print(all_test_loss)
print(all_test_acc)
print("ave_all_test_loss"+str(ave_all_test_loss))
print("ave_all_test_acc"+str(ave_all_test_acc))
