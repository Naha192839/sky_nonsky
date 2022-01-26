from numpy.lib.function_base import append
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import datetime,os
import seaborn as sns
from keras.preprocessing.image import array_to_img, img_to_array, load_img, save_img

# モデル読み込み用
from keras.models import load_model
#混合行列計算用
from sklearn.metrics import confusion_matrix,f1_score,recall_score,precision_score,classification_report

test_data_dir = './dataset/test2'
classes = ["空あり", "空なし"]
image_size = 224
num_classes = len(classes)
model_name = "ResNet50"
keras_param = "/home/student/e18/e185701/sky_nonsky_ver2/sky_nonsky/model/"+str(model_name)+"_3.h5"

img_width, img_height = 224, 224

train_batch_size = 64
val_batch_size = 400

test_datagen = ImageDataGenerator(rescale=1. / 255)
test_generator = test_datagen.flow_from_directory(
  test_data_dir,
  target_size=(img_height, img_width),
  color_mode='rgb',
  classes=classes,
  class_mode='categorical',
  batch_size=val_batch_size # 1回のバッチ生成で作る画像数
  )
model = keras.models.load_model(keras_param)

batch = test_generator.next()
x, y_test = batch
y_test=np.argmax(y_test,axis=1)

p_test = model.predict(x) # 予測
p_test = np.argmax(p_test,axis=1)


df = pd.DataFrame({'正解値':y_test, '予測値':p_test})
#誤分類を抽出
df2 = df[df['正解値']!=df['予測値']]

print(df2)
print("------------------------------------------------")


# for t in range(len(classes)):

#   print(f'■ 正解値「{t}」に対して正しく予測（分類）できなかったケース')

#   # 正解値が t の行を抽出
#   index_list = list(df2[df2['正解値']==t].index.values)
#   print(index_list)
#   # matplotlib 出力
#   n_cols = 7 #7列
#   n_rows = ((len(index_list)-1)//n_cols)+1 #indexによって行は変化する

#   fig, ax = plt.subplots(nrows=n_rows, ncols=n_cols ,figsize=(6.5, 0.9*n_rows), dpi=120)
#   for i,ax in enumerate( np.ravel(ax) ):
#     if i < len(index_list):

#       p = index_list[i]
#       print(p)
#       type(p)
#       ax.imshow(x[p],interpolation='nearest',vmin=0.,vmax=1.,cmap='Greys')

#       # 予測（分類）を左上に表示
#       t = ax.text(1, 1, f'{p_test[p]}', verticalalignment='top', fontsize=8, color='tab:red')
#       t.set_path_effects([pe.Stroke(linewidth=2, foreground='white'), pe.Normal()]) 

#       # 予測（分離）に対応する出力層のニューロンの値を括弧で表示
#       s = model.predict( np.array([x[p]]) ) # 出力層の値 
#       s = s[0]
#       t = ax.text(1, 200, f'({s[s.argmax()]:.3f})', verticalalignment='top', fontsize=6, color='tab:red')
#       t.set_path_effects([pe.Stroke(linewidth=2, foreground='white'), pe.Normal()]) 

#       # 目盛などを非表示に
#       ax.tick_params(axis='both', which='both', left=False, labelleft=False, 
#                      bottom=False, labelbottom=False)

#     else :
#       ax.axis('off') # 余白処理

#   plt.savefig(os.path.join("./fig/error_fig/"+str(model_name)+"/error_"+str(datetime.datetime.today())+".jpg"))

# plt.figure()
cm = confusion_matrix(y_test, p_test,normalize='true')
df_cm = pd.DataFrame(cm, index=['sky', 'non-sky'], columns=['sky', 'non-sky'])

# print("適合率:"+str(precision_score(y_test, p_test,average=None)))
# print("再現率"+str(recall_score(y_test, p_test,average=None)))
# print("F値"+str(f1_score(y_test, p_test,average=None)))
print(classification_report(y_test, p_test,digits=6))
# df_cm = sns.heatmap(df_cm, annot=True,cmap='Blues')
# df_cm.set_xlabel("predict label", fontsize = 10)
# df_cm.set_ylabel(" ture label", fontsize = 10)
# plt.savefig("./fig/conf_fig/"+str(model_name)+"/conf_"+str(datetime.datetime.today())+".jpg")