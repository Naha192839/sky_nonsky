from tensorflow import keras
import sys, os, glob
import numpy as np
import PIL.Image
import numpy as np
import tensorflow
import cv2
import datetime

from tensorflow.keras.preprocessing.image import ImageDataGenerator
# 画像用
from keras.preprocessing.image import array_to_img, img_to_array, load_img, save_img
# モデル読み込み用
from keras.models import load_model
# Grad−CAM計算用
from tensorflow.keras import models

test_data_dir = './dataset/test2'
testpic = "/home/student/e18/e185701/sky_nonsky_ver2/sky_nonsky/b1e9ee0e-67e26f2e.jpg" 
classes = ["空あり", "空なし"]
image_size = 224
num_classes = len(classes)
model_name = "ResNet50"
keras_param = "/home/student/e18/e185701/sky_nonsky_ver2/sky_nonsky/model/"+str(model_name)+".h5"

types = ['jpg']
files = []

img_width, img_height = 224, 224

def load_image(path,model):
    img = PIL.Image.open(path)
    img = img.convert('RGB')
    h, w= img.size
    if model == "top_cnn":
        img = img.crop((0, 0, h, w/2))
        print(img)
    elif model == "bottom_cnn":
         img = img.crop((0, w/2, h, w))
         print(img)    
    img = img.resize((224,224))
    return img
    
def grad_cam(input_model, x, tri_x,layer_name):
    """
    Args: 
        input_model(object): モデルオブジェクト
        x(ndarray): 画像
        layer_name(string): 畳み込み層の名前
    Returns:
        output_image(ndarray): 元の画像に色付けした画像
    """

    # 画像の前処理
    # 読み込む画像が1枚なため、次元を増やしておかないとmodel.predictが出来ない
    X = np.expand_dims(x, axis=0)
    preprocessed_input = X.astype('float32') / 255.0    

    grad_model = models.Model([input_model.inputs], [input_model.get_layer(layer_name).output, input_model.output])

    with tensorflow.GradientTape() as tape:
        conv_outputs, predictions = grad_model(preprocessed_input)
        class_idx = np.argmax(predictions[0])
        loss = predictions[:, class_idx]

    # 勾配を計算
    output = conv_outputs[0]
    grads = tape.gradient(loss, conv_outputs)[0]

    gate_f = tensorflow.cast(output > 0, 'float32')
    gate_r = tensorflow.cast(grads > 0, 'float32')

    guided_grads = gate_f * gate_r * grads

    # 重みを平均化して、レイヤーの出力に乗じる
    weights = np.mean(guided_grads, axis=(0, 1))
    cam = np.dot(output, weights)

    # 画像を元画像と同じ大きさにスケーリング
    cam = cv2.resize(cam, (224,224), cv2.INTER_LINEAR)
    # ReLUの代わり
    cam  = np.maximum(cam, 0)
    # ヒートマップを計算
    heatmap = cam / cam.max()

    # モノクロ画像に疑似的に色をつける
    jet_cam = cv2.applyColorMap(np.uint8(255.0*heatmap), cv2.COLORMAP_JET)
    # RGBに変換
    rgb_cam = cv2.cvtColor(jet_cam, cv2.COLOR_BGR2RGB)
    
    # for i in range(112,224):
    #     x[i] = 0
    
    # print("----------------------------------")
    # print(x.shape)
    # もとの画像に合成
    output_image = (np.float32(rgb_cam) + tri_x / 2)  

    return output_image

model = keras.models.load_model(keras_param)
if model_name == "ResNet50":
    target_layer = "conv5_block3_3_conv"
else :
    target_layer = 'conv2d_4'

#-----------------------------------------------------
# img = img_to_array(load_img(testpic, target_size=(image_size,image_size)))
# # トリンミグ画像の読み込み
# img_trimmig = img_to_array(load_image(testpic,model_name))

# prd = model.predict(np.array([img / 255.0]))
# print(prd) # 精度の表示
# prelabel = np.argmax(prd, axis=1)

# cam = grad_cam(model, img, img_trimmig, target_layer)
# save_img(os.path.join("./fig/gradcam_fig/"+str(model_name)+"/gradcam_"+str(datetime.datetime.today())+".jpg"),cam)



# if prelabel == 0:
#     print(">>> 空あり")
# elif prelabel == 1:
#     print(">>> 空なし")
# ----------------------------------------------------------------------------------

# 空あり画像のチェック
photos_dir = "/home/student/e18/e185701/sky_nonsky_ver2/sky_nonsky/dataset/test2/空あり" 
for ext in types:
  file_path = os.path.join(photos_dir, '*.{}'.format(ext))
  files.extend(glob.glob(file_path))
for i in files:
    img = img_to_array(load_img(i, target_size=(image_size,image_size)))
    # トリンミグ画像の読み込み
    img_trimmig = img_to_array(load_image(i,model_name))

    prd = model.predict(np.array([img / 255.0]))
    print(prd) # 精度の表示
    prelabel = np.argmax(prd, axis=1)
    

    if prelabel == 0:
        save_img(os.path.join("./fig/gradcam_fig/"+str(model_name)+"/空あり/gradcam_"+str(datetime.datetime.today())+"-"+os.path.basename(i)),img)
        cam = grad_cam(model, img,img_trimmig,target_layer)
        save_img(os.path.join("./fig/gradcam_fig/"+str(model_name)+"/空あり/gradcam_"+str(datetime.datetime.today())+".jpg"),cam)
        
    if prelabel == 1:
        save_img(os.path.join("./fig/gradcam_fig/"+str(model_name)+"/空あり→空なし/gradcam_"+str(datetime.datetime.today())+"-"+os.path.basename(i)),img)
        cam = grad_cam(model, img, img_trimmig,target_layer)
        save_img(os.path.join("./fig/gradcam_fig/"+str(model_name)+"/空あり→空なし/gradcam_"+str(datetime.datetime.today())+".jpg"),cam)
files = []

# # 空なし画像のチェック
# photos_dir = "/home/student/e18/e185701/sky_nonsky_ver2/sky_nonsky/dataset/test2/空なし" 
# for ext in types:
#   file_path = os.path.join(photos_dir, '*.{}'.format(ext))
#   files.extend(glob.glob(file_path))
# for i in files:
#     img = img_to_array(load_img(i, target_size=(image_size,image_size)))
#     # トリンミグ画像の読み込み
#     img_trimmig = img_to_array(load_image(i,model_name))

#     prd = model.predict(np.array([img / 255.0]))
#     print(prd) # 精度の表示
#     prelabel = np.argmax(prd, axis=1)  

#     if prelabel == 0:
#         save_img(os.path.join("./fig/gradcam_fig/"+str(model_name)+"/空なし→空あり/gradcam_"+str(datetime.datetime.today())+"-"+os.path.basename(i)),img)        
#         cam = grad_cam(model, img,img_trimmig, target_layer)
#         save_img(os.path.join("./fig/gradcam_fig/"+str(model_name)+"/空なし→空あり/gradcam_"+str(datetime.datetime.today())+".jpg"),cam)
        
#     if prelabel == 1:
#         save_img(os.path.join("./fig/gradcam_fig/"+str(model_name)+"/空なし/gradcam_"+str(datetime.datetime.today())+"-"+os.path.basename(i)),img)       
#         cam = grad_cam(model, img,img_trimmig, target_layer)
#         save_img(os.path.join("./fig/gradcam_fig/"+str(model_name)+"/空なし/gradcam_"+str(datetime.datetime.today())+".jpg"),cam)