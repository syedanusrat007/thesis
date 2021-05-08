import cv2
from random import shuffle
import numpy as np
import os
from tqdm import tqdm

trainn_DIRc= 'E:/MIST/L-4 T-1/CSE-400 Project and Thesis/python/image matching/dataset1/trainn'
testt_DIRc= 'E:/MIST/L-4 T-1/CSE-400 Project and Thesis/python/image matching/dataset1/testt'
imgg_SIZEs= 50
LR= 1e-3


MODEL_NAMEs = 'dogsvscats-{}-{}.model'.format(LR,'6conv-basic')

def label_imgg(imgg):
    word_labell= imgg.split('.')[-3]
    if word_labell== 'ee': return [1, 0]
    elif word_labell== 'oi': return [0, 1]

def create_trainn_dataa():
    trainning_datas= []

    for imgg in tqdm(os.listdir(trainn_DIRc)):

        label= label_imgg(imgg)

        path= os.path.join(trainn_DIRc,imgg)

        imgg= cv2.imread(path,cv2.IMREAD_GRAYSCALE)

        imgg= cv2.resize(imgg,(imgg_SIZEs,imgg_SIZEs))

        trainning_datas.append([np.array(imgg),np.array(label)])

    shuffle(trainning_datas)
    np.save('trainn_dataa.npy',trainning_datas)
    return trainning_datas


def process_testt_dataa():
    testting_datas = []
    for imgg in tqdm(os.listdir(testt_DIRc)):
        path = os.path.join(testt_DIRc, imgg)
		imgg = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        imgg = cv2.resize(imgg,(imgg_SIZEs,imgg_SIZEs))
        imgg_num = imgg.split('.')[0]
        testting_datas.append([np.array(imgg), imgg_num])

    shuffle(testting_datas)
    np.save('testt_dataa.npy',testting_datas)
    return testting_datas
trainn_dataa= create_trainn_dataa()
testt_dataa = process_testt_dataa()

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
import tensorflow as tf
from tflearn.layers.estimator import regression


tf.reset_default_graph()
convnet= input_data(shape =[None, imgg_SIZEs, imgg_SIZEs, 1], name ='input')

convnet= conv_2d(convnet,32,5,activation ='relu')
convnet= max_pool_2d(convnet,5)

convnet= conv_2d(convnet,64,5,activation ='relu')
convnet= max_pool_2d(convnet,5)

convnet= conv_2d(convnet,128,5,activation ='relu')
convnet= max_pool_2d(convnet,5)

convnet= conv_2d(convnet,64,5,activation ='relu')
convnet= max_pool_2d(convnet,5)

convnet= conv_2d(convnet,32,5,activation ='relu')
convnet= max_pool_2d(convnet,5)

convnet= fully_connected(convnet,1024,activation ='relu')
convnet= dropout(convnet,0.8)

convnet= fully_connected(convnet,2,activation ='softmax')
convnet= regression(convnet,optimizer ='adam',learning_rate= LR,
      loss='categorical_crossentropy',name ='targets')

model=tflearn.DNN(convnet,tensorboard_dir ='log')

trainn= trainn_dataa[:-500]
testt= trainn_dataa[-500:]

X= np.array([i[0] for i in trainn]).reshape(-1, imgg_SIZEs, imgg_SIZEs, 1)
Y= [i[1] for i in trainn]
testt_x= np.array([i[0] for i in testt]).reshape(-1, imgg_SIZEs, imgg_SIZEs, 1)
testt_y= [i[1] for i in testt]

model.fit({'input': X}, {'targets': Y},n_epoch= 5,
    validation_set=({'input': testt_x}, {'targets': testt_y}),
    snapshot_step= 500, show_metric = True, run_id = MODEL_NAMEs)
model.save(MODEL_NAMEs)

import matplotlib.pyplot as plt

testt_dataa = np.load('testt_dataa.npy')
fig= plt.figure()

for num, data in enumerate(testt_dataa[:20]):

    imgg_num= data[1]
    imgg_data= data[0]
    y= fig.add_subplot(4, 5, num + 1)
    orig= imgg_data
    data= imgg_data.reshape(imgg_SIZEs,imgg_SIZEs, 1)

    model_out= model.predict([data])[0]
    if np.argmax(model_out) == 1: str_label ='Oi'
    else: str_label ='EE'

    y.imshow(orig, cmap ='gray')
    plt.title(str_label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)
plt.show()