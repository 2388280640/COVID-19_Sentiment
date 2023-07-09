import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import os
import matplotlib.pyplot as plt
from transformers import BertTokenizer,TFBertModel
import pandas as pd
from keras.utils import np_utils
from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras.callbacks import *
from keras.regularizers import *
tokenizer=BertTokenizer.from_pretrained('./bert-base-chinese')
BertModel=TFBertModel.from_pretrained('./bert-base-chinese')

'''
predictDataPath = ["WeiBoComment_2019_12.txt",
                   "WeiBoComment_2020_3.txt","WeiBoComment_2020_6.txt","WeiBoComment_2020_9.txt","WeiBoComment_2020_12.txt",
                   "WeiBoComment_2021_3.txt","WeiBoComment_2021_6.txt","WeiBoComment_2021_9.txt","WeiBoComment_2021_12.txt",
                   "WeiBoComment_2022_3.txt","WeiBoComment_2022_6.txt","WeiBoComment_2022_9.txt","WeiBoComment_2022_12.txt",]
'''
predictDataPath = ["WeiBoComment_2020_9.txt","WeiBoComment_2022_6.txt","WeiBoComment_2022_9.txt"]
def CreateBiLSTMModel():
    x_input = Input(shape=(75,768))

    #x = Dense(units=512,activation='relu')(x_input)
    #x = BatchNormalization()(x)
    #x = Dropout(0.3)(x) 

    #x = Flatten()(x_input)
    #x = LSTM(units=512)(x_input)
    x = Bidirectional(LSTM(units=128,return_sequences=True,kernel_regularizer=l2(0.01)),merge_mode='concat',input_shape=(75,768))(x_input)
    x = MultiHeadAttention(key_dim=256,num_heads=2)(x,x,x)
    x = Dropout(0.4)(x)
    #x = LSTM(units=1024)(x_input)
    #x = Flatten()(x)
    #x = Flatten()(x)
    x = Dense(units=128,activation='relu',kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    #x = Dense(units=256,activation='relu')(x)
    #x = BatchNormalization()(x)
    #x = Dropout(0.3)(x)
    x = Dense(units=32,activation='relu',kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
   
    x = Flatten()(x)

    #x = Dense(units=64,activation='relu',kernel_regularizer=l2(0.01))(x)
    #x = BatchNormalization()(x)
    #x = Dropout(0.5)(x)

    #x = Flatten()(x)
    x = Dense(units=2,activation='softmax')(x)

    #x = BatchNormalization()(x)
    #x = Dropout(0.3)(x)
    model = Model(inputs=x_input,outputs=x)
    sgd = gradient_descent_v2.SGD(learning_rate=0.001)
    #adam = adam_v2.Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])

    return model

def save_train_history(history,train,validation,name):
    plt.plot(history.history[train],marker='s',markersize=1,markeredgecolor='yellow', markerfacecolor=(105/255,189/255,67/255),color=(105/255,189/255,67/255))
    plt.plot(history.history[validation],marker='*',markersize=1,markeredgecolor='red', markerfacecolor=(108/255,64/255,152/255),color=(108/255,64/255,152/255))
    plt.title('History')
    plt.xlabel('Epoch')
    plt.ylabel(train)
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(name)
    plt.close()

def readPredictionData(path):
    data=[]
    temp=[]
    count=0
    batchsize=5000
    with open(path,encoding='utf-8',mode='r') as file:
        lines = file.readlines()
        for line in lines:
            dataString = line.split("\t")
            temp=temp+[dataString[1].replace("\n","")]
            count=count+1
            if count % batchsize ==0:
                output=ConvertToNumpy(temp)
                data = data+[output]
                temp = []
    if len(temp) != 0:
        output = ConvertToNumpy(temp)
        data = data + [output]
    data = np.concatenate((data),axis=0)
    return data

def TrainBiLSTMModel(model,train_data,train_label,test_data,test_label):
    earlyStop = EarlyStopping(monitor='val_accuracy',min_delta=0,patience=25,mode='max',verbose=1,restore_best_weights=True)
    reduceLR=ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=10, verbose=0, model='auto', epsilon=0.0001,
                      cooldown=0, min_lr=0.00001)
    history =model.fit(
        x=train_data,
        y=train_label,
        batch_size=128,
        callbacks=[earlyStop,reduceLR],
        epochs=500,
        validation_split=0.3,
        verbose=1
    )
    model.save_weights('BiLSTMTest.h5')
    save_train_history(history,'accuracy','val_accuracy','history_accyracy.png')
    save_train_history(history,'loss','val_loss','history_loss.png')
    score=model.evaluate(x=test_data,y=test_label,batch_size=128)
    print("accuracy: ",score[1])

def ConvertToNumpy(data):
    token = tokenizer(data, return_tensors='tf', max_length=75, padding='max_length', truncation=True,add_special_tokens=True)
    modelOutput = BertModel(token)
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        cls=sess.run(modelOutput['last_hidden_state'])
        return cls

def readFileAndPreprocess(path):
    label=[]
    data=[]
    df = pd.read_csv(path)
    for index, row in df.iterrows():
        if index % 2 ==0:
            label = label +[row[0]]
            data = data + [row[1]]
    
    np.random.seed(999)
    np.random.shuffle(data)
    np.random.seed(999)
    np.random.shuffle(label)
    data = data[:20000]
    label=label[:20000]

    temp=[]
    batchSize=5000
    dataPreprocessed=[]
    count=0
    output=None
    for item in data:
        temp=temp+[item]
        count=count+1
        if count % batchSize == 0:
            output = ConvertToNumpy(temp)
            dataPreprocessed = dataPreprocessed+[output]
            temp=[]
    if len(temp)!=0:
        output = ConvertToNumpy(temp)
        dataPreprocessed = dataPreprocessed + [output]
    dataPreprocessed=np.concatenate((dataPreprocessed),axis=0)
    label = np_utils.to_categorical(label)
    print(label.shape)
    return dataPreprocessed,label
    
def Predict(model,data,fileName):
    fileName=fileName.split(".")[0]
    categoryMap = {0:"负面",1:"正面"}
    category=np.array([0,0])

    prediction = model.predict(data)
    prediction=np.argmax(prediction,axis=1)
    for item in prediction:
        category[item]=category[item]+1 
    
    total = np.sum(category)
    fileInfo = "totoal comment："+str(total)+"\n"
    fileInfo = fileInfo + categoryMap[0]+" comment number："+str(category[0])+" proportion："+str(category[0]/total)+"\n"
    fileInfo = fileInfo + categoryMap[1]+" comment number："+str(category[1])+" proportion："+str(category[1]/total)+"\n"
    print(fileInfo)
    
    with open(fileName+"_Result.txt",mode='w',encoding='utf-8') as file:
        file.write(fileInfo)

def Statistic(path):
    label=[]
    data=[]
    df = pd.read_csv(path)
    wordlen = 0
    count = 0
    for index, row in df.iterrows():
        label = label +[row[0]]
        data = data + [row[1]]
        wordlen = wordlen + len(row[1])
        count = count+1
    print("total sentences: "+ str(count))
    print("total length: "+str(wordlen))
    print("average length: "+str(wordlen/count))

def ProjectRun():
    Statistic("./weibo_senti_100k.csv")
    
    
    #data,label=readFileAndPreprocess("./weibo_senti_100k.csv")
    #np.save("data.npy",data)
    #np.save("label.npy",label)

    '''
    data = np.load("data.npy")
    label = np.load("label.npy")
    dataLen = len(data)
    np.random.seed(33)
    np.random.shuffle(data)
    np.random.seed(33)
    np.random.shuffle(label)
    train_data = data[:int(dataLen*0.8)]
    test_data = data[int(dataLen*0.8):]
    train_label = label[:int(dataLen*0.8)]
    test_label = label[int(dataLen*0.8):]
    

    model=CreateBiLSTMModel()
    #model.summary()
    model.load_weights("BiLSTMTest.h5")
    #score=model.evaluate(x=test_data,y=test_label,batch_size=128)
    #print("accuracy: ",score[1])

    #score=model.evaluate(x=test_data,y=test_label,batch_size=256)
    #print("accuracy: ",score[1])
    #model.load_weights('BiLSTMTest.h5')
    #TrainBiLSTMModel(model,train_data,train_label,test_data,test_label)
    
    for item in predictDataPath:
        data=readPredictionData("./AnalysisData/"+item)
        Predict(model,data,item)
    
    #temp=['我的天 不是吧 上海解封我刚从安徽回上海 准备上海能堂食了 然后爸妈来跟男朋友爸妈谈订婚的事情这下BBQ了 元旦男朋友家那边疫情 五一上海解封刚准备等可以堂食 咱晚晚中招了']
    '''
    '''
    temp=['更博了，爆照了，帅的呀，就是越来越爱你！生快傻缺[爱你][爱你][爱你]',
            '专家都不靠谱，豆瓣上高分的是不是都看的英文版？我有种一本好书毁在翻译上的感觉[泪]尼玛，感觉自己就是在看金山词霸翻译结果',
            '男女都通用吗？//@海伦的海伦：还真就进了这范围，我幸福 //@杨骁:我没有了 //@变成koala:二十五，还有机会么？[害羞] //@精彩语录:偶想现在就结婚，可是对象在哪里~~~[泪]',
            '完了，我前天吃了黄瓜，还曾一度迷恋这个，现在还两包架子上放着呢肿么办啊！//@掉链吨吨：@Nigger-LEE //@潮流时尚经典Style：天啊！我最喜欢吃辛拉面了！[泪]',
            '目的性很强的人生往往很悲催！有些人就是这么悲催的活着，真让人失望。[汗][晕]']
    '''
    #output=ConvertToNumpy(data)

    #prediction = model.predict(data)
    #print(prediction)
    #prediction=np.argmax(prediction,axis=1)
    #neg=0
    #for item in prediction:
    #    if item == 0:
    #        neg=neg+1
    #print("total："+str(len(prediction)))
    #print("neg："+str(neg))
    #print("pos："+str(len(prediction)-neg))
    

ProjectRun()
#print(tf.config.list_physical_devices('GPU'))
