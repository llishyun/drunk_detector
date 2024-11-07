#!/usr/bin/env python
# coding: utf-8

# In[166]:


import os
import tqdm
import numpy as np
import csv
from moviepy.editor import AudioFileClip
from decimal import Decimal
import subprocess
from collections import Counter
from decimal import *
import sys
import keras


# # Files description
# audio_DNN_feature.ipynb- Code to extract features for Audio DNN model. For each audio file (or face video), a feature of dimension 1582 is stored in .npy file.
# audio_LSTM_feature.ipynb- Code to extract features for Audio LSTM models. Each audio input is broken into small chunks and feature of dimension 1582 is extracted for each chunk.
# audio_DNN.ipynb- Code to train Audio_DNN model.
# audio_LSTM1.ipynb- Code to train Audio LSTM models (Fixed number of the chunks)
# audio_LSTM2.ipynb- Code to train Audio LSTM models (Fixed size of a audio chunk)

# # 1.audio_DNN_feature

# In[167]:


curr=os.getcwd()
print(curr)
audio_dir = os.path.join(curr, 'audio_data')  # wav 또는 mp3 파일 경로
print(audio_dir)
audio_list = os.listdir(audio_dir)  
print(audio_list)
audio_list.pop(0)
print(audio_list)


# ### openSMILE feature Extraction
# : file 경로명에 절대 공백 있으면 안됩니다!!!

# In[168]:



for i in tqdm.tqdm(audio_list):
    input_path = os.path.join(audio_dir, i)
    output_path = os.path.join(curr, 'opensmile_feat', i[:-4] + '.csv')
    print(input_path)
    print(output_path)
    smile_extract_path = r"C:\Program Files (x86)\openSMILE\bin\SMILExtract.exe"
    config_path = r"C:\Users\liy35\opensmile\config\is09-13\IS10_paraling.conf"
    command = f'SMILExtract -C "{config_path}" -I "{input_path}" -O "{output_path}"'

    
    try:
        result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Successfully processed {i}")
    except subprocess.CalledProcessError as e:
        print(f"Error processing {i}, exit code: {e.returncode}")
        print(f"Error output: {e.stderr.decode()}")


# ## Numpy conversion

# In[169]:


curr=os.getcwd()
in_path = os.path.join(curr, 'opensmile_feat')
out_path = os.path.join(curr, 'opensmile')

print(in_path)
print(out_path)

in_list=os.listdir(in_path)
in_list.pop(0)
print(in_list)
print(len(in_list))


# In[170]:


for csv_name in tqdm.tqdm(in_list):
    out_name = csv_name[:-4] + '.npy'
    feat = None
    if not os.path.isfile(os.path.join(out_path, out_name)):  # 경로 결합 수정
        with open(os.path.join(in_path, csv_name)) as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=',')
            count = 1
            for row in csv_reader:
                if count == 1590:  # 데이터 위치 확인 필요
                    feat = row[1:]
                    feat = feat[:-1]
                count += 1

        if feat is not None:  # feat가 None이 아닌지 확인
            np_array = np.zeros((1582))
            try:
                for i in range(1582):
                    np_array[i] = float(feat[i])  # 변환 오류 처리
            except ValueError as e:
                print(f"Error converting value to float: {e}")
                continue  # 오류 발생 시 다음 파일로 진행

            np.save(os.path.join(out_path, out_name), np_array)
        
        np.save(os.path.join(out_path, out_name),np_array)
    


# In[171]:


import os

out_path = "C:\\Users\\liy35\\project3\\opensmile"  # 저장된 경로
npy_files = os.listdir(out_path)
print("Generated .npy files:", npy_files)


# In[172]:


import numpy as np

# 예제 파일 불러오기
file_path = os.path.join(out_path, "drunk_female_1.npy")  # 파일 이름 수정
if os.path.isfile(file_path):
    data = np.load(file_path)
    print("Loaded .npy data:", data)
    print("Shape of data:", data.shape)
else:
    print("File not found:", file_path)


# ## Final feature dimension for variable length audio file input

# Each audio file is broken into 76 chunks with an overlap of 50ms. Feature Dimension -[76,1582]

# In[173]:


open_dir=os.path.join(curr, 'open_chunks')#path to save features
print(open_dir)


# ##### 변수 조정 가능
# time_stamp : 오디오 클립을 나누는 개수
# overlap : 각 클립이 겹치는 시간
# error_num : 오류개수 초기화

# In[174]:




time_stamp = Decimal(76)
over_lap = Decimal(50)
error_num = 0

for i in tqdm.tqdm(audio_list):
    if not os.path.isfile(open_dir + '/' + i[:-4] + '.npy'):
        print(f"Processing file: {i}")
        
        # Getting duration of audio file
        org_path = os.path.join(audio_dir, i)
        audioclip = AudioFileClip(org_path)
        dur = Decimal(audioclip.duration * 1000.0)
        audioclip.close()
        
        # Each chunk duration
        sample_dur = Decimal(dur + over_lap * (time_stamp - 1)) / time_stamp
        step = sample_dur - over_lap
        count = 0
        ini = Decimal(0.0)
        opensmile_array = []

        while count < 76:
            start = float(ini / 1000)
            end = float((ini + sample_dur) / 1000)
            audioclip = AudioFileClip(org_path).subclip(start, end)
            input_path = os.path.join(open_dir, f'{count}.wav')
            audioclip.write_audiofile(input_path, logger=None)
            audioclip.close()
        
            # OpenSMILE features
            csv_output_path = os.path.join(open_dir, f'{count}.csv')
            config_path = r"C:\Users\liy35\opensmile\config\is09-13\IS10_paraling.conf"
            command = f'SMILExtract -C "{config_path}" -I "{input_path}" -O "{csv_output_path}"'
            print(f"Executing command: {command}")

            result = subprocess.run(command, shell=True, capture_output=True, text=True)

            if result.returncode != 0:
                print(f"Error executing command: {command}")
                print(f"Error message: {result.stderr}")

            # Check if the CSV file was created
            if os.path.isfile(csv_output_path):
                print(f"CSV file created: {csv_output_path}")
                try:
                    with open(csv_output_path, 'r') as csvfile:
                        print("Reading CSV file...")
                        cnt = 1
                        for row in csv.reader(csvfile, delimiter=','):
                            if cnt == 1590:
                                feat = row[1:]
                                feat = feat[:-1]
                            cnt += 1                            
                    np_array = np.zeros((1582))
                    for k in range(1582):
                        np_array[k] = float(feat[k])

                    if np.count_nonzero(np_array) < 1000:
                        print("Zero values problem,", np.count_nonzero(np_array), " sample_dur", sample_dur)

                    opensmile_array.append(np_array)
                except Exception as e:
                    print(f"Error reading CSV file: {e}")

            else:
                print(f"CSV file does not exist: {csv_output_path}")

            os.remove(input_path)
            os.remove(csv_output_path)
            
            ini = ini + step
            count += 1

        np_file_name = i[:-4] + '.npy'
        
        np.save(os.path.join(open_dir, np_file_name), np.array(opensmile_array))

        # Load and verify the saved npy file
        try:
            loaded_array = np.load(os.path.join(open_dir, np_file_name))
            print("Loaded array shape:", loaded_array.shape)
            print("Loaded array contents (first 5 elements):", loaded_array[:5])
        except Exception as e:
            print(f"Error loading npy file: {e}")


# In[175]:


time_stamp=Decimal(76)
over_lap=Decimal(50)
error_num=0
for i in tqdm.tqdm(audio_list):
    if(not os.path.isfile(open_dir+'/'+i[:-4]+'.npy')):
        
        #getting duration of audio file
        org_path = os.path.join(audio_dir, i)
        audioclip = AudioFileClip(org_path)
        dur=Decimal(audioclip.duration*1000.0)
        audioclip.close()
        
        #each chunk duration
        sample_dur=Decimal(dur + over_lap*(time_stamp-1) )/time_stamp
        step=sample_dur-over_lap
        count=0
        ini=Decimal(0.0)
    
        # features appended in a list ini<=dur-sample_dur or 
        opensmile_array=[]
    
        while(count<76):
            start=float(ini/1000)
            end=float((ini+sample_dur)/1000)
            audioclip=AudioFileClip(org_path).subclip(start,end)
            input_path = os.path.join(open_dir, f'{count}.wav')
            audioclip.write_audiofile(input_path,logger =None)
            audioclip.close()
        
            #opensmile features
            csv_output_path = os.path.join(open_dir, f'{count}.csv')
            config_path = r"C:\Users\liy35\opensmile\config\is09-13\IS10_paraling.conf"
            command = f'SMILExtract -C "{config_path}" -I "{input_path}" -O "{csv_output_path}"'
            os.system(command)
            
        #creating npy file from csv file 
            print(f"CSV output path: {csv_output_path}")
            with open(csv_output_path) as csvfile:
                csv_reader=csv.reader(csvfile,delimiter=',')
                cnt=1
                for row in csv_reader:
                    if cnt==1590:
                        feat=row[1:]
                        feat=feat[:-1]
                    cnt+=1
                np_array=np.zeros((1582))
                for k in range(1582):
                    np_array[k]=float(feat[k])
        
                if np.count_nonzero(np_array)<1000:
                    print("zeros values problem,",np.count_nonzero(np_array)," sample_dur",sample_dur)
                opensmile_array.append(np_array)
        
            os.remove(input_path)
            os.remove(csv_output_path)
            
            ini=ini+step
            count+=1
        np_file_name=i[:-4]+'.npy'
        if len(opensmile_array)!=time_stamp:
            print("time dimension ",len(opensmile_array))
            print("filename ",i[:-4])
        np.save(os.path.join(open_dir,np_file_name),np.array(opensmile_array))


# ## audio_DNN.ipynb

# In[176]:


LENGTH=10 #in secs
FEATURE_DIM = 1582 # opensmile feature dimension for an input chunk


# In[177]:


# USER INPUT- features_dir
# features_dir- path to save or load vgg face features
curr=os.getcwd()
repo_path=curr.split('/code')[0]
data_path = os.path.join(repo_path, 'audio_data')
features_path = repo_path  # Folder containing all features audio and video
features_dir = os.path.join(features_path, 'opensmile')# Path to save/load opensmile features

print(curr)
print(repo_path)
print(data_path)
print(features_dir)


# ### generator data split code

# In[178]:


import csv
'''
Input-  csv_file
Output- partition train, val test. Each partition consists of list of .npy files and dictionary of labels.
'''
def train_test_split(csv_path):
    label={'Drunk':1, 'Sober':0}
    partition={}
    train={}
    val={}
    test={}
    
    train_list=[]
    val_list=[]
    test_list=[]
    train_label={}
    val_label={}
    test_label={}
    
    with open(csv_path) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            row[0] = row[0].strip()  # 공백 제거
            print(f"Row being processed: {row}")  # 디버깅용 출력
            filename = row[2]
            filename = filename[:-4]
            if row[0] == 'train':
                train_label[filename] = label[row[1]]
                train_list.append(filename)
            elif row[0] == 'val':
                val_label[filename] = label[row[1]]
                val_list.append(filename)
            elif row[0] == 'test':
                test_label[filename] = label[row[1]]
                test_list.append(filename)
            else:
                print("Error in label:", row[0])  # 디버깅용 출력
                return None


    train['list']=train_list
    val['list']=val_list
    test['list']=test_list
    
    train['label']=train_label
    val['label']=val_label
    test['label']=test_label
    
    partition['train']=train
    partition['val']=val
    partition['test']=test
    
    return partition

def count_classes(d):
    values=list(d.values())
    zeros=values.count(0)
    return (zeros,len(values)-zeros)

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, datapath, batch_size=32, dim=(1582,), n_classes=2, shuffle=True):
        'Initialization'        
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        self.path = datapath

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)
            
    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)
        import gc
        gc.collect()
        return X, y
    
    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            
            file_path = os.path.join(self.path, f"{ID}.npy")
            X[i,] = np.load(file_path)
            # Store class|||
            y[i] = self.labels[ID]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)


# ### Build Model

# In[179]:


from keras.optimizers import Adam
from keras.models import Model
from keras.models import load_model
from keras.layers import Dense, Input, Dropout, LSTM, Activation,BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
import keras
from time import time
import gc
import numpy as np
keras.backend.clear_session()


# In[180]:


def create_model(num_class,dense1_units,dense2_units,dropout,input_shape=(1582,)):
    """
        Two layer DNN
    """
    X=Input(shape=input_shape)
    norm=BatchNormalization()(X)
    layer1=Dense(dense1_units, activation='relu')(norm)
    drop=Dropout(rate=dropout)(layer1)
    layer2=Dense(dense2_units, activation='relu')(drop)
    drop=Dropout(rate=dropout)(layer2)
    prob=Dense(num_class, activation='sigmoid')(layer2)
    return Model(inputs = X, outputs = prob)

def basic_model(num_class,input_shape=(1582,)):
    """
        Single layer DNN
    """
    X=Input(shape=input_shape)
    prob=Dense(num_class, activation='sigmoid')(X)
    return Model(inputs = X, outputs = prob)


# In[181]:


dense1=256
dense2=128
#dense3=128
dropout=.2
hp=2
class_num=2
model=create_model(class_num,dense1,dense2,dropout,input_shape=(FEATURE_DIM,))
#USER INPUT, path to save/ load model
model_path=repo_path+'/saved_models/audio_open/'+str(LENGTH)+'/hp'+str(hp)
model.summary()


# In[212]:


def load_keras_model(path):
    if os.path.isfile(path):
        return load_model(path)
#Loading data filenames split
#USER INPUT

print(repo_path)
partition=train_test_split(repo_path+'\\split.csv')# or enter path to the split.csv in the parent directory
print(partition['train'])
print("Number of training examples ")
print(len(partition['train']['list']))
print("Number of validation examples ")
print(len(partition['val']['list']))

params = {'datapath':features_dir ,
          'dim': (FEATURE_DIM,),
          'batch_size': 12,#!!!!데이터 많으면 64로 업그레이드
          'n_classes': 2,
          'shuffle': True}
    
#weights for imbalance classes
count=count_classes(partition['train']['label'])
print("Class instances in training class.\n Sober:",count[0]," Drunk:",count[1])
weight_0=float(count[0]+count[1])/float(count[0])
weight_1=float(count[0]+count[1])/float(count[1])
class_weight={0:weight_0, 1:weight_1}

#instances in val set
count=count_classes(partition['val']['label'])
print("Class instances in val class.\n Sober:",count[0]," Drunk:",count[1])

#instances in test set
count=count_classes(partition['test']['label'])
print("Class instances in test class.\n Sober:",count[0]," Drunk:",count[1])


# In[213]:




model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=["accuracy"])

#saving best model
checkpoint = ModelCheckpoint(model_path+'/model-{epoch:03d}-{val_accuracy:4f}.h5', verbose=1, monitor='val_accuracy',save_best_only=False, mode='max',period=5,save_freq='epoch')

#tensorboard
tensorboard = TensorBoard(log_dir=model_path+"/log/{}".format(time()))

print(partition['train']['label'])

train_generator=DataGenerator(partition['train']['list'],partition['train']['label'], **params)
val_generator=DataGenerator(partition['val']['list'],partition['val']['label'], **params)
print("generator created")

model.fit(x=train_generator,
          epochs=10,
          validation_data=val_generator,
          use_multiprocessing=False,
          workers=6,
          callbacks=[checkpoint, tensorboard],
          class_weight=class_weight)

## !!! 추후 epoch도 늘리기


# ## 위의 모델은 DNN 모델임

# In[220]:


import numpy as np

# Test 데이터 생성
test_generator = DataGenerator(partition['test']['list'], partition['test']['label'], **params)

# 모델을 사용해 예측
predictions = model.predict(test_generator, use_multiprocessing=False, workers=6)

# 예측된 값 출력
#print("Predictions:", predictions)

# 테스트 레이블 가져오기
true_labels = [partition['test']['label'][ID] for ID in partition['test']['list']]

# 확률 높은 걸로
print(predictions)
classified_predictions = np.argmax(predictions, axis=1)

# 예측된 값과 실제 값 비교
for i, (pred, classified_pred, true_label) in enumerate(zip(predictions, classified_predictions, true_labels)):
    print(f"Sample {i}: Raw Predicted={pred}, Classified Predicted={classified_pred}, True={true_label}")


# In[221]:


# 길이 확인
print(f"Length of true_labels: {len(true_labels)}")
print(f"Length of classified_predictions: {len(classified_predictions)}")

# 예: 길이가 다른 경우, 수정
# 필요에 따라 true_labels 또는 classified_predictions를 수정하세요
if len(true_labels) != len(classified_predictions):
    min_length = min(len(true_labels), len(classified_predictions))
    true_labels = true_labels[:min_length]
    classified_predictions = classified_predictions[:min_length]


# In[222]:


from sklearn.metrics import accuracy_score, precision_score, f1_score, roc_auc_score, matthews_corrcoef, confusion_matrix

# 정확도 계산
accuracy = accuracy_score(true_labels, classified_predictions)

# 정밀도 계산
precision = precision_score(true_labels, classified_predictions)

# F1 점수 계산
f1 = f1_score(true_labels, classified_predictions)

# MCC 계산
mcc = matthews_corrcoef(true_labels, classified_predictions)

# 혼동 행렬 출력 (추가)
conf_matrix = confusion_matrix(true_labels, classified_predictions)

# 결과 출력
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"MCC: {mcc:.4f}")
print("Confusion Matrix:")
print(conf_matrix)


# In[ ]:





# In[ ]:





# In[ ]:




