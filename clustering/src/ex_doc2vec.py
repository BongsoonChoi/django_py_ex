# -*- coding: utf-8 -*-

import json
from konlpy.tag import Komoran
from time import time
import pickle
import os
import pandas as pd
import urllib.request
import matplotlib.pyplot as plt
import re
from konlpy.tag import Okt
# from tf.keras.preprocessing import
from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

# # %%  한번 다운받았으면 됫다.
# import urllib
#
# urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt",
#                            filename="ratings_train.txt")
# urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt",
#                            filename="ratings_test.txt")
# %%
train_data = pd.read_table('ratings_train.txt')
test_data = pd.read_table('ratings_test.txt')

# %% 1. 중복제거

print(train_data['document'].nunique(), train_data['label'].nunique())
# 중복 제거후 사이즈
train_data.drop_duplicates(subset=['document'], inplace=True)
# document 열에서 중복인 내용이 있다면 중복 제거

train_data.groupby('label').size().reset_index(name='count')
# 중복값 제거후 인덱스 정리

print("중복 제거후 사이즈 : ", len(train_data))

# 시각화
train_data['label'].value_counts().plot(kind='bar')
print(train_data['label'].value_counts())

plt.grid()
plt.plot(train_data['label'].value_counts())
plt.show()

# %% 널값 제거
train_data.isnull().values.any()
# True가 나온다면 null 값이 있다는 의미.

train_data.isnull().sum()  # 어디에 있는지 확인하는것.

train_data.loc[train_data.document.isnull()]
# 위치확인, train_data 중에 document에서 True인 값을 찾는거
train_data = train_data.dropna(how='any')  # Null 값이 존재하는 행 제거
print(train_data.isnull().values.any())  # Null 값이 존재하는지 확인

# %% 한글만 남겨두고 제거

train_data['document'] = train_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "")
# 한글 제외 모두 제거
train_data['document'].replace('', np.nan, inplace=True)  # 공백을 널로 변경.
print(train_data.isnull().sum())
train_data = train_data.dropna(how='any')  # 다시 제거
print(len(train_data))

# %% 테스트 데이터도 동일한 작업하기

test_data.drop_duplicates(subset=['document'], inplace=True)  # document 열에서 중복인 내용이 있다면 중복 제거
test_data['document'] = test_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "")  # 정규 표현식 수행
test_data['document'].replace('', np.nan, inplace=True)  # 공백은 Null 값으로 변경
test_data = test_data.dropna(how='any')  # Null 값 제거
print('전처리 후 테스트용 샘플의 개수 :', len(test_data))

# %% 토큰화
okt = Okt()  # konply 라이브러리

stopwords = ['의', '가', '이', '은', '는', '을', '를', '잘', '과', '와', '도', '으로', '에', '에서',
             '한', '하다', '로']
X_train = []
for sentence in train_data['document']:
    temp_X = []
    temp_X = okt.morphs(sentence, stem=True)  # 토큰화
    temp_X = [word for word in temp_X if not word in stopwords]  # 불용어 제거
    X_train.append(temp_X)
    if (len(X_train) % 1000 == 0):
        print("train {} 개 성공".format(len(X_train)))

X_test = []
for sentence in test_data['document']:
    temp_X = []
    temp_X = okt.morphs(sentence, stem=True)  # 토큰화
    temp_X = [word for word in temp_X if not word in stopwords]  # 불용어 제거
    X_test.append(temp_X)
    if (len(X_test) % 1000 == 0):
        print("test {} 개 성공".format(len(X_test)))

# %% 정수인코딩 필요
# 각 단어에 정수를 부여, 빈도수가 높은순서대로 부여됨. 빈도수 낮은 한글은 제외하기.
tokenizer = keras.preprocessing.text.Tokenizer();
tokenizer.fit_on_texts(X_train)

threshold = 3
total_cnt = len(tokenizer.word_index)  # 단어의 수
rare_cnt = 0  # 등장 빈도수가 threshold보다 작은 단어의 개수를 카운트
total_freq = 0  # 훈련 데이터의 전체 단어 빈도수 총 합
rare_freq = 0  # 등장 빈도수가 threshold보다 작은 단어의 등장 빈도수의 총 합

# 단어와 빈도수의 쌍(pair)을 key와 value로 받는다.
for key, value in tokenizer.word_counts.items():
    total_freq = total_freq + value

    # 단어의 등장 빈도수가 threshold보다 작으면
    if (value < threshold):
        rare_cnt = rare_cnt + 1
        rare_freq = rare_freq + value

print('단어 집합(vocabulary)의 크기 :', total_cnt)
print('등장 빈도가 %s번 이하인 희귀 단어의 수: %s' % (threshold - 1, rare_cnt))
print("단어 집합에서 희귀 단어의 비율:", (rare_cnt / total_cnt) * 100)
print("전체 등장 빈도에서 희귀 단어 등장 빈도 비율:", (rare_freq / total_freq) * 100)

vocab_size = total_cnt - rare_cnt + 1  # 전체 단어 개수 중 빈도수 2이하인 단어 개수는 제거. 0번 패딩 토큰을 고려하여 +1
print('단어 집합의 크기 :', vocab_size)  # 빈도 순서대로 되어있으니 사이즈로 짜르면 됨.

tokenizer = keras.preprocessing.text.Tokenizer(vocab_size);
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)
# %%
Y_train = np.array(train_data['label'])
Y_test = np.array(test_data['label'])

# %% 빈샘플 제거 : 빈도수 낮은 단어 제거과정에서 생긴 빈샘플 제거하기.

drop_train = [index for index, sentence in enumerate(X_train) if len(sentence) < 1]
drop_test = [index for index, sentence in enumerate(X_test) if len(sentence) < 1]

X_train = np.delete(X_train, drop_train, axis=0)
Y_train = np.delete(Y_train, drop_train, axis=0)

X_test = np.delete(X_test, drop_test, axis=0)
Y_test = np.delete(Y_test, drop_test, axis=0)

# %% 패딩 - 글자수 맞추기
print('리뷰의 최대길이 : ', max(len(l) for l in X_train))
print('리뷰의 평균 길이', sum(map(len, X_train)) / len(X_train))
plt.hist([len(s) for s in X_train], bins=50)
plt.xlabel('length of samples')
plt.ylabel('number of samples')
plt.show()


def below_threshold_len(max_len, nested_list):
    cnt = 0
    for s in nested_list:
        if (len(s) <= max_len):
            cnt = cnt + 1
    print('전체 샘플중 %s 이하인 샘플의 비율 : %s' % (max_len, (cnt / len(nested_list)) * 100))


max_len = 25 # 91%가 25글자 미만임.
below_threshold_len(max_len, X_train)
# %% 최대 길이 설정
X_train = pad_sequences(X_train, maxlen=max_len)
X_test = pad_sequences(X_test, maxlen=max_len)

###############################  여기까지 전처리.

# %%
from tensorflow import keras

model = keras.models.Sequential()
model.add(keras.layers.Embedding(vocab_size, 100))
model.add(keras.layers.LSTM(128))
model.add(keras.layers.Dense(1, activation='sigmoid'))

es = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)  # 4회 이상 손실 증가시 스탑
mc = keras.callbacks.ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)
# 검증데이터가 좋아질 경우에만 모델에 저장함.

## 텐서보드를 활용하기
import datetime
log_dir = "logs\\naver_review_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

#%%
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(X_train, Y_train, epochs=15,
                    callbacks=[es, mc, tensorboard_callback],
                    batch_size=60, validation_split=0.2)


