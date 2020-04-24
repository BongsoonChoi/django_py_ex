# -*- coding: utf-8 -*-

"""
This is a temporary script file.
"""

import numpy as np
from konlpy.tag import Okt
from konlpy.tag import Kkma
from matplotlib import font_manager

import cx_Oracle
import os
import re

from gensim.models import Word2Vec as w2v

import matplotlib.pyplot as plt
import mglearn

from sklearn.cluster import KMeans as km

# https://www.oracle.com/kr/database/technologies/instant-client/downloads.html  # 유틸 다운로드 url

LOCATION = r"C:\oracle\instantclient_11_2"  # oracle db를 쓰기위한 유틸파일
# print("ARCH:", platform.architecture())
# print("FILES AT LOCATION:")
# for name in os.listdir(LOCATION):
#     print(name)
os.environ["PATH"] = LOCATION + ";" + os.environ["PATH"]  # 환경변수 등록


# %%

def tokenize_sentense3(text):
    okt = Okt()
    return okt.phrases(text)
    # return okt.nouns(text);


def tokenize_sentense2(text):
    kkma = Kkma();
    return kkma.morphs(text);


def tokenize_sentense(text):
    okt = Okt();
    return okt.morphs(text);


# %%

def sql(start, end):
    return """
    select TITLE,CONTENT, tab_category from    
    (select a.*  , rownum rnum  from JSOUP_CONTENT a  WHERE LENGTH(CONTENT) > 0 and tab_category is not null)   
    where rnum between """ + str(start) + """ and  """ + str(end) + """  order by rnum desc   
    """


def db_select(query, address):
    os.putenv('NLS_LANG', '.UTF8')  # 한글입력을 위해
    # 연결에 필요한 기본 정보 (유저, 비밀번호, 데이터베이스 서버 주소)
    conn = cx_Oracle.connect(address)
    cur = conn.cursor()
    cur.execute(query)  # 100개를 가져오는 쿼리

    list = []  # 제목과 본문 토큰과
    list_only_subject = []  # 제목 토큰화
    list_only_contents = []  # 본문 토큰화
    list_only_label = []  # 본문 토큰화
    list_only_subject_non_token = []  # 토큰화 없는 제목
    list_only_contents_non_token = []  # 토큰화 없는 본문
    for name in cur:
        # list_news=[] # 제목과 본문을 담기위한 리스트
        # print("테스트 이름 리스트 : ", tokenize_sentense(name[0]))
        word_subject = str(name[0])
        word_subject = re.sub(r"[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》“”’·.a-zA-Z0-9]+", " ", word_subject)
        # 특수문자 영어 숫자 제거.
        # list_news.append(tokenize_sentense(word_subject))
        list_only_subject_non_token.append(word_subject)
        list_only_subject.append(tokenize_sentense3(word_subject))

        # print("content : ", tokenize_sentense3(name[1].read()))  # clob 데이터라서 stream을 string으로 변경
        word_content = str(name[1].read())
        word_content = re.sub(r"[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》“”’·.a-zA-Z0-9]+", " ", word_content)
        # word_content = re.sub(r"[^가-힣ㄱ-ㅎㅏ-ㅣ]+", "", word_content)
        list_only_contents_non_token.append(word_content)
        # list_news.append(tokenize_sentense3(word_content))
        list_only_contents.append(tokenize_sentense3(word_content))
        # list.append(list_news)

        word_label = name[2]
        label_list = ['문화', '경제', '기타', '정치', '지역', 'IT_과학', '사회', '스포츠', '국제']
        label_index = [0 for n in range(9)]
        idx=0
        for label_idx in range(len(label_list)):
            if word_label == label_list[label_idx]:
                label_index[label_idx] = 1
                idx = label_idx
                break

        list_only_label.append(label_index)
        # list_only_label.append(idx)

    cur.close()
    conn.close()

    # return list_only_subject, list_only_contents, list_only_label
    return list_only_subject_non_token, list_only_contents, list_only_label

# %%
import numpy as np
address = 'itrinity/irskorea@192.168.2.35:1521/itrinity'
result = db_select(sql(1, 10), address)  # 0- 제목, 1- 본문 2- 라벨
result=np.asarray(result)
result = result.transpose()

#%%
from collections import namedtuple

TaggedDocument = namedtuple('TaggedDocument', 'words tags')

tagged_train_docs = [TaggedDocument(d[0], d[2]) for d in result]
# tagged_train_docs = [TaggedDocument(d, c) for d, c in df_movie[['token_review', 'class']].values]
# tagged_test_docs = [TaggedDocument(d, c) for d, c in df_test[['X_test_tokkened', 'y_test']].values]
print(tagged_train_docs)



# %%
from gensim.models import doc2vec
import sys
import multiprocessing

# reload(sys)
# sys.setdefaultencoding('utf-8')

cores = multiprocessing.cpu_count()

# doc2vec parameters
vector_size = 300
window_size = 15
word_min_count = 2
sampling_threshold = 1e-5
negative_size = 5
train_epoch = 100
dm = 1              # 0 = dbow; 1 = dmpv
worker_count = cores  # number of parallel processes

# print(len(sys.argv))
# if len(sys.argv) >= 3:
#     inputfile = sys.argv[1]
#     modelfile = sys.argv[2]
# else:
#     inputfile = "./data/sample.txt"
#     modelfile = "./model/doc2vec.model"
#
# word2vec_file = modelfile + ".word2vec_format"          # 출력 데이터가 될것이고.
#
# sentences = doc2vec.TaggedLineDocument(inputfile)       # 인풋 데이터가 될것.
sentences = doc2vec.TaggedLineDocument(tagged_train_docs)       # 제목을 가지고 진행....
print("build voca")
# build voca
# doc_vectorizer = doc2vec.Doc2Vec(min_count=word_min_count,
#                                  size=vector_size,
#                                  alpha=0.025,
#                                  min_alpha=0.025,
#                                  seed=1234,
#                                  workers=worker_count,
#                                  window=window_size,
#                                  epochs=train_epoch,
#                                  dm=dm)
doc_vectorizer = doc2vec.Doc2Vec(
    dm=0,            # PV-DBOW / default 1
    dbow_words=1,    # w2v simultaneous with DBOW d2v / default 0
    window=8,        # distance between the predicted word and context words
    # size=300,        # vector size
    alpha=0.025,     # learning-rate
    seed=1234,
    min_count=20,    # ignore with freq lower
    min_alpha=0.025, # min learning-rate
    workers=cores,   # multi cpu
    hs = 1,          # hierarchical softmax / default 0
    negative = 10,   # negative sampling / default 5
)
#%%
print("build voca")
# doc_vectorizer.build_vocab(sentences)
doc_vectorizer.build_vocab(tagged_train_docs)

print("build voca")
import time
start = time.time()
print("start : {}".format(start))
# Train document vectors!
# for epoch in range(10):
#     doc_vectorizer.train(sentences)
#     doc_vectorizer.alpha -= 0.002  # decrease the learning rate
#     doc_vectorizer.min_alpha = doc_vectorizer.alpha  # fix the learning rate, no decay
for epoch in range(10):
    doc_vectorizer.train(sentences, total_examples=doc_vectorizer.corpus_count, epochs=doc_vectorizer.iter)
    doc_vectorizer.alpha -= 0.002 # decrease the learning rate
    doc_vectorizer.min_alpha = doc_vectorizer.alpha # fix the learning rate, no decay
end=time.time()
print("During Time: {}".format(end-start))
# # To save
# doc_vectorizer.save(modelfile)
# doc_vectorizer.save_word2vec_format(word2vec_file, binary=False)
#%%
# model = doc2vec.Doc2Vec.