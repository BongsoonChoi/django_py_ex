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
    select TITLE,CONTENT from    
    (select a.*  , rownum rnum  from JSOUP_CONTENT a  WHERE LENGTH(CONTENT) > 0)   
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
    list_only_contents_non_token = []  # 토큰화 없는 본문
    for name in cur:
        # list_news=[] # 제목과 본문을 담기위한 리스트
        # print("테스트 이름 리스트 : ", tokenize_sentense(name[0]))
        word_subject = str(name[0])
        word_subject = re.sub(r"[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》“”’·.a-zA-Z0-9]+", " ", word_subject)
        # 특수문자 영어 숫자 제거.
        # list_news.append(tokenize_sentense(word_subject))
        list_only_subject.append(tokenize_sentense3(word_subject))

        # print("content : ", tokenize_sentense3(name[1].read()))  # clob 데이터라서 stream을 string으로 변경
        word_content = str(name[1].read())
        word_content = re.sub(r"[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》“”’·.a-zA-Z0-9]+", " ", word_content)
        # word_content = re.sub(r"[^가-힣ㄱ-ㅎㅏ-ㅣ]+", "", word_content)
        list_only_contents_non_token.append(word_content)
        # list_news.append(tokenize_sentense3(word_content))
        list_only_contents.append(tokenize_sentense3(word_content))
        # list.append(list_news)

    cur.close()
    conn.close()

    return list_only_subject, list_only_contents


# %%
# df_list = pd.DataFrame(data=list, columns=["subject", "content"])


# %%
# import re
# from nltk.tokenize import word_tokenize, sent_tokenize
#
# tot_string=''
# for item in list_only_contents_non_token:
#     tot_string = tot_string+ item
#
# # content_text = re.sub(r'\([^)]*\)', '', parser)7
# # 정규 표현식의 sub 모듈을 통해 content 중간에 등장하는 (Audio), (Laughter) 등의 배경음 부분을 제거.
# # 해당 코드는 괄호로 구성된 내용을 제거.
#
# sent_text=sent_tokenize(tot_string)
# # 입력 코퍼스에 대해서 NLTK를 이용하여 문장 토큰화를 수행.
#
# normalized_text = []
# for string in sent_text:
#      tokens = re.sub(r"[^a-z0-9]+", " ", string.lower())
#      normalized_text.append(tokens)
# # 각 문장에 대해서 구두점을 제거하고, 대문자를 소문자로 변환.
#
# result = [word_tokenize(sentence) for sentence in normalized_text]
# 각 문장에 대해서 NLTK를 이용하여 단어 토큰화를 수행.


# %%        기사 본문을 가지고 w2v
# model_list=[]

# model = w2v(list_only_contents, size=100, window=2, min_count=20, iter=100, sg=1)    # 모델을 만들기 위한 본문 수집.
def makeModel(data, size, min_count):
    model1 = w2v(data, size=size, window=2, min_count=min_count, iter=300, sg=1)  # 모델을 만들기 위한 본문 수집.
    return model1


# for list_con in list_only_contents :
#     print(list_con[1])
#     model = w2v(list_con[1], size=100, window=2, min_count=50, iter=20, sg=1)
#     model_list.append(model)
# model = w2v(tokenized_data, size=100, window=2, min_count=50, iter=20, sg=1)


# %%
def get_n_tokens(data, model):  # 모델 인덱스 수 * 총 기사의 수 벡터가 만들어짐.
    result = []
    # word_list = ['택시', '분실', '나눔', '판매', '설문', '공지', '상품', '질문', '연락', '구해요']
    word_list = model.wv.index2word  # 모델의 인덱스 가져옴.
    for i, j in enumerate(data):
        dist = [0] * len(word_list)
        if not data[i]:  # 내용이 없는 글
            pass
        elif len(data[i]) == 1:  # 한 토큰으로 된 글
            for l, m in enumerate(word_list):
                try:
                    dist[l] = model.similarity(m, data[0])
                except:
                    continue
        else:  # 한 토큰 이상으로 된 글
            for idx, k in enumerate(j):
                if idx >= 50:  # 토큰 50개만 사용한다. idx >= 50인 이유는, 너무 글의 길이가 길면 similarity의 평균이 점점 낮아진다.
                    break
                for l, m in enumerate(word_list):
                    try:
                        dist[l] += model.similarity(m, k)
                    except:
                        continue
        result.append([x / (idx + 1) for x in dist])  ## 전체 토큰에 유사도를 계산하여 평균낸것.
    return result


# %%
def unsupervised_learning(data, model1, list1, max_iter, n=3):
    # cluster 개수 -> km_model.inertia_ 를 보면 기울기 변화가 적어지는 부분이 3
    km_model = km(n_clusters=n, algorithm='auto', max_iter=max_iter)  # kmeans 알고리즘 설정
    km_model.fit(data)  # 알고리즘 적용
    center = km_model.cluster_centers_
    label = km_model.labels_
    predict_list = km_model.predict(data)  # clustering. predict는 어떻게 사용하는지 파악해야함.
    predict_fit_list = km_model.fit_predict(data)
    score_list = km_model.score(data)
    group_list_topic = [[] for _ in range(n)]  # 잘 됐는지 확인하려는 list
    group_list_data = [[] for _ in range(n)]  # 잘 됐는지 확인하려는 list
    group_list = [[] for _ in range(n)]  # 잘 됐는지 확인하려는 list
    for i in range(n):
        # global model1
        group_list_topic[i] = [model1.wv.index2word[data[idx].index(max(data[idx]))]
                               for idx, j in enumerate(predict_list) if j == i]  # 토픽을 보여주는것.
        group_list_data[i] = [list1[0][idx]
                              for idx, j in enumerate(predict_list) if j == i]  # 기사 데이터를 보여주는것
        group_list[i] = [data[idx]
                         for idx, j in enumerate(predict_list) if j == i]  # 유사도를 보여줌.
        # init_data는 ['이건 하나의 글입니다', '이건 다른 글이죠', ...] 같은 초기 data
        # print(i, 'group is', group_list[i][:10])     # 각 cluster 당 10개씩 출력
    for i in enumerate(predict_list):
        print(i)
    return predict_list, predict_fit_list, score_list, center, label, group_list, group_list_topic, group_list_data


# %%
# total =[]
# for i in result_k[1]:
#     for j in range(len(i)):
#         for k in range(len(i[j])):
#             total.append(i[j][k])

def show_cluster(numpy1, result_k, title, max_clu):
    font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
    plt.rc('font', family=font_name)  # 한글 적용
    plt.grid()
    plt.title(title, size=20)
    # plt.scatter(result_k[3][:,0], result_k[3][:, 1],
    #             marker='^', s=100, linewidth=2, edgecolors='k')
    # plt.scatter(result_c_numpy[:,0], result_c_numpy[:,1], result_k[4], marker='o')
    mglearn.discrete_scatter(numpy1[:, 0], numpy1[:, 1], result_k[4], s=4,
                             markers='o', markeredgewidth=0)

    mglearn.discrete_scatter(
        result_k[3][:, 0], result_k[3][:, 1], [n for n in range(0, max_clu)], s=15,
        markers='^', markeredgewidth=2)



    plt.legend([(str(n) + " cluster") for n in range(0, max_clu)], loc="best")
    plt.xlabel(xlabel="feature1")
    plt.ylabel(ylabel="feature2")
    plt.show()


# %%

def show_cluster_inertia(result_c):
    inertia = []  # cluster 응집도
    for k in range(1, 11):  # 10개까지
        model = km(n_clusters=k);
        model.fit(result_c);
        inertia.append(model.inertia_);
    print(inertia)

    font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
    plt.rc('font', family=font_name)  # 한글 적용

    plt.grid()
    plt.title('k-means elbow', size=20)
    plt.scatter(range(1, 11), inertia, s=10)
    plt.xlabel(xlabel="클러스터 수")
    plt.ylabel(ylabel="응집도")
    # plt.scatter(centers2[:,0], centers2[:,1], s=150, marker='*', color='red')
    plt.show()

# # %% 모듈화 완료
#
#
# address = 'itrinity/irskorea@192.168.2.35:1521/itrinity'
#
# start_num = 1
# end_num = 100
# list1 = db_select(sql(start_num, end_num), address)
# model1 = makeModel(list1[0])
#
# # %%
#
# result_c = get_n_tokens(list1[0], model1)
# result_c_numpy = np.array(result_c)
# for i in range(len(result_c)):
#     print(list1[0][i])
#     print(max(result_c[i]))
#     # 모델의 유사도 중에 최대값
#     print(model1.wv.index2word[result_c[i].index(max(result_c[i]))])
#     # 모델 중에 최대값을 가지는 인덱스의 모델값
#     """
#     최대값이 0 이면 모델중에 제일 앞에있는 값이 나옴. 이에 대한 예외처리가 필요함.
#     """
#
# result_k = unsupervised_learning(result_c, 7)
# # for i in result_c:
# #     max(i)
# #     print(model.wv.index2word[i.index(max(i))])
#
# show_cluster(result_c_numpy, result_k)
# show_cluster_inertia(result_c)
#


# # %%  신규데이터 테스트
#
#
# start_num = 101
# end_num = 200
# list2 = db_select(sql(start_num, end_num), address)
#
# result_c = get_n_tokens(list2[0], model1)
# result_c_numpy = np.array(result_c)
# for i in range(len(result_c)):
#     print(list2[0][i])
#     print(max(result_c[i]))
#     # 모델의 유사도 중에 최대값
#     print(model1.wv.index2word[result_c[i].index(max(result_c[i]))])
#     # 모델 중에 최대값을 가지는 인덱스의 모델값
#     """
#     최대값이 0 이면 모델중에 제일 앞에있는 값이 나옴. 이에 대한 예외처리가 필요함.
#     """
#
# result_k = unsupervised_learning(result_c_numpy, 7)
#
# show_cluster(result_c_numpy, result_k)
# show_cluster_inertia(result_c)
