# %%
from django.http import JsonResponse
from clustering.src.clustring_test import *
import pandas as pd


def show_cluster1(numpy1, result_k, title, max_clu):
    font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
    plt.rc('font', family=font_name)  # 한글 적용
    plt.grid()
    plt.title(title, size=20)
    # plt.scatter(result_k[3][:,0], result_k[3][:, 1],
    #             marker='^', s=100, linewidth=2, edgecolors='k')
    # plt.scatter(result_c_numpy[:,0], result_c_numpy[:,1], result_k[4], marker='o')
    numpy_df = pd.DataFrame(numpy1)

    import numpy as np
    from sklearn.manifold import TSNE

    # 2개의 차원으로 축소
    transformed = TSNE(n_components=2).fit_transform(numpy_df)
    transformed.shape

    xs = transformed[:, 0]
    ys = transformed[:, 1]
    plt.scatter(xs, ys, c=result_k[4])  # 라벨은 색상으로 분류됨

    # mglearn.discrete_scatter(xs,ys,c=result_k[4])
    # plt.show()

    # mglearn.discrete_scatter(numpy1[:, 0], numpy1[:, 1], result_k[4], s=4,
    #                          markers='o', markeredgewidth=0)


    # transformed_k = TSNE(n_components=2).fit_transform(result_k[3])
    # transformed_k.shape
    # plt.scatter(transformed_k[:, 0], transformed_k[:, 1], [n for n in range(0, max_clu)], marker='^', edgecolors="black")
    # mglearn.discrete_scatter(
    #     result_k[3][:, 0], result_k[3][:, 1], [n for n in range(0, max_clu)], s=15,
    #     markers='^', markeredgewidth=2)

# legend가 필요없음.
    # plt.legend([(str(n) + " cluster") for n in range(0, max_clu)], loc="best")
    plt.xlabel(xlabel="feature1")
    plt.ylabel(ylabel="feature2")
    plt.show()


# %%
def post_list_json(request):
    # %% 모듈화 완료
    address = 'itrinity/irskorea@192.168.2.35:1521/itrinity'
    start_num1 = 1
    end_num1 = 1000

    size = 100      # 벡터 사이즈
    mincount = 5    # 최소 반복값

    list1 = db_select(sql(start_num1, end_num1), address)
    model1 = makeModel(list1[0], size=size, min_count=mincount) # 본문으로 하기엔 단어양이 많음. 처리속도 느림.

    # model.save('word2vec.model')
    # model = Word2Vec.load('word2vec.model')
    start_num2 = 1001
    end_num2 = 2001
    list2 = db_select(sql(start_num2, end_num2), address)

    result_c = get_n_tokens(list2[0], model1)
    result_c_numpy = np.array(result_c)
    show_cluster_inertia(result_c)
#%%
    max_clu = 6
    result_k = unsupervised_learning(result_c, model1, list2, 300, max_clu)
    show_cluster1(result_c_numpy, result_k,
                 "news_cnt:{},m_cnt:{}, size:{}, min:{}"
                 .format(end_num1,model1.wv.index2word.__len__(), size, mincount)
                 ,max_clu)
    # %%
    return JsonResponse({  # ndarray는 serialize 가 안됨.
        # 'message1': result_k[0],
        # 'message2': result_k[1],
        'message3': result_k[2],
        # 'message4': result_k[3],
        # 'message5': result_k[4],
        'message6': result_k[5],
        'message7': result_k[6],
        'message8': result_k[7],
        'message9': '안녕 파이썬 장고',
        'items': ['파이썬', '장고', 'AWS', 'Azure'],
    }, json_dumps_params={'ensure_ascii': False})  # True로 바꾸면 한글이 깨지는 문제생김.
