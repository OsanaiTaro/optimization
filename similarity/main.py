#!/usr/bin/env python
# coding: utf-8

# In[1]:


from cv2 import rectangle, imwrite
import numpy as np
from pandas import DataFrame
from decimal import Decimal
import pareto_GA as ga
from random import randint
import practice_evaluate_sampleple as es
from multiprocessing import Pool
from multiprocessing import Process
from time import perf_counter


# In[2]:


# 設置可能ランドマーク数
GENOM_LENGTH = 111
# １世代ごとの幾何地図の枚数
MAX_GENOM_LIST = 20
# 親個体選択数
SELECT_GENOM = 3
# 個体突然変異確率
INDIVIDUAL_MUTATION = 0.3
# 遺伝子突然変異確率
GENOM_MUTATION = 0.2
# 繰り返す世代数
MAX_GENERATION = 1000

#LANDMARK_NUM = 40


# In[3]:


def create_genom(length):
    """
    :param length: 遺伝子情報の長さ
    :return: 生成した個体集団genomClass
    """
    genome_list = [randint(0, 1) for i in range(length)]
    return ga.genom(genome_list, 0)#評価値0のgenomオブジェクト_osanai


# In[4]:


def evaluation_1(li):
    """
    :return: 評価処理をしたgenomClassを返す
    """

    img = np.full((1050, 1700, 3), 255, dtype=np.uint8)

    #左の壁
    rectangle(img, (435, 410), (435, 636), (0, 0, 0), thickness=-1)
    rectangle(img, (436, 479), (442, 479), (0, 0, 0), thickness=-1)
    rectangle(img, (441, 527), (442, 528), (0, 0, 0), thickness=-1)
    rectangle(img, (436, 576), (442, 576), (0, 0, 0), thickness=-1)
    rectangle(img, (441, 480), (441, 575), (0, 0, 0), thickness=-1)

    #上の柱
    rectangle(img, (541, 411), (542, 419), (0, 0, 0), thickness=-1)
    rectangle(img, (663, 411), (664, 419), (0, 0, 0), thickness=-1)
    rectangle(img, (785, 411), (786, 419), (0, 0, 0), thickness=-1)
    rectangle(img, (907, 411), (908, 419), (0, 0, 0), thickness=-1)
    rectangle(img, (1030, 411), (1031, 419), (0, 0, 0), thickness=-1)
    rectangle(img, (1153, 411), (1154, 419), (0, 0, 0), thickness=-1)

    #下の壁
    rectangle(img, (457, 641), (1246, 641), (0, 0, 0), thickness=-1)

    #右の壁
    rectangle(img, (1163+100, 310+100), (1163+100, 536+100), (0, 0, 0), thickness=-1)
    rectangle(img, (1156+100, 379+100), (1162+100, 379+100), (0, 0, 0), thickness=-1)
    rectangle(img, (1156+100, 427+100), (1157+100, 428+100), (0, 0, 0), thickness=-1)
    rectangle(img, (1156+100, 476+100), (1162+100, 476+100), (0, 0, 0), thickness=-1)
    rectangle(img, (1157+100, 380+100), (1157+100, 475+100), (0, 0, 0), thickness=-1)

    #支柱の表示_1
    rectangle(img, (421+100, 457), (421+100, 457), (0, 0, 0), thickness=-1)
    rectangle(img, (482+100, 457), (482+100, 457), (0, 0, 0), thickness=-1)
    rectangle(img, (542+100, 457), (542+100, 457), (0, 0, 0), thickness=-1)
    rectangle(img, (603+100, 457), (603+100, 457), (0, 0, 0), thickness=-1)
    rectangle(img, (665+100, 457), (665+100, 457), (0, 0, 0), thickness=-1)
    rectangle(img, (726+100, 457), (726+100, 457), (0, 0, 0), thickness=-1)
    rectangle(img, (786+100, 457), (786+100, 457), (0, 0, 0), thickness=-1)
    rectangle(img, (847+100, 457), (847+100, 457), (0, 0, 0), thickness=-1)
    rectangle(img, (908+100, 457), (908+100, 457), (0, 0, 0), thickness=-1)
    rectangle(img, (949+100, 457), (949+100, 457), (0, 0, 0), thickness=-1)
    rectangle(img, (991+100, 457), (991+100, 457), (0, 0, 0), thickness=-1)
    rectangle(img, (1052+100, 457), (1052+100, 457), (0, 0, 0), thickness=-1)
    rectangle(img, (1113+100, 457), (1113+100, 457), (0, 0, 0), thickness=-1)

    #支柱の表示_2
    rectangle(img, (421+100, 474), (421+100, 474), (0, 0, 0), thickness=-1)
    rectangle(img, (482+100, 474), (482+100, 474), (0, 0, 0), thickness=-1)
    rectangle(img, (542+100, 474), (542+100, 474), (0, 0, 0), thickness=-1)
    rectangle(img, (603+100, 474), (603+100, 474), (0, 0, 0), thickness=-1)
    rectangle(img, (665+100, 474), (665+100, 474), (0, 0, 0), thickness=-1)
    rectangle(img, (726+100, 474), (726+100, 474), (0, 0, 0), thickness=-1)
    rectangle(img, (786+100, 474), (786+100, 474), (0, 0, 0), thickness=-1)
    rectangle(img, (847+100, 474), (847+100, 474), (0, 0, 0), thickness=-1)
    rectangle(img, (908+100, 474), (908+100, 474), (0, 0, 0), thickness=-1)
    rectangle(img, (949+100, 474), (949+100, 474), (0, 0, 0), thickness=-1)
    rectangle(img, (991+100, 474), (991+100, 474), (0, 0, 0), thickness=-1)
    rectangle(img, (1052+100, 474), (1052+100, 474), (0, 0, 0), thickness=-1)
    rectangle(img, (1113+100, 474), (1113+100, 474), (0, 0, 0), thickness=-1)

    #支柱の表示_3
    rectangle(img, (421+100, 517), (421+100, 517), (0, 0, 0), thickness=-1)
    rectangle(img, (482+100, 517), (482+100, 517), (0, 0, 0), thickness=-1)
    rectangle(img, (542+100, 517), (542+100, 517), (0, 0, 0), thickness=-1)
    rectangle(img, (603+100, 517), (603+100, 517), (0, 0, 0), thickness=-1)
    rectangle(img, (665+100, 517), (665+100, 517), (0, 0, 0), thickness=-1)
    rectangle(img, (726+100, 517), (726+100, 517), (0, 0, 0), thickness=-1)
    rectangle(img, (786+100, 517), (786+100, 517), (0, 0, 0), thickness=-1)
    rectangle(img, (847+100, 517), (847+100, 517), (0, 0, 0), thickness=-1)
    rectangle(img, (908+100, 517), (908+100, 517), (0, 0, 0), thickness=-1)
    rectangle(img, (949+100, 517), (949+100, 517), (0, 0, 0), thickness=-1)
    rectangle(img, (991+100, 517), (991+100, 517), (0, 0, 0), thickness=-1)
    rectangle(img, (1052+100, 517), (1052+100, 517), (0, 0, 0), thickness=-1)
    rectangle(img, (1113+100, 517), (1113+100, 517), (0, 0, 0), thickness=-1)

    #支柱の表示_4
    rectangle(img, (421+100, 534), (421+100, 534), (0, 0, 0), thickness=-1)
    rectangle(img, (482+100, 534), (482+100, 534), (0, 0, 0), thickness=-1)
    rectangle(img, (542+100, 534), (542+100, 534), (0, 0, 0), thickness=-1)
    rectangle(img, (603+100, 534), (603+100, 534), (0, 0, 0), thickness=-1)
    rectangle(img, (665+100, 534), (665+100, 534), (0, 0, 0), thickness=-1)
    rectangle(img, (726+100, 534), (726+100, 534), (0, 0, 0), thickness=-1)
    rectangle(img, (786+100, 534), (786+100, 534), (0, 0, 0), thickness=-1)
    rectangle(img, (847+100, 534), (847+100, 534), (0, 0, 0), thickness=-1)
    rectangle(img, (908+100, 534), (908+100, 534), (0, 0, 0), thickness=-1)
    rectangle(img, (949+100, 534), (949+100, 534), (0, 0, 0), thickness=-1)
    rectangle(img, (991+100, 534), (991+100, 534), (0, 0, 0), thickness=-1)
    rectangle(img, (1052+100, 534), (1052+100, 534), (0, 0, 0), thickness=-1)
    rectangle(img, (1113+100, 534), (1113+100, 534), (0, 0, 0), thickness=-1)

    #支柱の表示_5
    rectangle(img, (421+100, 575), (421+100, 575), (0, 0, 0), thickness=-1)
    rectangle(img, (482+100, 575), (482+100, 575), (0, 0, 0), thickness=-1)
    rectangle(img, (542+100, 575), (542+100, 575), (0, 0, 0), thickness=-1)
    rectangle(img, (603+100, 575), (603+100, 575), (0, 0, 0), thickness=-1)
    rectangle(img, (665+100, 575), (665+100, 575), (0, 0, 0), thickness=-1)
    rectangle(img, (726+100, 575), (726+100, 575), (0, 0, 0), thickness=-1)
    rectangle(img, (786+100, 575), (786+100, 575), (0, 0, 0), thickness=-1)
    rectangle(img, (847+100, 575), (847+100, 575), (0, 0, 0), thickness=-1)
    rectangle(img, (908+100, 575), (908+100, 575), (0, 0, 0), thickness=-1)
    rectangle(img, (949+100, 575), (949+100, 575), (0, 0, 0), thickness=-1)
    rectangle(img, (991+100, 575), (991+100, 575), (0, 0, 0), thickness=-1)
    rectangle(img, (1052+100, 575), (1052+100, 575), (0, 0, 0), thickness=-1)
    rectangle(img, (1113+100, 575), (1113+100, 575), (0, 0, 0), thickness=-1)

    #支柱の表示_6
    rectangle(img, (421+100, 592), (421+100, 592), (0, 0, 0), thickness=-1)
    rectangle(img, (482+100, 592), (482+100, 592), (0, 0, 0), thickness=-1)
    rectangle(img, (542+100, 592), (542+100, 592), (0, 0, 0), thickness=-1)
    rectangle(img, (603+100, 592), (603+100, 592), (0, 0, 0), thickness=-1)
    rectangle(img, (665+100, 592), (665+100, 592), (0, 0, 0), thickness=-1)
    rectangle(img, (726+100, 592), (726+100, 592), (0, 0, 0), thickness=-1)
    rectangle(img, (786+100, 592), (786+100, 592), (0, 0, 0), thickness=-1)
    rectangle(img, (847+100, 592), (847+100, 592), (0, 0, 0), thickness=-1)
    rectangle(img, (908+100, 592), (908+100, 592), (0, 0, 0), thickness=-1)
    rectangle(img, (949+100, 592), (949+100, 592), (0, 0, 0), thickness=-1)
    rectangle(img, (991+100, 592), (991+100, 592), (0, 0, 0), thickness=-1)
    rectangle(img, (1052+100, 592), (1052+100, 592), (0, 0, 0), thickness=-1)
    rectangle(img, (1113+100, 592), (1113+100, 592), (0, 0, 0), thickness=-1)

    #ランドマークの表示
    list_land = [(421+100, 357+100), (421+100, 374+100), (421+100, 357+100), (482+100, 357+100), (421+100, 374+100), (482+100, 374+100), (482+100, 357+100), (482+100, 374+100), (482+100, 357+100), (542+100, 357+100), (482+100, 374+100), (542+100, 374+100), (542+100, 357+100), (542+100, 374+100),
            (542+100, 357+100), (603+100, 357+100), (542+100, 374+100), (603+100, 374+100), (603+100, 357+100), (603+100, 374+100), (603+100, 357+100), (665+100, 357+100), (603+100, 374+100), (665+100, 374+100), (665+100, 357+100), (665+100, 374+100), (665+100, 357+100), (726+100, 357+100),
            (665+100, 374+100), (726+100, 374+100), (726+100, 357+100), (726+100, 374+100), (726+100, 357+100), (786+100, 357+100), (726+100, 374+100), (786+100, 374+100), (786+100, 357+100), (786+100, 374+100), (786+100, 357+100), (847+100, 357+100), (786+100, 374+100), (847+100, 374+100),
            (847+100, 357+100), (847+100, 374+100), (847+100, 357+100), (908+100, 357+100), (847+100, 374+100), (908+100, 374+100), (908+100, 357+100), (908+100, 374+100), (908+100, 357+100), (949+100, 357+100), (908+100, 374+100), (949+100, 374+100), (949+100, 357+100), (949+100, 374+100),
            (949+100, 357+100), (991+100, 357+100), (949+100, 374+100), (991+100, 374+100), (991+100, 357+100), (991+100, 374+100), (991+100, 357+100), (1052+100, 357+100), (991+100, 374+100), (1052+100, 374+100), (1052+100, 357+100), (1052+100, 374+100), (1052+100, 357+100), (1113+100, 357+100),
            (1052+100, 374+100), (1113+100, 374+100), (1113+100, 357+100), (1113+100, 374+100),
            (421+100, 417+100), (421+100, 434+100), (421+100, 417+100), (482+100, 417+100), (421+100, 434+100), (482+100, 434+100), (482+100, 417+100), (482+100, 434+100), (482+100, 417+100), (542+100, 417+100), (482+100, 434+100), (542+100, 434+100), (542+100, 417+100), (542+100, 434+100),
            (542+100, 417+100), (603+100, 417+100), (542+100, 434+100), (603+100, 434+100), (603+100, 417+100), (603+100, 434+100), (603+100, 417+100), (665+100, 417+100), (603+100, 434+100), (665+100, 434+100), (665+100, 417+100), (665+100, 434+100), (665+100, 417+100), (726+100, 417+100),
            (665+100, 434+100), (726+100, 434+100), (726+100, 417+100), (726+100, 434+100), (726+100, 417+100), (786+100, 417+100), (726+100, 434+100), (786+100, 434+100), (786+100, 417+100), (786+100, 434+100), (786+100, 417+100), (847+100, 417+100), (786+100, 434+100), (847+100, 434+100),
            (847+100, 417+100), (847+100, 434+100), (847+100, 417+100), (908+100, 417+100), (847+100, 434+100), (908+100, 434+100), (908+100, 417+100), (908+100, 434+100), (908+100, 417+100), (949+100, 417+100), (908+100, 434+100), (949+100, 434+100), (949+100, 417+100), (949+100, 434+100),
            (949+100, 417+100), (991+100, 417+100), (949+100, 434+100), (991+100, 434+100), (991+100, 417+100), (991+100, 434+100), (991+100, 417+100), (1052+100, 417+100), (991+100, 434+100), (1052+100, 434+100), (1052+100, 417+100), (1052+100, 434+100), (1052+100, 417+100), (1113+100, 417+100),
            (1052+100, 434+100), (1113+100, 434+100), (1113+100, 417+100), (1113+100, 434+100), 
            (421+100, 475+100), (421+100, 492+100), (421+100, 475+100), (482+100, 475+100), (421+100, 492+100), (482+100, 492+100), (482+100, 475+100), (482+100, 492+100), (482+100, 475+100), (542+100, 475+100), (482+100, 492+100), (542+100, 492+100), (542+100, 475+100), (542+100, 492+100),
            (542+100, 475+100), (603+100, 475+100), (542+100, 492+100), (603+100, 492+100), (603+100, 475+100), (603+100, 492+100), (603+100, 475+100), (665+100, 475+100), (603+100, 492+100), (665+100, 492+100), (665+100, 475+100), (665+100, 492+100), (665+100, 475+100), (726+100, 475+100),
            (665+100, 492+100), (726+100, 492+100), (726+100, 475+100), (726+100, 492+100), (726+100, 475+100), (786+100, 475+100), (726+100, 492+100), (786+100, 492+100), (786+100, 475+100), (786+100, 492+100), (786+100, 475+100), (847+100, 475+100), (786+100, 492+100), (847+100, 492+100),
            (847+100, 475+100), (847+100, 492+100), (847+100, 475+100), (908+100, 475+100), (847+100, 492+100), (908+100, 492+100), (908+100, 475+100), (908+100, 492+100), (908+100, 475+100), (949+100, 475+100), (908+100, 492+100), (949+100, 492+100), (949+100, 475+100), (949+100, 492+100),
            (949+100, 475+100), (991+100, 475+100), (949+100, 492+100), (991+100, 492+100), (991+100, 475+100), (991+100, 492+100), (991+100, 475+100), (1052+100, 475+100), (991+100, 492+100), (1052+100, 492+100), (1052+100, 475+100), (1052+100, 492+100), (1052+100, 475+100), (1113+100, 475+100),
            (1052+100, 492+100), (1113+100, 492+100), (1113+100, 475+100), (1113+100, 492+100)]
    
    data_map_dic = {}
    for d in range(GENOM_LENGTH):
        if li[d] == 1:
            rectangle(img, list_land[2*d], list_land[2*d+1], (0,0,0), thickness=-1)

    
    #画像生成
    imwrite('1221_pareto_roulette_1000_%d.png' % current_generation_individual_genomlist.index(li), img)
    #imwrite('../map_evaluate/debug/uc_taromap_noprop_%d.png' % current_generation_individual_group.index(ga), img)

    #Nscore算出！！！！
    sample = es.evaluate_sample(urg_pix=400, geomap='1221_pareto_roulette_1000_%d.png' % current_generation_individual_genomlist.index(li))

    count = sample.map_from_img()
    
    map_init_u = sample.map_init_u
    urg_scan = sample.urg_scan
    bresenham = sample.bresenham
    feature_df = sample.feature_df
    urg_scan_rotation = sample.urg_scan_rotation
    bresenham_rotation = sample.bresenham_rotation
    feature_df_rotation = sample.feature_df_rotation
    
    Urg_similarity = sample.Urg_similarity
    Urg_similarity_rotation = sample.Urg_similarity_rotation
    Urg_similarity_rotation_both = sample.Urg_similarity_rotation_both
    
    for d in range(count):
        map_init_u(current_num=d)
        urg_scan(current_num=d)
        bresenham(current_num=d)
        feature_df(current_num=d,total_num=count)
        urg_scan_rotation(current_num=d)
        bresenham_rotation(current_num=d)
        feature_df_rotation(current_num=d,total_num=count)

    resemblance = [Urg_similarity(current_num=d,total_num=count) for d in range(count)]
    resemblance_rotation = [Urg_similarity_rotation(current_num=d,total_num=count) for d in range(count)]
    resemblance_rotation_both = [Urg_similarity_rotation_both(current_num=d,total_num=count) for d in range(count)]

    replace = np.array(resemblance)#縦横0度同士path1~3_osanai
    replace_rotation = np.array(resemblance_rotation)#縦180度，横0度path1~3_osanai
    replace_rotation_both = np.array(resemblance_rotation_both)#縦横180度同士path1~3_osanai

    df1 = DataFrame(replace)
    df1_sum = df1.sum()
    df1_sum_sum = (df1_sum.sum()-len(sample.robot_points))/2#同じとこ除外(-類似度1*114)と、かぶってるとこ除外(÷２),要素数は(114*114)/2_osanai

    df2 = DataFrame(replace_rotation)
    df2_sum = df2.sum()
    df2_sum_sum = df2_sum.sum()#要素数は114*114_osanai

    df3 = DataFrame(replace_rotation_both)
    df3_sum = df3.sum()
    df3_sum_sum = (df3_sum.sum()-len(sample.robot_points))/2#同じとこ除外(-類似度1*114)と、かぶってるとこ除外(÷２),要素数は(114*114)/2_osanai

    Nscore = (df1_sum_sum + df2_sum_sum + df3_sum_sum)/(len(sample.robot_points)*len(sample.robot_points)*2-len(sample.robot_points))#114*2C2(組み合わせ)_osanai
    
    print("map" + str(current_generation_individual_genomlist.index(li)), Nscore, li)
    return Nscore

    #genom_total = sum(ga.getGenom())
    #return Decimal(genom_total) / Decimal(GENOM_LENGTH)

#def evaluation_2(li):
    #landmark_num_er = abs(sum(li) - LANDMARK_NUM)
    #print("map" + str(current_generation_individual_genomlist.index(li)), landmark_num_er, li)
    #return landmark_num_er


# In[5]:


def select(ga, elite):
    """選択関数です。エリート選択を行います
    評価が高い順番にソートを行った後、一定以上
    :param ga: 選択を行うgenomClassの配列
    :return: 選択処理をした一定のエリート、genomClassを返す
    """
    result = []
    p = []
    p_sum_list = []
    # 現行世代個体集団の評価を高い順番にソートする
    sort_result = sorted(ga, key=lambda u: u.evaluation_1)#新たなリストを生成_1206osanai

    print("sort_result_len_0:" + str(len(sort_result)))#20個のはず
    for i in range(0, MAX_GENOM_LIST):
        p.append(100000000000000000000/(sort_result[i].getEvaluation_1()))
    print("p_len_0:" + str(len(p)))#20個のはず
    for i in range(len(p)):
        p_sum_list.append(sum(p[0:i+1]))
    #print("p_sum_list_0:" + str(p_sum_list))#どんどん大きくなる20個のはず
    # 一定の上位を抽出する
    #result = [sort_result.pop(0) for i in range(elite)]

    r = randint(0, p_sum_list[len(p_sum_list) - 1])
    if r >= p_sum_list[len(p_sum_list) - 2]:
        result.append(sort_result.pop(len(sort_result) - 1))
    elif r >= p_sum_list[len(p_sum_list) - 3]:
        result.append(sort_result.pop(len(sort_result) - 2))
    elif r >= p_sum_list[len(p_sum_list) - 4]:
        result.append(sort_result.pop(len(sort_result) - 3))
    elif r >= p_sum_list[len(p_sum_list) - 5]:
        result.append(sort_result.pop(len(sort_result) - 4))
    elif r >= p_sum_list[len(p_sum_list) - 6]:
        result.append(sort_result.pop(len(sort_result) - 5))
    elif r >= p_sum_list[len(p_sum_list) - 7]:
        result.append(sort_result.pop(len(sort_result) - 6))
    elif r >= p_sum_list[len(p_sum_list) - 8]:
        result.append(sort_result.pop(len(sort_result) - 7))    	
    elif r >= p_sum_list[len(p_sum_list) - 9]:
        result.append(sort_result.pop(len(sort_result) - 8))    	
    elif r >= p_sum_list[len(p_sum_list) - 10]:
        result.append(sort_result.pop(len(sort_result) - 9))    	
    elif r >= p_sum_list[len(p_sum_list) - 11]:
        result.append(sort_result.pop(len(sort_result) - 10))
    elif r >= p_sum_list[len(p_sum_list) - 12]:
        result.append(sort_result.pop(len(sort_result) - 11))    	
    elif r >= p_sum_list[len(p_sum_list) - 13]:
        result.append(sort_result.pop(len(sort_result) - 12))
    elif r >= p_sum_list[len(p_sum_list) - 14]:
        result.append(sort_result.pop(len(sort_result) - 13))    	
    elif r >= p_sum_list[len(p_sum_list) - 15]:
        result.append(sort_result.pop(len(sort_result) - 14))
    elif r >= p_sum_list[len(p_sum_list) - 16]:
        result.append(sort_result.pop(len(sort_result) - 15))    	
    elif r >= p_sum_list[len(p_sum_list) - 17]:
        result.append(sort_result.pop(len(sort_result) - 16))    	
    elif r >= p_sum_list[len(p_sum_list) - 18]:
        result.append(sort_result.pop(len(sort_result) - 17))
    elif r >= p_sum_list[len(p_sum_list) - 19]:
        result.append(sort_result.pop(len(sort_result) - 18))                
    else:
        result.append(sort_result.pop(len(sort_result) - 19))
    
    p = []
    p_sum_list = []
    print("sort_result_len_1:" + str(len(sort_result)))#19のはず
 
    for i in range(0, MAX_GENOM_LIST-1):
        p.append(100000000000000000000/(sort_result[i].getEvaluation_1()))
    print("p_len_1:" + str(len(p)))#19のはず

    for i in range(len(p)):
        p_sum_list.append(sum(p[0:i+1]))
    #print("p_1_sum_list:" + str(p_sum_list))#どんどん大きくなる19個のはず

    r = randint(0, p_sum_list[len(p_sum_list) - 1])
    if r >= p_sum_list[len(p_sum_list) - 2]:
        result.append(sort_result.pop(len(sort_result) - 1))
    elif r >= p_sum_list[len(p_sum_list) - 3]:
        result.append(sort_result.pop(len(sort_result) - 2))
    elif r >= p_sum_list[len(p_sum_list) - 4]:
        result.append(sort_result.pop(len(sort_result) - 3))
    elif r >= p_sum_list[len(p_sum_list) - 5]:
        result.append(sort_result.pop(len(sort_result) - 4))
    elif r >= p_sum_list[len(p_sum_list) - 6]:
        result.append(sort_result.pop(len(sort_result) - 5))
    elif r >= p_sum_list[len(p_sum_list) - 7]:
        result.append(sort_result.pop(len(sort_result) - 6))
    elif r >= p_sum_list[len(p_sum_list) - 8]:
        result.append(sort_result.pop(len(sort_result) - 7))    	
    elif r >= p_sum_list[len(p_sum_list) - 9]:
        result.append(sort_result.pop(len(sort_result) - 8))    	
    elif r >= p_sum_list[len(p_sum_list) - 10]:
        result.append(sort_result.pop(len(sort_result) - 9))    	
    elif r >= p_sum_list[len(p_sum_list) - 11]:
        result.append(sort_result.pop(len(sort_result) - 10))
    elif r >= p_sum_list[len(p_sum_list) - 12]:
        result.append(sort_result.pop(len(sort_result) - 11))    	
    elif r >= p_sum_list[len(p_sum_list) - 13]:
        result.append(sort_result.pop(len(sort_result) - 12))
    elif r >= p_sum_list[len(p_sum_list) - 14]:
        result.append(sort_result.pop(len(sort_result) - 13))    	
    elif r >= p_sum_list[len(p_sum_list) - 15]:
        result.append(sort_result.pop(len(sort_result) - 14))
    elif r >= p_sum_list[len(p_sum_list) - 16]:
        result.append(sort_result.pop(len(sort_result) - 15))    	
    elif r >= p_sum_list[len(p_sum_list) - 17]:
        result.append(sort_result.pop(len(sort_result) - 16))    	
    elif r >= p_sum_list[len(p_sum_list) - 18]:
        result.append(sort_result.pop(len(sort_result) - 17))
    elif r >= p_sum_list[len(p_sum_list) - 19]:
        result.append(sort_result.pop(len(sort_result) - 18))                
    else:
        result.append(sort_result.pop(len(sort_result) - 19))

    p = []
    p_sum_list = []
    print("sort_result_len_2:" + str(len(sort_result)))#18のはず
 
    for i in range(0, MAX_GENOM_LIST-2):
        p.append(100000000000000000000/(sort_result[i].getEvaluation_1()))
    print("p_len_2:" + str(len(p)))#18のはず

    for i in range(len(p)):
        p_sum_list.append(sum(p[0:i+1]))
    #print("p_2_sum_list:" + str(p_sum_list))#どんどん大きくなる18個のはず

    r = randint(0, p_sum_list[len(p_sum_list) - 1])
    if r >= p_sum_list[len(p_sum_list) - 2]:
        result.append(sort_result.pop(len(sort_result) - 1))
    elif r >= p_sum_list[len(p_sum_list) - 3]:
        result.append(sort_result.pop(len(sort_result) - 2))
    elif r >= p_sum_list[len(p_sum_list) - 4]:
        result.append(sort_result.pop(len(sort_result) - 3))
    elif r >= p_sum_list[len(p_sum_list) - 5]:
        result.append(sort_result.pop(len(sort_result) - 4))
    elif r >= p_sum_list[len(p_sum_list) - 6]:
        result.append(sort_result.pop(len(sort_result) - 5))
    elif r >= p_sum_list[len(p_sum_list) - 7]:
        result.append(sort_result.pop(len(sort_result) - 6))
    elif r >= p_sum_list[len(p_sum_list) - 8]:
        result.append(sort_result.pop(len(sort_result) - 7))    	
    elif r >= p_sum_list[len(p_sum_list) - 9]:
        result.append(sort_result.pop(len(sort_result) - 8))    	
    elif r >= p_sum_list[len(p_sum_list) - 10]:
        result.append(sort_result.pop(len(sort_result) - 9))    	
    elif r >= p_sum_list[len(p_sum_list) - 11]:
        result.append(sort_result.pop(len(sort_result) - 10))
    elif r >= p_sum_list[len(p_sum_list) - 12]:
        result.append(sort_result.pop(len(sort_result) - 11))    	
    elif r >= p_sum_list[len(p_sum_list) - 13]:
        result.append(sort_result.pop(len(sort_result) - 12))
    elif r >= p_sum_list[len(p_sum_list) - 14]:
        result.append(sort_result.pop(len(sort_result) - 13))    	
    elif r >= p_sum_list[len(p_sum_list) - 15]:
        result.append(sort_result.pop(len(sort_result) - 14))
    elif r >= p_sum_list[len(p_sum_list) - 16]:
        result.append(sort_result.pop(len(sort_result) - 15))    	
    elif r >= p_sum_list[len(p_sum_list) - 17]:
        result.append(sort_result.pop(len(sort_result) - 16))    	
    elif r >= p_sum_list[len(p_sum_list) - 18]:
        result.append(sort_result.pop(len(sort_result) - 17))                
    else:
        result.append(sort_result.pop(len(sort_result) - 18))

    print("result:" + str(len(result)))#3のはず	   		
    return result


# In[6]:


def crossover(ga_one, ga_second):
    """交叉関数です。二点交叉を行います。
    :param ga: 交叉させるgenomClassの配列
    :param ga_one:
    :param ga_second:
    :return: 二つの子孫genomClassを格納したリスト返す
    """
    # 子孫を格納するリストを生成します
    genom_list = []
    # 入れ替える二点の点を設定します→[1:25]
    cross_one = randint(0, GENOM_LENGTH)
    cross_second = randint(cross_one, GENOM_LENGTH)
    # 遺伝子を取り出します
    one = ga_one.getGenom()
    second = ga_second.getGenom()
    # 交叉させます
    progeny_one = one[:cross_one] + second[cross_one:cross_second] + one[cross_second:]
    progeny_second = second[:cross_one] + one[cross_one:cross_second] + second[cross_second:]
    # genomClassインスタンスを生成して子孫をリストに格納する
    genom_list.append(ga.genom(progeny_one, 0))
    genom_list.append(ga.genom(progeny_second, 0))
    return genom_list


# In[7]:


def next_generation_gene_create(ga, ga_progeny):
    """
    世代交代処理を行います
    :param ga: 現行世代個体集団
    :param ga_progeny: 現行世代子孫集団
    :return: 次世代個体集団
    """
    # 現行世代個体集団の評価を低い順番にソートする
    next_generation_geno = sorted(ga, reverse=True, key=lambda u: u.evaluation_1)
    
    # 追加する子孫集団の合計ぶんを取り除く
    for i in range(0, len(ga_progeny)):
        next_generation_geno.pop(0)

    # 子孫集団を次世代集団へ追加します
    next_generation_geno = ga_progeny + next_generation_geno#子孫集団をリストの最初に追加_osanai

    return next_generation_geno


# In[8]:


def mutation(ga, induvidual_mutation, genom_mutation):
    """突然変異関数です。
    :param ga: genomClass
    :return: 突然変異処理をしたgenomClassを返す"""
    ga_list = []
    for i in ga[0:(MAX_GENOM_LIST - 1)]:
        # 個体に対して一定の確率で突然変異が起きる
        if induvidual_mutation > (randint(0, 100) / Decimal(100)):
            genom_list = []
            for i_ in i.getGenom():
                # 個体の遺伝子情報一つ一つに対して突然変異がおこる
                if genom_mutation > (randint(0, 100) / Decimal(100)):
                    genom_list.append(randint(0, 1))
                else:
                    genom_list.append(i_)
            i.setGenom(genom_list)
            ga_list.append(i)
        else:
            ga_list.append(i)
    ga_list.append(ga[MAX_GENOM_LIST - 1])
    print("ga_list_len：" + str(len(ga_list)))
    return ga_list


# In[9]:


def multi_1(n):
    p = Pool(10) #最大プロセス数:10
    result_1 = p.map(evaluation_1, n)
    #print(result_1)
    return result_1


# In[10]:


if __name__ == '__main__':

    # 一番最初の現行世代個体集団を生成します。
    current_generation_individual_group = []

    #min_group = []
    #max_group = []
    #avg_group = []

    #min_geomap_group = []
    current_generation_individual_genomlist = []
    next_generation_individual_genomlist = []
    for i in range(MAX_GENOM_LIST):
        current_generation_individual_group.append(create_genom(GENOM_LENGTH))
        current_generation_individual_genomlist.append(current_generation_individual_group[i].getGenom())
    for count_ in range(1, MAX_GENERATION + 1):
        start_time = perf_counter()
        evaluation_1_result_1 = multi_1(current_generation_individual_genomlist[0:10])#current_generation_individual_genomlist=二次元リスト_1129osanai
        evaluation_1_result_2 = multi_1(current_generation_individual_genomlist[10:20])
        evaluation_1_result = evaluation_1_result_1 + evaluation_1_result_2
        print(evaluation_1_result)
        end_time = perf_counter()
        elapsed_time = end_time - start_time
        print("評価時間：{}".format(elapsed_time))
#        evaluation_2_result_1 = multi_2(current_generation_individual_genomlist[0:10])
#        evaluation_2_result_2 = multi_2(current_generation_individual_genomlist[10:20])
#        evaluation_2_result = evaluation_2_result_1 + evaluation_2_result_2
#        print(evaluation_2_result)
        
#        rank_list = []
#        for i in range (0, MAX_GENOM_LIST):
#            rank_evaluatuon_1_bool = []
#            rank_evaluatuon_2_bool = []
#            rank_bool = []
#            rank = 0

#            rank_evaluatuon_1_bool = [evaluation_1_result[i] >= j for j in evaluation_1_result]
#            rank_evaluatuon_2_bool = [evaluation_2_result[i] >= j for j in evaluation_2_result]
#            rank_bool = []
#            for k in range (0, MAX_GENOM_LIST):
#                rank_bool.append(bool(rank_evaluatuon_1_bool[k] == True and rank_evaluatuon_2_bool[k] == True))
#            rank = sum(j == True for j in rank_bool)

#            rank_list.append(rank)
#        print(rank_list)
        for i in range (0, MAX_GENOM_LIST):
            current_generation_individual_group[i].setEvaluation_1(evaluation_1_result[i])#対象のゲノムオブジェクトに評価値をセット_osanai
#            current_generation_individual_group[i].setEvaluation_2(evaluation_2_result[i])
#            current_generation_individual_group[i].setRank(rank_list[i])
        #print("現世代個体は" + str(current_generation_individual_group))#9個体いるはず_osanai
        
        # エリート個体を選択します
        elite_genes = select(current_generation_individual_group,SELECT_GENOM)
        #print("エリート個体は" + str(elite_genes))#3個体いるはず_osanai

        # エリート遺伝子を交叉させ、リストに格納します
        progeny_gene = []
        for i in range(SELECT_GENOM):
            progeny_gene.extend(crossover(elite_genes[i - 1], elite_genes[i]))

        # 次世代個体集団を現行世代、子孫集団から作成します
        next_generation_individual_group = next_generation_gene_create(current_generation_individual_group, progeny_gene)

        # 次世代個体集団の一番類似度が低い個体以外に突然変異を施します
        next_generation_individual_group = mutation(next_generation_individual_group,INDIVIDUAL_MUTATION,GENOM_MUTATION)

        next_generation_individual_genomlist = []
        for i in range(MAX_GENOM_LIST):
            next_generation_individual_genomlist.append(next_generation_individual_group[i].getGenom())

        # 1世代の進化的計算終了。評価に移ります

        # 各個体適用度を配列化します。
        fits_1 = [i.getEvaluation_1() for i in current_generation_individual_group]
#        fits_2 = [i.getEvaluation_2() for i in current_generation_individual_group]
#        fits_3 = [i.getRank() for i in current_generation_individual_group]

        #print("現行世代の類似度は" + str(fits_1))
#        print("現行世代のランドマーク数は" + str(fits_2))
#        print("現行世代のランクは" + str(fits_3))

        # 進化結果を評価します
        min_1 = min(fits_1)
        max_1 = max(fits_1)
        avg_1 = Decimal(str(sum(fits_1))) / Decimal(len(fits_1))
        avg_1 = float(avg_1)

#        min_2 = min(fits_2)
#        max_2 = max(fits_2)
#        avg_2 = Decimal(str(sum(fits_2))) / Decimal(len(fits_2))
#        avg_2 = float(avg_2)

        # 現行世代の進化結果を出力します
        print ("-----第{}世代の結果-----".format(count_))
        print ("  Min_1:{}".format(min_1))
        print ("  Max_1:{}".format(max_1))
        print ("  Avg_1:{}".format(avg_1))

#        print ("  Min_2:{}".format(min_2))
#        print ("  Max_2:{}".format(max_2))
#        print ("  Avg_2:{}".format(avg_2))

        #min_group.append(min_)
        #max_group.append(max_)
        #avg_group.append(avg_)
        #print("現時点で各世代の最小値は{}".format(min_group))
        #print("現時点で各世代の最大値は{}".format(max_group))
        #print("現時点で各世代の平均値は{}".format(avg_group))
        file = open('1221_all_1.txt', 'a')
        file.write(str(count_) + " " + str(fits_1) + "\n")
        file.close()

#        file = open('1203_all_2.txt', 'a')
#        file.write(str(count_) + " " + str(fits_2) + "\n")
#        file.close()

        sorted_group = sorted(current_generation_individual_group, key=lambda u: u.evaluation_1)
        file = open('1221_min_1.txt', 'a')
        file.write(str(count_) + " " + str(sorted_group[0].getEvaluation_1()) + "\n")
        file.close()

        file = open('1221_max_1.txt', 'a')
        file.write(str(count_) + " " + str(max_1) + "\n")
        file.close()

        file = open('1221_avg_1.txt', 'a')
        file.write(str(count_) + " " + str(avg_1) + "\n")
        file.close()

        file = open('1221_minmap_1.txt', 'a')
        file.write(str(count_) + " " + str(sorted_group[0].getGenom()) + "\n")
        file.close()

        file = open('1221_minmap_1_landmark_number.txt', 'a')
        file.write(str(count_) + " " + str(sum(eval(str(sorted_group[0].getGenom()).rstrip()))) + "\n")
        file.close()
        # 現行世代と次世代を入れ替えます
        current_generation_individual_group = next_generation_individual_group

        current_generation_individual_genomlist = next_generation_individual_genomlist
        #print("現世代で最も優れた個体は{}".format(elite_genes[0].getGenom()))
        #min_geomap_group.append(elite_genes[0].getGenom())
        #print("現時点で各世代の猛者は{}".format(min_geomap_group))

    # 最終結果出力
    
    #print("各世代の最小値は{}".format(min_group))
    #print("各世代の最大値は{}".format(max_group))
    #print("各世代の平均値は{}".format(avg_group))
    #print("最も優れた個体は{}".format(elite_genes[0].getGenom()))
    #print("各世代の猛者は{}".format(min_geomap_group))


# In[ ]:




