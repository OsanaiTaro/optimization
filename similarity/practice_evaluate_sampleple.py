import cv2
import numpy as np
import scipy.ndimage
from pandas import DataFrame, concat
from math import sin, cos, radians
from decimal import Decimal, ROUND_HALF_UP
import pareto_GA as ga
import practice_evaluate_sampleple as es

class evaluate_sample:
    def __init__(self, **kwargs):
        self.robot_path_1 = kwargs.get('robot_map','../../map_evaluate/debug/38_robot_path_1.png')
        self.robot_path_2 = kwargs.get('robot_map','../../map_evaluate/debug/38_robot_path_2.png')
        self.robot_path_3 = kwargs.get('robot_map','../../map_evaluate/debug/38_robot_path_3.png')
        self.geomap_path = kwargs.get('geomap','../../map_evaluate/debug/taromap.png')#実際に検証をするマップ
        
        self.urg_range = kwargs.get('urg_pix',400)#Lidarの測定範囲
        self.urg_resolution = kwargs.get('resolution', 1)#角度分解能
        
        self.rotation = kwargs.get('rotaion', 180)#回転角度
        
        self.deg_list = np.arange(0.0, 181.0, self.urg_resolution)#角度分解能のlist
        self.range_list = None
        self.range_list_rotation = None
                
        self.x_fin = []#LiDARの最端座標
        self.y_fin = []
        self.x_fin_rotation = []
        self.y_fin_rotation = []
        
        self.x_object = []#LiDARの分解能ごとの最近傍座標
        self.y_object = []
        self.x_object_rotation = []
        self.y_object_rotation = []
        
        self.robot_map = None#初期位置だけの画像配列
        self.object_map = None
        self.distance_map = None#測位可能距離だけの画像配列

        self.robot_points = None
        
        self.debug_map = None
        self.original_geomap = None
        
        self.comparison_list = None
        self.comparison_list_rotation = None
        
        self.delta_list = None
        
        self.match_count = None

        self.initial_flag = False
        self.initial_flag_rotation = False
        
        self.path_dist = None
        
    def map_from_img(self):#u方向に進めるときに軌跡画像からスタート位置とその個数、画像からロボットの座標が入る
        img_robo_path_1 = cv2.imread(self.robot_path_1)#画像open
        robot_map_1 = cv2.cvtColor(img_robo_path_1, cv2.COLOR_BGR2GRAY)#path1mapのグリッド毎の色情報255,255,・・・
        #print(robot_map_1)
        img_robo_path_2 = cv2.imread(self.robot_path_2)#画像open
        robot_map_2 = cv2.cvtColor(img_robo_path_2, cv2.COLOR_BGR2GRAY)
        img_robo_path_3 = cv2.imread(self.robot_path_3)#画像open
        robot_map_3 = cv2.cvtColor(img_robo_path_3, cv2.COLOR_BGR2GRAY)
        
        self.robot_points_1 = np.column_stack(np.where(robot_map_1 < 5))#黒いところの座標情報u,vが逆
        #print(self.robot_points_1)
        self.robot_points_2 = np.column_stack(np.where(robot_map_2 < 5))
        self.robot_points_3 = np.column_stack(np.where(robot_map_3 < 5))#grayscaleにした時に5未満の座標配列(????kita)
        
        self.robot_points = np.concatenate([self.robot_points_1, self.robot_points_2, self.robot_points_3])#ロボットポイントの配列を連結
        #print(self.robot_points)
        self.robot_map = robot_map_1
        
        img_geomap_path = cv2.imread(self.geomap_path)#画像open
        self.original_geomap = cv2.cvtColor(img_geomap_path, cv2.COLOR_BGR2GRAY)#geomapから取った元画像
        self.object_map = np.where(self.original_geomap < 5, 1, 0)#self.original_geomapが5より小さければ1大きければ0を返す,黒いところを1白いところを0と表示したmap_osanai
        self.path_dist = int((len(self.robot_points))/3)
        #self.path_dist = self.robot_points[-1][1] - (self.robot_points[0][1]-1)#u座標の最初から最後の個数,
        
        print("length = %d, grid = %d" % (len(self.robot_points),self.path_dist))
        return len(self.robot_points)
       
    def map_init_u(self, current_num=0):#基準となるロボットから距離情報を確保、
        self.robot_map[:,:] = 0#初期化
        
        self.robot_map[self.robot_points[current_num][0], self.robot_points[current_num][1]] = 155#start座標の色を１に,path3本分を灰色に
        self.robot_map[self.robot_map != 155] = 0#start座標以外は0に,path3本以外を黒に
        
        #print(self.robot_points[current_num][0])
    
        self.distance_map = scipy.ndimage.morphology.distance_transform_edt(self.robot_map==0)
        #各黒グリッドから各灰グリッドまでの距離、0~1499 & 0~849 & 2193(経路上の全グリッド数)個の数値を三次元リストで格納_osanai
        
        #print(sample.distance_map)
        
    def urg_scan(self, current_num):
        self.x_fin[:] = []#Lidarそれぞれの最端座標??
        self.y_fin[:] = []
        original_x = self.robot_points[current_num][1]#u座標　基準点
        original_y = self.robot_points[current_num][0]#v座標

        for r in self.deg_list:#角度分解能でループ
            x_ = self.urg_range * cos(radians(90-r))
            y_ = self.urg_range * sin(radians(90-r))
            x_ = Decimal(str(x_)).quantize(Decimal('0'), rounding=ROUND_HALF_UP)#少数第一位で四捨五入
            y_ = Decimal(str(y_)).quantize(Decimal('0'), rounding=ROUND_HALF_UP)
            x  = original_x + x_
            y  = original_y - y_#v座標からyへの変換でマイナス
            self.x_fin.append(int(x))#基準点でのLiDARの測定範囲のx座標
            self.y_fin.append(int(y))
    
    def bresenham(self, current_num):
        self.x_object = []#total_num番目の基準点におけるLiDARが当たっている物体座標が格納される
        self.y_object = []
        self.range_list = []
        
        for i in range(len(self.deg_list)):
            original_x = self.robot_points[current_num][1]#u座標　基準点
            original_y = self.robot_points[current_num][0]#v座標

            x = self.x_fin[i]
            y = self.y_fin[i]

            step = True if abs(y - original_y) > abs(x - original_x) else False #直線が急か、ゆるやかか_osanai
            #print(step)

            if step:#xとyを反転(直線が急ならばゆるやかに)_osanai
                original_x, original_y = original_y, original_x 
                x, y = y, x

            delta_x = abs(x - original_x) 
            delta_y = abs(y - original_y) 
            error = int(delta_x / 2) 
            write_y = original_y 

            inc = 1 if original_x < x else -1 #x座標の増分　基準点から見て右か左か_osanai
            y_step = 1 if original_y < y else -1 #y座標の増分　基準点から見て上か下か_osanai

            for write_x in range(original_x, x+inc, inc):#基準点からLiDARの最端まで1か-1ずつ増加させる_osanai
                if step:
                    if self.object_map[write_x][write_y] == 1:#障害物がある場合_osanai
                        d = self.distance_map[write_x][write_y]#LiDARの位置から障害物までの距離をdとする_osanai
                        self.range_list.append(d)#距離情報をlistに追加
                        self.x_object.append(write_y)#xとyを反転させてたから戻す_osanai
                        self.y_object.append(write_x)
                        break
                    if write_x == x and write_y == y:#LiDARの測定範囲限界まで来たら物体距離情報0とする_osanai
                        self.range_list.append(0.0)
                        self.x_object.append(y)
                        self.y_object.append(x)                
                else:#object_mapもdistance_mapもu,vが逆だからx,yも逆?_osanai
                    if self.object_map[write_y][write_x] == 1:
                        d = self.distance_map[write_y][write_x]
                        self.range_list.append(d)
                        self.x_object.append(write_x)
                        self.y_object.append(write_y)
                        break
                    if write_x == x and write_y == y:
                        self.range_list.append(0.0)
                        self.x_object.append(x)
                        self.y_object.append(y) 
                error = error - delta_y
                if error < 0:#右上に進む(でなければ右に進む)
                    write_y = write_y + y_step
                    error = error + delta_x
        
    def feature_df(self, current_num=0, total_num=0):
        if self.initial_flag == False:
            self.comparison_list = DataFrame(data=self.range_list, index=self.deg_list)#行でLiDARの照射角度、列で経路上の基準点を指定する、物体距離情報のデータフレーム_osanai
            self.initial_flag = True
        else:
            update_list = DataFrame(data=self.range_list, index=self.deg_list, columns=[current_num])
            self.comparison_list = concat([self.comparison_list, update_list], axis=1, sort=True)#もともとのやつと結合させる
        comparison_bool = (self.comparison_list.iloc[:-1,:total_num] > 0.0)#ローカル変数　最後の行、2193列までdがあればTrue、0ならばFalseを格納_osanai
        self.comparison_list.loc['count'] = comparison_bool.sum()#それぞれのポイントでの物体座標数　データフレームにcount行を追加？？ Trueの数を数えている？？_osanai
        #print(comparison_bool)
        
    def Urg_similarity(self, current_num, total_num):
        self.delta_list = self.comparison_list.copy()
        self.delta_list.drop('count', axis=0, inplace=True)#count行を削除_osanai
        for i in range(total_num):
            self.delta_list.iloc[:,i] = np.abs((self.comparison_list.iloc[:,i] - self.comparison_list[current_num]))#基準としたものとの差分　後ろの項が基準_osanai
        comparison_bool = (self.delta_list.iloc[:,:total_num] < 4.0)#基準との差が4？？より小さいところ(物体距離情報がマッチしているところ)_osanai
        comparison_bool2 = (self.comparison_list.iloc[:-1,:total_num] > 0.0)#物体距離情報があるところ(d=0ではないところ)_osanai
        comparison_bool3 = comparison_bool & comparison_bool2
        self.delta_list.loc['match'] = comparison_bool3.sum()#マッチしてる数_osanai

        
        match_count = self.delta_list.loc['match']
        ori_count = self.comparison_list.loc['count']
        rate = np.divide(match_count, ori_count, out=np.zeros_like(match_count), where=ori_count!=0)#rate　Nzの元の類似度の配列、グラフの横一線_osanai
        return rate

#new
    def urg_scan_rotation(self, current_num):
        self.x_fin_rotation[:] = []#Lidarそれぞれの最端座標
        self.y_fin_rotation[:] = []
        original_x = self.robot_points[current_num][1]#u座標　基準点
        original_y = self.robot_points[current_num][0]#v座標

        for r in self.deg_list:
            x_ = self.urg_range * cos(radians(90-(r-self.rotation)))#180°回転！！_osanai
            y_ = self.urg_range * sin(radians(90-(r-self.rotation)))#180°回転！！_osanai
            x_ = Decimal(str(x_)).quantize(Decimal('0'), rounding=ROUND_HALF_UP)#四捨五入
            y_ = Decimal(str(y_)).quantize(Decimal('0'), rounding=ROUND_HALF_UP)
            x  = original_x + x_
            y  = original_y - y_
            self.x_fin_rotation.append(int(x))
            self.y_fin_rotation.append(int(y))
       
    def bresenham_rotation(self, current_num):
        self.x_object_rotation = []#total_num番目の基準点におけるLiDARが当たっている物体座標が格納される
        self.y_object_rotation = []
        self.range_list_rotation = []
        
        for i in range(len(self.deg_list)):
            original_x = self.robot_points[current_num][1]#u座標　基準点
            original_y = self.robot_points[current_num][0]#v座標

            x = self.x_fin_rotation[i]
            y = self.y_fin_rotation[i]

            step = True if abs(y - original_y) > abs(x - original_x) else False 
            #print(step)

            if step:
                original_x, original_y = original_y, original_x 
                x, y = y, x

            delta_x = abs(x - original_x) 
            delta_y = abs(y - original_y) 
            error = int(delta_x / 2) 
            write_y = original_y 

            inc = 1 if original_x < x else -1 
            y_step = 1 if original_y < y else -1 

            for write_x in range(original_x, x+inc, inc):
                if step:
                    if self.object_map[write_x][write_y] == 1:
                        d = self.distance_map[write_x][write_y]
                        self.range_list_rotation.append(d)#距離情報をlistに追加
                        self.x_object_rotation.append(write_y)
                        self.y_object_rotation.append(write_x)
                        break
                    if write_x == x and write_y == y:
                        self.range_list_rotation.append(0.0)
                        self.x_object_rotation.append(y)
                        self.y_object_rotation.append(x)
                else:
                    if self.object_map[write_y][write_x] == 1:
                        d = self.distance_map[write_y][write_x]
                        self.range_list_rotation.append(d)
                        self.x_object_rotation.append(write_x)
                        self.y_object_rotation.append(write_y)
                        break
                    if write_x == x and write_y == y:
                        self.range_list_rotation.append(0.0)
                        self.x_object_rotation.append(x)
                        self.y_object_rotation.append(y) 
                error = error - delta_y
                if error < 0:
                    write_y = write_y + y_step
                    error = error + delta_x
    
    def feature_df_rotation(self, current_num=0, total_num=0):
        if self.initial_flag_rotation == False:
            self.comparison_list_rotation = DataFrame(data=self.range_list_rotation, index=self.deg_list)
            self.initial_flag_rotation = True
        else:
            update_list = DataFrame(data=self.range_list_rotation, index=self.deg_list, columns=[current_num])
            self.comparison_list_rotation = concat([self.comparison_list_rotation, update_list], axis=1, sort=True)#もともとのやつと結合させる
        comparison_bool = (self.comparison_list_rotation.iloc[:-1,:total_num] > 0.0)
        self.comparison_list_rotation.loc['count'] = comparison_bool.sum()#それぞれのポイントでの物体座標数
    
    def Urg_similarity_rotation(self, current_num, total_num):
        self.delta_list = self.comparison_list.copy()
        self.delta_list.drop('count', axis=0, inplace=True)
        for i in range(total_num):
            self.delta_list.iloc[:,i] = np.abs((self.comparison_list.iloc[:,i] - self.comparison_list_rotation[current_num]))#基準としたものとの差分 0度と180度を比較_osanai
        comparison_bool = (self.delta_list.iloc[:,:total_num] < 4.0)
        #self.delta_list.loc['match'] = comparison_bool.sum() 
        comparison_bool2 = (self.comparison_list_rotation.iloc[:-1,:total_num] > 0.0)
        comparison_bool3 = comparison_bool & comparison_bool2
        self.delta_list.loc['match'] = comparison_bool3.sum()
      
        match_count = self.delta_list.loc['match']
        ori_count = self.comparison_list.loc['count']
        rate_rotation = np.divide(match_count, ori_count, out=np.zeros_like(match_count), where=ori_count!=0)
        return rate_rotation
    
#追加！！！！_osanai

    def Urg_similarity_rotation_both(self, current_num, total_num):
        self.delta_list_rotation = self.comparison_list_rotation.copy()
        self.delta_list_rotation.drop('count', axis=0, inplace=True)
        for i in range(total_num):
            self.delta_list_rotation.iloc[:,i] = np.abs((self.comparison_list_rotation.iloc[:,i] - self.comparison_list_rotation[current_num]))
        comparison_bool = (self.delta_list_rotation.iloc[:,:total_num] < 4.0)
        #self.delta_list.loc['match'] = comparison_bool.sum() 
        comparison_bool2 = (self.comparison_list_rotation.iloc[:-1,:total_num] > 0.0)
        comparison_bool3 = comparison_bool & comparison_bool2
        self.delta_list_rotation.loc['match'] = comparison_bool3.sum()
      
        match_count = self.delta_list_rotation.loc['match']
        ori_count = self.comparison_list_rotation.loc['count']
        rate_rotation_both = np.divide(match_count, ori_count, out=np.zeros_like(match_count), where=ori_count!=0)
        return rate_rotation_both
