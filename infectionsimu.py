import math
import random
import numpy as np
import matplotlib
#import matplotlib.pyplot as plt
#from matplotlib.animation import FuncAnimation
import datetime

N = 1000 # エージェントの個数
size = 1000 # 仮想空間のサイズ
max_step = 144 # シミュレーションの打ち切りステップ
max_day = 100 #e1 シミュレーション期間日数
SEED = 6551245 # 乱数の初期化
r = 1 # 感染範囲


professions = ["worker", "wife", "student"]   #a4職業のパラメータはこの中からランダムに選ばれる

company_coord = [[(size/5)*1.5, (size/5)*4.5], [(size/5)*3.5, (size/5)*0.5]]    #e4 施設座標(会社)
store_coord =  [[(size/5)*1.5, (size/5)*2.5], [(size/5)*3.5, (size/5)*2.5]]    #e5 施設座標(お店)
school_coord = [[(size/5)*1.5, (size/5)*0.5], [(size/5)*3.5, (size/5)*4.5]]    #e6 施設座標(学校)

speed = 50   #e7 1stepあたりのエージェントの歩幅
infection_rate = 0.0005 # e8 感染確率
EtoI_periods = [288, 576, 864]     #e9 EI状態遷移期間 2-4-6
ItoRD_periods = [1440, 1728, 2016]    #e10 IRD状態遷移期間 10-12-14

vis_cap_prob = 0.60 #e11 来院確率
used_bed = 0 #e12 使用病床数
max_bed = 20 #e13 最大病床数
num_rej = 0 #受け入れ拒否人数

mortality_rate_hos = 0.01 #e14 致死率(入院時)
mortality_rate = 0.1 #e15 致死率(非入院時)

control_rate_inf = 0.30 #e16 外出確率減少値(感染)
control_day = 7 #e17 外出自粛発令日
control_rate_after = 0.80 #e18 外出確率減少値(外出自粛)

control_flag = False #自粛宣言したか

now_time = datetime.datetime(2021,1,1)  #e19 現在時刻

mask_rate = 0.0 # マスクを装着している割合
#over_capacity = N/2 # 感染拡大の目安


#自宅から目的地までの角度を算出
def getRadian(x1, y1, x2, y2):
    a = np.array([x1, y1])
    b = np.array([x2, y2])
    vec = b - a
    #print(math.degrees(np.arctan2(vec[1], vec[0])))
    #引数の順番がx, yではなくy, xなので注意
    return np.arctan2(vec[1], vec[0]) #ラジアンで返す
    
#ノルム算出
def getNorm(x1, y1, x2, y2):
    a = np.array([x1, y1])
    b = np.array([x2, y2])
    return np.linalg.norm(b - a)


#Agent class
class Agent:
    def __init__(self, state, now_time):
        self.state = state #a10 感染状態

        self.step = 0 # ステップ
        #self.agent_time = now_time #エージェント個人の時間

        self.inf_prob = random.random()  #発症するかどうか

        self.home_coord = np.array([random.randint(1, size), random.randint(1, size)]) #a1 自宅座標
        self.current_coord = np.array([self.home_coord[0], self.home_coord[1]]) #現在座標

        self.profession = random.choice(professions)   #a4 職業
        #職業によって決定
        if self.profession == "worker":
            t = random.randint(0,1)
            self.obj_coord = np.array([company_coord[t][0], company_coord[t][1]]) #a3 会社の座標
            self.go_out_prob = np.random.uniform(0.99, 1.00)  #a5 外出確率(会社員) 0.99～1.00の一様乱数
            self.go_out_time = np.random.normal(480, 60)         #a7 外出時間(会社員) 平均8時(480分),標準偏差1時間(60分)
            self.stay_time = np.random.randint(360, 480)      #a8 滞在時間(会社員) 360分~480分
        elif self.profession == "wife":
            t = random.randint(0,1)
            self.obj_coord = np.array([store_coord[t][0], store_coord[t][1]]) #a3 お店の座標
            self.go_out_prob = np.random.uniform(0.50, 1.00)  #a5 外出確率(主婦) 0.50～1.00の一様乱数
            self.go_out_time = np.random.normal(600, 60)        #a7 外出時間(主婦) 平均10時(600分),標準偏差1時間(60分)
            self.stay_time = np.random.randint(10, 30)        #a8 滞在時間(主婦) 10分~30分
        elif self.profession == "student":
            t = random.randint(0,1)
            self.obj_coord = np.array([school_coord[t][0], school_coord[t][1]]) #a3 学校の座標
            self.go_out_prob = np.random.uniform(0.99, 1.00)  #a5 外出確率(学生) 0.99～1.00の一様乱数
            self.go_out_time = np.random.normal(480, 60)         #a7 外出時間(学生) 平均8時(480分),標準偏差1時間(60分)
            self.stay_time = np.random.randint(300, 360)      #a8 滞在時間(学生) 300分~360分

        self.go_out_flag = 1    #a6 活動自粛の有無(外出フラグ)
        self.go_obj_flag = False   #目的地に移動中かどうかのフラグ
        self.go_home_flag = False  #帰宅中かどうかのフラグ
        self.stay_home_flag = True #自宅にいるかどうかのフラグ
        self.stay_obj_flag = False #目的に滞在しているかどうかのフラグ
        self.temp_flag = False

        self.staying_time = 0   #a9 滞在経過時間

        self.hos_flag = False #a13 入院フラグ
        self.hos_proc_flag = False #入退院判定を受けたかどうか
        self.jikkou = 0 #a14 実行再生産数


        self.EtoI_period = random.choice(EtoI_periods)
        self.ItoRD_period = random.choice(ItoRD_periods)

        self.go_vx = math.cos(getRadian(self.home_coord[0], self.home_coord[1], self.obj_coord[0], self.obj_coord[1])) * speed #行きのx速度
        self.go_vy = math.sin(getRadian(self.home_coord[0], self.home_coord[1], self.obj_coord[0], self.obj_coord[1])) * speed #行きのy速度
        self.re_vx = math.cos(getRadian(self.home_coord[0], self.home_coord[1], self.obj_coord[0], self.obj_coord[1]) + math.radians(180)) * speed #帰りのx速度
        self.re_vy = math.sin(getRadian(self.home_coord[0], self.home_coord[1], self.obj_coord[0], self.obj_coord[1]) + math.radians(180)) * speed #帰りのy速度

        self.term_E = 0 # 潜伏状態になってからの日数
        self.term_I = 0 # 発症状態になってからの日数
        self.mask_f = 0 # マスク着用の有無, 0はマスク無し, 1はマスクあり

        #self.infection = random.random() #条件を満たした時,感染するかどうかのID
        self.mortality = random.random() #感染した場合生き残れるかどうかのID

    def _calcnext(self, agents):  # 次時刻の計算
        #self.agent_time += datetime.timedelta(minutes=10)

        if self.state == "S":
            self.decide_action()   # 行動を決定
            self._state_S(agents)  # 状態S用の計算
        elif self.state == "E":
            self.decide_action()   # 行動を決定
            self._state_E(agents)  # 状態E用の計算
        elif self.state == "I":
            self.decide_action()   # 行動を決定
            self._state_I(agents)  # 状態I用の計算
        elif self.state == "R":
            self.decide_action()   # 行動を決定
            self._state_R(agents)  # 状態R用の計算
        elif self.state == "D":
            self._state_D()  # 状態D用の計算
        else:  # 合致するカテゴリがない
            print("ERROR カテゴリがありません")

    def decide_action(self):
        if self.stay_home_flag and self.go_obj_flag == False and self.stay_obj_flag == False :
            #print("自宅滞在中")

            #外出時間を超えたら外出開始
            if self.go_out_flag == 1 and self.temp_flag and (now_time.hour)*60 + (now_time.minute) >= self.go_out_time :
                self.stay_home_flag = False
                self.go_obj_flag = True
                self.temp_flag = False
                #print("外出開始")

        if self.go_obj_flag :
            #print("目的地に移動中")
            self._update_coord_obj()
            #print(getNorm(self.current_coord[0], self.current_coord[1], self.obj_coord[0], self.obj_coord[1]))
            if getNorm(self.current_coord[0], self.current_coord[1], self.obj_coord[0], self.obj_coord[1]) <= speed :
                self.go_obj_flag = False
                self.stay_obj_flag = True
                self.current_coord[0] = self.obj_coord[0]
                self.current_coord[1] = self.obj_coord[1]
                self.staying_time = 0
                #print("目的地到着")
        
        if self.stay_obj_flag :
            #print("目的地滞在中")
            if self.staying_time >= self.stay_time :
                self.stay_obj_flag = False
                self.go_home_flag = True
                #print("目的地滞在終了")
                #print("帰宅開始")
            self.staying_time += 10
        
        if self.go_home_flag :
            #print("帰宅中")
            self._update_coord_home()
            #print(getNorm(self.current_coord[0], self.current_coord[1], self.home_coord[0], self.home_coord[1]))
            if getNorm(self.current_coord[0], self.current_coord[1], self.home_coord[0], self.home_coord[1]) <= speed :
                self.stay_home_flag = True
                self.go_home_flag = False
                self.current_coord[0] = self.home_coord[0]
                self.current_coord[1] = self.home_coord[1]
                #print("帰宅完了")


    #自宅から目的地への移動処理
    def _update_coord_obj(self):
        self.current_coord[0] = self.current_coord[0] + self.go_vx * self.go_out_flag
        self.current_coord[1] = self.current_coord[1] + self.go_vy * self.go_out_flag
    
    #目的地から自宅への移動処理
    def _update_coord_home(self):
        self.current_coord[0] = self.current_coord[0] + self.re_vx * self.go_out_flag
        self.current_coord[1] = self.current_coord[1] + self.re_vy * self.go_out_flag

    
    def _state_S(self, agents):  # 状態Sの計算メソッド
        # すべてのエージェントとの距離を調べる
        sx = self.current_coord[0]  # 状態Sのx座標
        sy = self.current_coord[1]  # 状態Sのy座標
        for i in range(len(agents)):
            ax = agents[i].current_coord[0]  # 抽出したstate1のx座標
            ay = agents[i].current_coord[1]  # 抽出したstate1のx座標

            if agents[i].mask_f == 1:  # マスクの有無による感染範囲の設定
                aR = r/4
            else:
                aR = r
            
            if agents[i].state == "I":
                if getNorm(sx, sy, ax, ay) < aR and random.random() <= infection_rate: # 指定した範囲内に状態Eか状態Iがいる場合
                    self.state = "E"  # 状態Eに変換（感染）
                    self.term_E += 1  # 感染日数に1を追加する
                    break
    
    def _state_E(self, agents): # 状態Eの計算メソッド
        self.term_E += 1 # 経過ステップ数を追加
        if self.inf_prob < 0.90:
            if self.term_E > self.EtoI_period: #一定の潜伏日数経過すると発症状態に遷移
                self.state = "I"
        
        if self.term_E > self.ItoRD_period: #10%の人は発症せず無症状として一定日数経過後回復
            self.state = "R"

    def _state_I(self, agents):  # 状態Iの計算メソッド
        self.term_I += 1  # 経過ステップ数を追加

        if self.term_I > self.ItoRD_period:
            if self.hos_flag != True:
                if self.term_I > self.ItoRD_period and self.mortality < mortality_rate: # 一定確率で死亡する(非入院)
                    self.state = "D"
                else:
                    self.state = "R"    #そうでなければ免疫を獲得する
            else:
                if self.term_I > self.ItoRD_period and self.mortality < mortality_rate_hos: # 一定確率で死亡する(入院)
                    self.state = "D"
                else:
                    self.state = "R"

    def _state_R(self, agents):  # 状態Rの計算メソッド
        pass
    
    def _state_D(self): # 状態Dの計算メソッド
        pass

# agentクラスの定義終わり


def calcn(agents):
    """次時刻の状態を計算"""

    for i in range(len(agents)):
        agents[i]._calcnext(agents)
    
    # clacn()関数の終わり


#外出自粛発令
def control_go_out(agents):
    for i in range(len(agents)):
        agents[i].go_out_prob -= control_rate_after
    #control_flag = True

#入退院判定
def control_hos(agents):
    global used_bed
    global num_rej
    for i in range(len(agents)):
        #退院判定
        if agents[i].hos_flag and agents[i].state == 'R':
            agents[i].hos_flag = False
            agents[i].go_out_prob = np.random.uniform(0.50, 1.00)
            agents[i].current_coord[0] = agents[i].home_coord[0]
            agents[i].current_coord[1] = agents[i].home_coord[1]
            used_bed -= 1

        #入院判定
        if agents[i].hos_flag != True and agents[i].state == 'I' and agents[i].hos_proc_flag != True and random.random() < vis_cap_prob:
            #病床数に余裕があれば入院
            if(used_bed < max_bed):
                agents[i].hos_flag = True
                agents[i].go_out_prob = 0
                used_bed += 1
                agents[i].current_coord[0] = random.randint(-200, -100)
                agents[i].current_coord[1] = random.randint(0, size)
            else:
                num_rej += 1
                agents[i].go_out_prob -= control_rate_inf
            agents[i].hos_proc_flag = True
    
    print("使用病床数:{0}".format(used_bed))
    print("受け入れ拒否人数:{0}".format(num_rej))


#外出判定(その日外出するかどうか)
def decide_go_or_stay(agents):
    for i in range(len(agents)):
        if random.random() < agents[i].go_out_prob:
            agents[i].go_out_flag = 1
        else:
            agents[i].go_out_flag = 0


# 初期化
random.seed(SEED) # 乱数の初期化

# 状態SのエージェントをN個生成
agentsA = []
for i in range(N):
    agent = Agent("S", now_time)
    agentsA.append(agent)

# マスクを着用するエージェントの設定
"""for agent in random.sample(agentsA, int(mask_rate*N)):
    agent.mask_f = 1"""

# 初期状態Iのエージェントの設定
for i in range(10):
    agentsA[i].state = "I"


#1day処理
def proc_day(day):
    for i in range(len(agentsA)):
        agentsA[i].temp_flag = True

    #外出自粛判定
    global control_flag
    if day >= control_day and control_flag != True:
        print("外出自粛発令")
        control_go_out(agentsA)
        control_flag = True
    
    #入退院判定
    """if day != 0:
        control_hos(agentsA)"""

    #外出判定
    decide_go_or_stay(agentsA)
    
#シミュレーション
def simulation():
    
    #次時刻の状態を計算
    calcn(agentsA)


#状態集計
def toTally(day):
    num_S = 0
    num_E = 0
    num_I = 0
    num_R = 0
    num_D = 0

    for agent in agentsA:
        if agent.state == "S":
            num_S += 1
        elif agent.state == "E":
            num_E += 1
        elif agent.state == "I":
            num_I += 1
        elif agent.state == "R":
            num_R += 1
        elif agent.state == "D":
            num_D += 1
        
    print("Day:{0} 非感染:{1} 潜伏:{2} 発症:{3} 回復:{4} 死亡:{5}".format(day+1, num_S, num_E, num_I, num_R, num_D))

    fileobj = open("control_day7.txt", "a", encoding = "utf_8")
    fileobj.write("Day:{0} 非感染:{1} 潜伏:{2} 発症:{3} 回復:{4} 死亡:{5}\n".format(day+1, num_S, num_E, num_I, num_R, num_D))
    #fileobj.write("使用病床数:{0}".format(used_bed))
    #fileobj.write("受け入れ拒否数:{0}".format(num_rej))
    fileobj.close()

def simu_test():

    global now_time
    for day in range(max_day):
        for step in range(max_step):
            print("Day:{0} Time:{1:%H:%M:%S}".format(day+1, now_time))
            #1日の最初のときのみ処理を実行

            if now_time.hour == 0 and now_time.minute == 0:
                print("1day処理実行")
                proc_day(day)
            
            simulation()

            #1日の最後のときのみ処理を実行
            if now_time.hour == 23 and now_time.minute == 50:
                print("集計実行")
                toTally(day)

            now_time += datetime.timedelta(minutes=10)

simu_test()
