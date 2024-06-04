import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import datetime

N = 1500 # エージェントの個数
size = 800 # 仮想空間のサイズ
max_step = 144 # シミュレーションの打ち切りステップ
max_day = 3 #e1 シミュレーション期間日数
SEED = 655122 # 乱数の初期化
r = 5 # 感染範囲


professions = ["worker", "wife", "student"]      #a4職業のパラメータはこの中からランダムに選ばれる

company_coord = [[(size/5)*1.5, (size/5)*4.5], [(size/5)*3.5, (size/5)*0.5]]    #e4 施設座標(会社)
store_coord =  [[(size/5)*1.5, (size/5)*2.5], [(size/5)*3.5, (size/5)*2.5]]    #e5 施設座標(お店)
school_coord = [[(size/5)*1.5, (size/5)*0.5], [(size/5)*3.5, (size/5)*4.5]]     #e6 施設座標(学校)

speed = 40   #e7 1stepあたりのエージェントの歩幅
infection_rate = 0.01 # e8 感染確率
EtoI_periods = [288, 576, 864]     #e9 EI状態遷移期間
ItoRD_periods = [288, 576, 864]    #e10 IRD状態遷移期間

vis_cap_prob = 0.60 #e11 来院確率
used_bed = 0 #e12 使用病床数
max_bed = 20 #e13 最大病床数
num_rej = 0 #受け入れ拒否人数

mortality_rate_hos = 0.01 #e14 致死率(入院時)
mortality_rate = 0.1 #e15 致死率(非入院時)

control_rate_inf = 0.30 #e16 外出確率減少値(感染)
control_day = 50 #e17 外出自粛発令日
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
        self.agent_time = now_time #エージェント個人の時間

        self.home_coord = np.array([random.randint(1, size), random.randint(1, size)]) #a1 自宅座標
        self.current_coord = np.array([self.home_coord[0], self.home_coord[1]]) #現在座標

        self.profession = random.choice(professions)   #a4 職業
        #職業によって決定
        if self.profession == "worker":
            t = random.randint(0,1)
            self.obj_coord = np.array([company_coord[t][0], company_coord[t][1]]) #a3 会社の座標
            self.go_out_prob = np.random.uniform(0.99, 1.00)  #a5 外出確率(会社員) 0.99～1.00の一様乱数
            self.go_out_time = np.random.normal(480, 60)         #a7 外出時間(会社員) 平均8時(480分),標準偏差1時間(60分)
            self.stay_time = np.random.randint(240, 480)      #a8 滞在時間(会社員) 240分~480分
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
        self.temp_day = 0

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

        self.infection = random.random() #条件を満たした時,感染するかどうかのID
        self.mortality = random.random() #感染した場合生き残れるかどうかのID

    def _calcnext(self, agents):  # 次時刻の計算
        self.agent_time += datetime.timedelta(minutes=10)

        if self.state == "S":
            self._state_S(agents)  # 状態S用の計算
        elif self.state == "E":
            self._state_E(agents)  # 状態E用の計算
        elif self.state == "I":
            self._state_I(agents)  # 状態I用の計算
        elif self.state == "R":
            self._state_R()  # 状態R用の計算
        elif self.state == "D":
            self._state_D()  # 状態D用の計算
        else:  # 合致するカテゴリがない
            print("ERROR カテゴリがありません")

    def decide_action(self):
        if self.stay_home_flag and self.go_obj_flag == False and self.stay_obj_flag == False :
            #print("自宅滞在中")
            #print("temp = {0}".format(self.temp))
            #print("agent_time.day = {0}".format(self.agent_time.day))
            #外出時間を超えたら外出開始
            if self.go_out_flag == 1 and self.agent_time.day > self.temp_day and (self.agent_time.hour)*60 + (self.agent_time.minute) >= self.go_out_time :
                self.stay_home_flag = False
                self.go_obj_flag = True
                #print("外出開始")

        if self.go_obj_flag :
            #print("目的地に移動中")
            self._update_coord_obj()
            #print(getNorm(self.current_coord[0], self.current_coord[1], self.obj_coord[0], self.obj_coord[1]))
            if getNorm(self.current_coord[0], self.current_coord[1], self.obj_coord[0], self.obj_coord[1]) <= 40 :
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
            if getNorm(self.current_coord[0], self.current_coord[1], self.home_coord[0], self.home_coord[1]) <= 40 :
                self.stay_home_flag = True
                self.go_home_flag = False
                self.current_coord[0] = self.home_coord[0]
                self.current_coord[1] = self.home_coord[1]
                #print("帰宅完了")
                self.temp_day = self.agent_time.day


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
            if agents[i].state == "E" or agents[i].state == "I":
                if (sx-ax)*(sx-ax) + (sy-ay)*(sy-ay) < aR*aR and self.infection <= infection_rate: # 指定した範囲内に状態Eか状態Iがいる場合
                    self.state = "E"  # 状態Eに変換（感染）
                    self.term_E += 1  # 感染日数に1を追加する
                    break
            
        # 行動を決定
        self.decide_action()
    
    def _state_E(self, agents): # 状態Eの計算メソッド
        self.term_E += 1 # 感染日数を追加
        if self.term_E > self.EtoI_period: #一定の潜伏日数経過すると発症状態に遷移
            self.state = "I"

        sx = self.current_coord[0]  # 状態Sのx座標
        sy = self.current_coord[1]  # 状態Sのy座標
        for i in range(len(agents)):
            ax = agents[i].current_coord[0]  # 抽出したstate1のx座標
            ay = agents[i].current_coord[1]  # 抽出したstate1のx座標

            if agents[i].mask_f == 1:  # マスクの有無による感染範囲の設定
                aR = r/4
            else:
                aR = r
            if agents[i].state == "S":
                if (sx-ax)*(sx-ax) + (sy-ay)*(sy-ay) < aR*aR and agents[i].infection <= infection_rate: # 指定した範囲内に状態Eか状態Iがいる場合
                    agents[i].state = "E"  # 状態Eに変換（感染）
                    agents[i].term_E += 1  # 感染日数に1を追加する

        #行動を決定
        self.decide_action()

    def _state_I(self, agents):  # 状態Iの計算メソッド
        self.term_I += 1  # 感染日数を追加

        sx = self.current_coord[0]  # 状態Sのx座標
        sy = self.current_coord[1]  # 状態Sのy座標
        for i in range(len(agents)):
            ax = agents[i].current_coord[0]  # 抽出したstate1のx座標
            ay = agents[i].current_coord[1]  # 抽出したstate1のx座標

            if agents[i].mask_f == 1:  # マスクの有無による感染範囲の設定
                aR = r/4
            else:
                aR = r
            if agents[i].state == "S":
                if (sx-ax)*(sx-ax) + (sy-ay)*(sy-ay) < aR*aR and agents[i].infection <= infection_rate: # 指定した範囲内に状態Eか状態Iがいる場合
                    agents[i].state = "E"  # 状態Eに変換（感染）
                    agents[i].term_E += 1  # 感染日数に1を追加する

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

        #行動を決定
        self.decide_action()
    
    def _state_R(self):  # 状態Rの計算メソッド
        # 行動を決定
        self.decide_action()
    
    def _state_D(self):
        pass

# agentクラスの定義終わり


def calcn(agents):
    """次時刻の状態を計算"""
    #状態Sのデータ
    xlistS, ylistS = [], []
    #状態Eのデータ
    xlistE, ylistE = [], []
    #状態Iのデータ
    xlistI, ylistI = [], []
    #状態Rのデータ
    xlistR, ylistR = [], []
    #状態Dのデータ
    xlistD, ylistD = [], []

    for i in range(len(agents)):
        agents[i]._calcnext(agents)
        # a[i].putstate()
        # グラフデータに現在位置を追加
        if agents[i].state == "S":
            xlistS.append(agents[i].current_coord[0])
            ylistS.append(agents[i].current_coord[1])
        elif agents[i].state == "E":
            xlistE.append(agents[i].current_coord[0])
            ylistE.append(agents[i].current_coord[1])
        elif agents[i].state == "I":
            xlistI.append(agents[i].current_coord[0])
            ylistI.append(agents[i].current_coord[1])
        elif agents[i].state == "R":
            xlistR.append(agents[i].current_coord[0])
            ylistR.append(agents[i].current_coord[1])
        elif agents[i].state == "D":
            xlistD.append(agents[i].current_coord[0])
            ylistD.append(agents[i].current_coord[1])
    
    return xlistS, ylistS, xlistE, ylistE, xlistI, ylistI, xlistR, ylistR, xlistD, ylistD
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
#random.seed(SEED) # 乱数の初期化

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


# グラフデータの初期化
T = []
# Statas数推移
statasS_sum= []
statasE_sum= []
statasI_sum= []
statasR_sum= []
statasD_sum= []

#描画するグラフの設定
fig = plt.figure(figsize=(12, 5))
axs = [fig.add_subplot(121), fig.add_subplot(122)]
#空のグラフが出てしまうのを回避
#plt.close()

flag_legend = True #凡例描画のフラグ

#1day処理
def proc_day(now_time):
    for i in range(len(agentsA)):
        """if agentsA[i].profession == "worker":
            agentsA[i].go_out_time = np.random.normal(480, 60)
        elif agentsA[i].profession == "wife":
            agentsA[i].profession == np.random.normal(600, 60)
        elif agentsA[i].profession == "student":
            agentsA[i].go_out_time = np.random.normal(480, 60)"""

        if agentsA[i].hos_flag != True:
            #agentsA[i].stay_home_flag = True
            #agentsA[i].go_home_flag = False
            agentsA[i].current_coord[0] = agentsA[i].home_coord[0]
            agentsA[i].current_coord[1] = agentsA[i].home_coord[1]

    #外出自粛判定
    """global control_flag
    if now_time.day > control_day and control_flag != True:
        print("外出自粛発令")
        control_go_out(agentsA)
        control_flag = True"""
    
    #入退院判定
    """if now_time.day != 1:
        control_hos(agentsA)"""

    #外出判定
    decide_go_or_stay(agentsA)
    
#シミュレーション
def simulation(step):
    T.append(step)
    
    #次時刻の状態を計算
    xlistS, ylistS, xlistE, ylistE, xlistI, ylistI, xlistR, ylistR, xlistD, ylistD = calcn(agentsA)

    #状態Sのプロット
    axs[0].plot(xlistS, ylistS, ".", markersize=7, label="Susceptible", color="b")
    #状態Eのプロット
    axs[0].plot(xlistE, ylistE, ".", markersize=7, label="Exposed", color="y")
    #状態Iのプロット
    axs[0].plot(xlistI, ylistI, ".", markersize=7, label="Infected", color="r")
    #状態Rのプロット
    axs[0].plot(xlistR, ylistR, ".", markersize=7, label="Recovered", color="g")
    #状態Dのプロット
    axs[0].plot(xlistD, ylistD, ".", markersize=7, label="Dead", color="k")

    #subplot2 : 推移図
    statasS_sum.append(len(xlistS))
    statasE_sum.append(len(xlistE))
    statasI_sum.append(len(xlistI))
    statasR_sum.append(len(xlistR))
    statasD_sum.append(len(xlistD))
    axs[1].stackplot(T, statasE_sum, statasI_sum, statasR_sum, statasS_sum, statasD_sum, colors=["y", "r", "g", "b", "k"], alpha=0.7)

    #fig.savefig("pic/img{0}.png".format(step))
    #print("{0}step".format(step+1))

num_S = 0
num_E = 0
num_I = 0
num_R = 0
num_D = 0

#状態集計
def toTally(now_time):
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
        
    print("非感染:{0}人".format(num_S))
    print("潜伏:{0}人".format(num_E))
    print("発症:{0}人".format(num_I))
    print("回復:{0}人".format(num_R))
    print("死亡:{0}人".format(num_D))

    fileobj = open("statistics.txt", "a", encoding = "utf_8")
    fileobj.write("Day:{0}".format(now_time.day))
    fileobj.write("非感染:{0}".format(num_S))
    fileobj.write("潜伏:{0}".format(num_E))
    fileobj.write("発症:{0}".format(num_I))
    fileobj.write("回復:{0}".format(num_R))
    fileobj.write("死亡:{0}\n".format(num_D))
    #fileobj.write("使用病床数:{0}".format(used_bed))
    #fileobj.write("受け入れ拒否数:{0}".format(num_rej))
    fileobj.close()


def update_ani(step, now_time):
    for ax in axs:
        ax.cla() # ax をクリア

    #axs[0].legend()
    axs[0].set_xlim(0, size)
    axs[0].set_ylim(0, size)

    #axs[0].tick_params(labelbottom=False,labelleft=False,labelright=False,labeltop=False, length=0)
    #axs[0].tick_params(length=0)

    #plt.text(100, 100, "Day:{0}".format((now_time + datetime.timedelta(minutes=10*step)).day), fontsize=10, horizontalalignment="center")
    axs[0].set_title("Day:{0:%d %H:%M:%S}".format(now_time + datetime.timedelta(minutes=10*step)))
    #axs[0].set_title("Day:{0:%d %H:%M:%S}".format(now_time))
    #print(step)
    now_time += datetime.timedelta(minutes=10*step)
    print(now_time)

    axs[0].grid()

    boxdic = {
    "facecolor" : "white",
    "edgecolor" : "black",
    "boxstyle" : "Round",
    "linewidth" : 1
    }

    axs[0].text(company_coord[0][0],company_coord[0][1],"Company", fontsize=10, bbox=boxdic, horizontalalignment="center")
    axs[0].text(store_coord[0][0],store_coord[0][1],"Store", fontsize=10, bbox=boxdic, horizontalalignment="center")
    axs[0].text(school_coord[0][0],school_coord[0][1],"School",fontsize=10, bbox=boxdic, horizontalalignment="center")
    axs[0].text(company_coord[1][0],company_coord[1][1],"Company", fontsize=10, bbox=boxdic, horizontalalignment="center")
    axs[0].text(store_coord[1][0],store_coord[1][1],"Store", fontsize=10, bbox=boxdic, horizontalalignment="center")
    axs[0].text(school_coord[1][0],school_coord[1][1],"School",fontsize=10, bbox=boxdic, horizontalalignment="center")
    
    #text_bed = "使用病床数:{0}".format(used_bed)
    #axs[0].text(100, 100, text_bed,fontsize=10 ,horizontalalignment="center")

    axs[1].set_xlim(0, max_step*max_day)

    axs[1].tick_params(labelbottom=True,labelleft=True,labelright=False,labeltop=False)
    #axs[1].axhline(over_capacity, ls = "--", color = "black")

    #axs[2].set_xlim(0, 100)
    #axs[2].set_ylim(0, 100)

    
    #1日の最初のときのみ処理を実行
    if now_time.hour == 0 and now_time.minute == 0:
        print("1day処理実行")
        proc_day(now_time)

    simulation(step)

    #1日の最後のときのみ処理を実行
    if now_time.hour == 23 and now_time.minute == 50:
        print("集計実行")
        toTally(now_time)
    
    axs[0].legend(loc='lower center', bbox_to_anchor=(1.05, 1.05), ncol=5, fontsize=10)

    #fig.savefig("pic3/img{0}.png".format(step+1))

    if step >= max_step:
        plt.close()


anim = FuncAnimation(fig, update_ani, fargs = (now_time,), frames=max_step*max_day, interval=80, blit=False, repeat=False)

anim.save('anim.gif', writer="pillow")