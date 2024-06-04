import math
import random
import numpy as np
import matplotlib
import datetime

N = 1000 # エージェントの数
size = 1000 # 仮想空間のサイズ
max_step = 144 # 1日当たりのステップ数
max_day = 60 # シミュレーション期間日数
SEED = 6551245 # 乱数の初期化
r = 1 # 感染範囲

professions = ["worker", "wife", "student"]   # 職業のパラメータはこの中からランダムに選ばれる

company_coord = [[(size/5)*1.5, (size/5)*4.5], [(size/5)*3.5, (size/5)*0.5]]   # 施設座標(会社)
store_coord =  [[(size/5)*1.5, (size/5)*2.5], [(size/5)*3.5, (size/5)*2.5]]    # 施設座標(お店)
school_coord = [[(size/5)*1.5, (size/5)*0.5], [(size/5)*3.5, (size/5)*4.5]]    # 施設座標(学校)

speed = 50      # 1stepあたりのエージェントの歩幅
infection_rate = 0.0005     # 感染確率
EtoIR_periods = [288, 576, 864]     # EIR状態遷移期間 [2日, 4日, 6日]
ItoRD_periods = [1440, 1728, 2016]  # IRD状態遷移期間 [10日, 12日, 14日]

mortality_rate = 0.1    # 致死率

control_day = 10             # 外出自粛発令日
control_rate_value = 0.80    # 外出確率減少値(外出自粛)

control_flag = False    # 外出自粛要請を発令したか

now_time = datetime.datetime(2021,1,1)  # 現在時刻


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
        self.state = state  # 感染状態

        self.step = 0       # ステップ

        self.inf_prob = random.random()    # 発症するかどうか

        self.home_coord = np.array([random.randint(1, size), random.randint(1, size)]) # 自宅座標
        self.current_coord = np.array([self.home_coord[0], self.home_coord[1]]) # 現在座標

        self.profession = random.choice(professions)   # 職業

        #職業によって決定
        if self.profession == "worker":
            t = random.randint(0,1)
            self.obj_coord = np.array([company_coord[t][0], company_coord[t][1]]) # 会社の座標
            self.go_out_prob = np.random.uniform(0.99, 1.00)  # 外出確率(会社員) 0.99～1.00の一様乱数
            self.go_out_time = np.random.normal(480, 60)      # 外出時間(会社員) 平均8時(480分),標準偏差1時間(60分)
            self.stay_time = np.random.randint(360, 480)      # 滞在時間(会社員) 360分~480分
        elif self.profession == "wife":
            t = random.randint(0,1)
            self.obj_coord = np.array([store_coord[t][0], store_coord[t][1]]) # お店の座標
            self.go_out_prob = np.random.uniform(0.50, 1.00)  # 外出確率(主婦) 0.50～1.00の一様乱数
            self.go_out_time = np.random.normal(600, 60)      # 外出時間(主婦) 平均10時(600分),標準偏差1時間(60分)
            self.stay_time = np.random.randint(10, 30)        # 滞在時間(主婦) 10分~30分
        elif self.profession == "student":
            t = random.randint(0,1)
            self.obj_coord = np.array([school_coord[t][0], school_coord[t][1]]) #a3 学校の座標
            self.go_out_prob = np.random.uniform(0.99, 1.00)  #a5 外出確率(学生) 0.99～1.00の一様乱数
            self.go_out_time = np.random.normal(480, 60)         #a7 外出時間(学生) 平均8時(480分),標準偏差1時間(60分)
            self.stay_time = np.random.randint(300, 360)      #a8 滞在時間(学生) 300分~360分

        self.go_out_flag = 1       # 活動自粛の有無(外出フラグ)
        self.go_obj_flag = False   # 目的地に移動中かどうかのフラグ
        self.go_home_flag = False  # 帰宅中かどうかのフラグ
        self.stay_home_flag = True # 自宅にいるかどうかのフラグ
        self.stay_obj_flag = False # 目的に滞在しているかどうかのフラグ
        self.temp_flag = False

        self.staying_time = 0      # 滞在経過時間

        self.EtoI_period = random.choice(EtoIR_periods)
        self.ItoRD_period = random.choice(ItoRD_periods)

        self.go_vx = math.cos(getRadian(self.home_coord[0], self.home_coord[1], self.obj_coord[0], self.obj_coord[1])) * speed #行きのx速度
        self.go_vy = math.sin(getRadian(self.home_coord[0], self.home_coord[1], self.obj_coord[0], self.obj_coord[1])) * speed #行きのy速度
        self.re_vx = math.cos(getRadian(self.home_coord[0], self.home_coord[1], self.obj_coord[0], self.obj_coord[1]) + math.radians(180)) * speed #帰りのx速度
        self.re_vy = math.sin(getRadian(self.home_coord[0], self.home_coord[1], self.obj_coord[0], self.obj_coord[1]) + math.radians(180)) * speed #帰りのy速度

        self.term_E = 0 # 潜伏状態になってからの日数
        self.term_I = 0 # 発症状態になってからの日数

        self.mortality = random.random() #感染した場合生き残れるかどうかのID

    def _calcnext(self, agents):  # 次時刻の計算

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
            self._state_D()        # 状態D用の計算
        else:   # 合致するカテゴリがない
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
            
            if agents[i].state == "I":
                if getNorm(sx, sy, ax, ay) < r and random.random() <= infection_rate: # 指定した範囲内に状態Iがいる場合
                    self.state = "E"  # 状態Eに変換（感染）
                    self.term_E += 1  # 感染ステップ数に1を追加する
                    break
    
    def _state_E(self, agents): # 状態Eの計算メソッド
        self.term_E += 1        # 経過ステップ数を追加
        if self.inf_prob < 0.90:
            if self.term_E > self.EtoI_period: # 一定の潜伏日数経過すると発症状態に遷移
                self.state = "I"
        
        if self.term_E > self.ItoRD_period:    # 10%の人は発症せず無症状として一定日数経過後回復
            self.state = "R"

    def _state_I(self, agents):  # 状態Iの計算メソッド
        self.term_I += 1  # 経過ステップ数を追加

        if self.term_I > self.ItoRD_period:
            if self.hos_flag != True:
                if self.term_I > self.ItoRD_period and self.mortality < mortality_rate: 
                    self.state = "D"   # 一定確率で死亡する
                else:
                    self.state = "R"   # そうでなければ免疫を獲得する

    def _state_R(self, agents):  # 状態Rの計算メソッド
        pass
    
    def _state_D(self): # 状態Dの計算メソッド
        pass
# agentクラスの定義終了


#次時刻の状態を計算
def calcn(agents):
    for i in range(len(agents)):
        agents[i]._calcnext(agents)

#外出自粛発令
def control_go_out(agents):
    for i in range(len(agents)):
        agents[i].go_out_prob -= control_rate_value

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

    # テキストファイルに出力
    fileobj = open("control_day10.txt", "a", encoding = "utf_8")
    fileobj.write("Day:{0} 非感染:{1} 潜伏:{2} 発症:{3} 回復:{4} 死亡:{5}\n".format(day+1, num_S, num_E, num_I, num_R, num_D))
    fileobj.close()

def simu_test():

    global now_time
    for day in range(max_day):
        for step in range(max_step):
            print("Day:{0} Time:{1:%H:%M:%S}".format(day+1, now_time))

            # 1日の最初のときのみ1day処理を実行
            if now_time.hour == 0 and now_time.minute == 0:
                print("1day処理実行")
                proc_day(day)
            
            # シミュレーション(1step処理)実行
            simulation()

            # 1日の最後のときのみ集計処理を実行
            if now_time.hour == 23 and now_time.minute == 50:
                print("集計実行")
                toTally(day)

            now_time += datetime.timedelta(minutes=10)

# シミュレーション実行
simu_test()