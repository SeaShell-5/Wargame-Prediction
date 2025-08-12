import numpy as np
import sys
sys.path.append("./..")
from util.util import cvtInt6loc2HexOffset, cvtHexOffset2Int6loc, to_categorical, from_categorical
import copy, math


bops_dic = ["red_tank1", "red_tank2", "red_car1", "red_car2", "red_soldier1", "red_soldier2",
             "blue_tank1", "blue_tank2", "blue_car1", "blue_car2", "blue_soldier1", "blue_soldier2"]
score_dic = ["red_shoot", "red_lost", "red_occupy", "red_total", 
             "blue_shoot", "blue_lost", "blue_occupy", "blue_total", 
             "red_win"]
act_dic = ["north_east",
           "east",
           "south_east",
            "south_west",
            "west",
            "north_west",
            "occupy",
            "on_car",
            "off_car",
            "to_pass",
            "to_move",
            "to_hide",
            "cancel_keep",
            "shoot"]

class GameState(object):
    def __init__(self, Filename, csv_object):
        # csv_object中的Filename是string类型
        
        self.csv_object = csv_object
        
        self.board = {"stage": 1,       # 游戏共有20个阶段，当前阶段为1到20, 城市状态初始化为0
                      "city": [0, 0, 0, 0]}  
        
        # 初始化每个作战单元（bop_name）的状态, 以字典形式储存。
        # {'stage': 1, 'city': [0, 0, 0, 0], 'red_tank1':{'ID': None, 'state': None, 'Pos': None},...}
        for bop_name in bops_dic:
            self.board[bop_name] = {"ID": None, "state": None, "Pos": None}
        
        # 初始化分数系统, {'red_shoot': 0, 'red_lost': 0, 'red_occupy': 0,...}    
        self.score = {score_name: 0 for score_name in score_dic}

        # 初始化行动系统, {'north_east': 0, 'east': 0, 'south_east': 0, 'south_west':...}
        self.act = {act_name: 0 for act_name in act_dic}
        
        # 初始化所有作战单位的行动记录, 每一个bops_dic中记录一个act_dic
        # {'red_tank1': {'north_east': 0, 'east': 0, 'south_east': 0,...}
        self.acts = {bop_name: copy.deepcopy(self.act) for bop_name in bops_dic}
        self.acts['no_act'] = 1
        

        # 从CSV对象中筛选出红色方（RED）的数据，并按ObjPos排序
        csv_object_red = csv_object.loc[
            (csv_object["Filename"] == Filename) & (csv_object["GameColor"] == "RED"), ['ID', 'ObjName', "ObjPos"]]
        
        csv_object_red = csv_object_red.sort_values(by="ObjPos")

        if csv_object_red.empty or len(csv_object_red) != 6:
            raise Exception('program error')
        
        # 遍历红方的数据，初始化作战单位
        for indexs, row in csv_object_red.iterrows():
            if row.ObjName == "坦克" or row.ObjName == "重型坦克":
                if not self.board["red_tank1"]["ID"]:
                    self.board["red_tank1"]["ID"] = row.ID
                    self.board["red_tank1"]["state"], self.board["red_tank1"]["Pos"] = self.get_bop_init(row.ObjPos)

                else:
                    self.board["red_tank2"]["ID"] = row.ID
                    self.board["red_tank2"]["state"], self.board["red_tank2"]["Pos"] = self.get_bop_init(row.ObjPos)
            if row.ObjName == "战车" or row.ObjName == "中型战车" or row.ObjName == "重型战车":
                if not self.board["red_car1"]["ID"]:
                    self.board["red_car1"]["ID"] = row.ID
                    self.board["red_car1"]["state"], self.board["red_car1"]["Pos"] = self.get_bop_init(row.ObjPos)
                else:
                    self.board["red_car2"]["ID"] = row.ID
                    self.board["red_car2"]["state"], self.board["red_car2"]["Pos"] = self.get_bop_init(row.ObjPos)
            if row.ObjName == "步兵小队":
                if not self.board["red_soldier1"]["ID"]:
                    self.board["red_soldier1"]["ID"] = row.ID
                    self.board["red_soldier1"]["state"], self.board["red_soldier1"]["Pos"] = self.get_bop_init(row.ObjPos)
                else:
                    self.board["red_soldier2"]["ID"] = row.ID
                    self.board["red_soldier2"]["state"], self.board["red_soldier2"]["Pos"] = self.get_bop_init(row.ObjPos)
        
        # 遍历蓝方的数据，初始化作战单位
        csv_object_blue = csv_object.loc[
            (csv_object["Filename"] == Filename) & (csv_object["GameColor"] == "BLUE"), ['ID', 'ObjName', "ObjPos"]]
        csv_object_blue = csv_object_blue.sort_values(by="ObjPos")
        for indexs, row in csv_object_blue.iterrows():
            if row.ObjName == "坦克" or row.ObjName == "重型坦克":
                if not self.board["blue_tank1"]["ID"]:
                    self.board["blue_tank1"]["ID"] = row.ID
                    self.board["blue_tank1"]["state"], self.board["blue_tank1"]["Pos"] = self.get_bop_init(row.ObjPos)
                else:
                    self.board["blue_tank2"]["ID"] = row.ID
                    self.board["blue_tank2"]["state"], self.board["blue_tank2"]["Pos"] = self.get_bop_init(row.ObjPos)
            if row.ObjName == "战车" or row.ObjName == "中型战车" or row.ObjName == "重型战车":
                if not self.board["blue_car1"]["ID"]:
                    self.board["blue_car1"]["ID"] = row.ID
                    self.board["blue_car1"]["state"], self.board["blue_car1"]["Pos"] = self.get_bop_init(row.ObjPos)
                else:
                    self.board["blue_car2"]["ID"] = row.ID
                    self.board["blue_car2"]["state"], self.board["blue_car2"]["Pos"] = self.get_bop_init(row.ObjPos)
            if row.ObjName == "步兵小队":
                if not self.board["blue_soldier1"]["ID"]:
                    self.board["blue_soldier1"]["ID"] = row.ID
                    self.board["blue_soldier1"]["state"], self.board["blue_soldier1"]["Pos"] = self.get_bop_init(row.ObjPos)
                else:
                    self.board["blue_soldier2"]["ID"] = row.ID
                    self.board["blue_soldier2"]["state"], self.board["blue_soldier2"]["Pos"] = self.get_bop_init(row.ObjPos)
    
    # 初始化作战单元状态和位置
    def get_bop_init(self, Pos):
        """
        living:存活;ObjPass:"人员1 1级疲劳,人员2 2级疲劳,车辆1 行军中";ObjHide:隐匿;
        ObjKeep:第几阶段压制;ObjTire:是否疲劳;ObjRound:是否已机动;ObjAttack:第几阶段射击过;
        ObjSon:是否载人;ObjInto:上车,0 无动作，>0 上哪辆车;ObjBlood:该棋子数量;ObjStep:机动值
        """
        bop_state = {}
        bop_state["living"] = 1
        bop_state["ObjPass"] = 0
        bop_state["ObjHide"] = 0
        bop_state["ObjKeep"] = 0
        bop_state["ObjTire"] = to_categorical(0, 3)  # E.g:(0, 3)array:[1., 0., 0.]
        bop_state["ObjRound"] = 0
        bop_state["ObjAttack"] = 0
        bop_state["ObjSon"] = 0
        bop_state["ObjInto"] = 0
        # bop_state["blood_value"] = 
        bop_state["ObjBlood"] = to_categorical(3, 4) # E.g:(3, 4)arrayarray:[0., 0., 0., 1.]
        bop_state["ObjStep"] = to_categorical(6, 8)  # E.g:(6, 8)arrayarray:[0., 0., 0., 0., 0., 0., 1., 0.]
        bop_Pos = cvtInt6loc2HexOffset(Pos)
        return bop_state, bop_Pos
    
    # 加载作战单元状态
    def load_bop_state(self, bop_name, row):
        self.board[bop_name]["state"]["living"] = 1
        self.board[bop_name]["state"]["ObjPass"] = 0 if row.ObjPass == 0 else 1
        self.board[bop_name]["state"]["ObjHide"] = 0 if row.ObjHide == 0 else 1
        self.board[bop_name]["state"]["ObjKeep"] = 0 if row.ObjKeep == 0 else 1
        if row.ObjTire <= 2:
            Tire = row.ObjTire
        else:
            Tire = 2
        self.board[bop_name]["state"]["ObjTire"] = to_categorical(Tire, 3)
        if row.ObjRound <= 1:
            Round = row.ObjRound
        else:
            Round = 1
        self.board[bop_name]["state"]["ObjRound"] = Round
        if row.ObjAttack <= 1:
            Attack = row.ObjAttack
        else:
            Attack = 1
        self.board[bop_name]["state"]["ObjAttack"] = Attack
        self.board[bop_name]["state"]["ObjSon"] = 0  if row.ObjSon == 0 else 1
        self.board[bop_name]["state"]["ObjInto"] = 0 if row.ObjInto == 0 else 1
        self.board[bop_name]["state"]["ObjBlood"] = to_categorical(row.ObjBlood, 4)
        self.board[bop_name]["state"]["ObjStep"] = to_categorical(row.ObjStep, 8)
        self.board[bop_name]["Pos"] = cvtInt6loc2HexOffset(row.ObjNewPos)
    
    # 设置作战单元死亡状态
    def get_bop_death(self):
        bop_state = {}
        bop_state["living"] = 0
        bop_state["ObjPass"] = 0
        bop_state["ObjHide"] = 0
        bop_state["ObjKeep"] = 0
        bop_state["ObjTire"] = np.zeros(3)
        bop_state["ObjRound"] = 0
        bop_state["ObjAttack"] = 0
        bop_state["ObjSon"] = 0
        bop_state["ObjInto"] = 0
        bop_state["ObjBlood"] = np.zeros(4)
        bop_state["ObjStep"] = np.zeros(8)
        bop_Pos = (0, 0)
        return bop_state, bop_Pos
    
    # 初始化行动记录
    def init_acts(self):
        for key1 in self.acts.keys():
            if key1 != 'no_act':
                for key2 in act_dic:
                    self.acts[key1][key2] = 0
            else:
                self.acts['no_act'] = 0

    def fill_no_act(self):
        for key1 in self.acts.keys():
            if key1 != 'no_act':
                for key2 in act_dic:
                    self.acts[key1][key2] = 0
            else:
                self.acts["no_act"] = 1

    def initCancelKeep(self):
        if self.board["stage"] % 2 == 0:
            self.board["red_tank1"]["state"]["ObjKeep"] = 0
            self.board["red_tank2"]["state"]["ObjKeep"] = 0
            self.board["red_car1"]["state"]["ObjKeep"] = 0
            self.board["red_car2"]["state"]["ObjKeep"] = 0
            self.board["red_soldier1"]["state"]["ObjKeep"] = 0
            self.board["red_soldier2"]["state"]["ObjKeep"] = 0
        else:
            self.board["blue_tank1"]["state"]["ObjKeep"] = 0
            self.board["blue_tank2"]["state"]["ObjKeep"] = 0
            self.board["blue_car1"]["state"]["ObjKeep"] = 0
            self.board["blue_car2"]["state"]["ObjKeep"] = 0
            self.board["blue_soldier1"]["state"]["ObjKeep"] = 0
            self.board["blue_soldier2"]["state"]["ObjKeep"] = 0

    def UpdateOn(self, row, bop_name):
        self.load_bop_state(bop_name, row)
        on_bop_name = self.Get_bop_name(row.ObjInto)
        self.board[on_bop_name]["state"]["ObjInto"] = 1
        # update the action
        self.init_acts()
        self.acts[bop_name]["on_car"] = 1


    def UpdateOut(self, row, bop_name):
        self.load_bop_state(bop_name, row)
        out_bop_name = self.Get_bop_name(row.ObjOut)
        self.board[out_bop_name]["state"]["ObjSon"] = 0
        # update the action
        self.init_acts()
        self.acts[bop_name]["off_car"] = 1

    # 处理攻击事件
    def Attack(self, row, bop_name, stageid, group):
        # 检查是否有攻击事件发生
        if not row.AttackID:
            raise Exception('This row is not attacking happened')
        # 加载攻击方状态
        self.load_bop_state(bop_name, row) 
        # 得到目标方的名字
        target_bop_name = self.Get_bop_name(row.TarID)
        # 筛选小于等于当前阶段的比赛阶段，并得到等于目标单位的ID
        TarBop_select = group.loc[(group["StageID"] <= stageid) & (group["ObjID"] == row.TarID)]
        if not TarBop_select.empty:
            # 按照时间戳(DateAndTime)升序排序，取最后一行(iloc[-1])，获取目标单位的士兵ID(ObjSon)
            SoldierID = int(TarBop_select.sort_values(by="DateAndTime", axis=0, ascending=True).iloc[-1]['ObjSon'])
        else:
            raise Exception

        if row.TarBlood == 0:
            self.board[target_bop_name]["state"], self.board[target_bop_name]["Pos"] = self.get_bop_death()
            if SoldierID:
                soldier_name = self.Get_bop_name(SoldierID)
                self.board[soldier_name]["state"], self.board[soldier_name]["Pos"] = self.get_bop_death()
        else:
            self.board[target_bop_name]["state"]["ObjBlood"] = to_categorical(row.TarBlood, 4)
            if row.TarKeep != 0:
                self.board[target_bop_name]["state"]["ObjKeep"] = 1
            if SoldierID:
                soldier_name = self.Get_bop_name(SoldierID)
                soldier_blood = from_categorical(self.board[soldier_name]["state"]["ObjBlood"]) - row.TarLost
    
                if soldier_blood == 0:
                    self.board[soldier_name]["state"], self.board[soldier_name]["Pos"] = self.get_bop_death()
                else:
                    self.board[soldier_name]["state"]["ObjBlood"] = to_categorical(soldier_blood, 4)

        # 更新分数
        self.get_score(row, SoldierID)

        # self.Updata_score(row, SoldierID)

        # 更新行动记录
        self.init_acts()
        self.acts[bop_name]["shoot"] = 1


    def Occupy(self, row, bop_name):
        if not row.CityTake:
            raise Exception('This row is not occupying happened')
        # 更新全部属性和分数
        if row.ObjPos == 80048:
            if row.StageID % 2 == 1:
                if self.board["city"][0:2] == [0, 1]:
                    self.score["red_occupy"] += 80
                    self.score["red_total"] += 80
                    self.score["blue_occupy"] -= 80
                    self.score["blue_total"] -= 80
                else:
                    self.score["red_occupy"] += 80
                    self.score["red_total"] += 80
                self.board["city"][0:2] = [1, 0]

            else:
                if self.board["city"][0:2] == [1, 0]:
                    self.score["red_occupy"] -= 80
                    self.score["red_total"] -= 80
                    # self.score["red_value_remain"] -= 80
                    self.score["blue_occupy"] += 80
                    self.score["blue_total"] += 80
                    # self.score["blue_value_remain"] += 80
                else:
                    self.score["blue_occupy"] += 80
                    self.score["blue_total"] += 80
                    # self.score["blue_value_remain"] += 80
                self.board["city"][0:2] = [0, 1]

        elif row.ObjPos == 100049:
            if row.StageID % 2 == 1:
                if self.board["city"][2:4] == [0, 1]:
                    self.score["red_occupy"] += 50
                    self.score["red_total"] += 50
                    # self.score["red_value_remain"] += 50
                    self.score["blue_occupy"] -= 50
                    self.score["blue_total"] -= 50
                    # self.score["blue_value_remain"] -= 50
                else:
                    self.score["red_occupy"] += 50
                    self.score["red_total"] += 50
                    # self.score["red_value_remain"] += 50
                self.board["city"][2:4] = [1, 0]

            else:
                if self.board["city"][2:4] == [1, 0]:
                    self.score["red_occupy"] -= 50
                    self.score["red_total"] -= 50
                    
                    self.score["blue_occupy"] += 50
                    self.score["blue_total"] += 50
                    
                else:
                    self.score["blue_occupy"] += 50
                    self.score["blue_total"] += 50
                    
                self.board["city"][2:4] = [0, 1]
        else:
            raise Exception('program error')

        # update the action
        self.init_acts()
        self.acts[bop_name]["occupy"] = 1

    def to_Hide(self, row, bop_name):
        if not row.ObjHide:
            raise Exception('This row is not hiding happened')
        self.load_bop_state(bop_name, row)
        # update the action
        self.init_acts()
        self.acts[bop_name]["to_hide"] = 1

    def to_Pass(self, row, bop_name):
        self.load_bop_state(bop_name, row)
        # update the action
        self.init_acts()
        self.acts[bop_name]["to_pass"] = 1

    def to_move(self, row, bop_name):
        self.load_bop_state(bop_name, row)
        # update the action
        self.init_acts()
        self.acts[bop_name]["to_move"] = 1

    def KeepCancel(self, row, bop_name):
        self.load_bop_state(bop_name, row)
        if not row.ObjBlood:
            self.board[bop_name]["state"], self.board[bop_name]["Pos"] = self.get_bop_death()
        # update the action
        self.init_acts()
        self.acts[bop_name]["cancel_keep"] = 1

    def Move(self, row, bop_name):
        self.load_bop_state(bop_name, row)
        if row.ObjSon:
            ObjSon_name = self.Get_bop_name(row.ObjSon)
            self.board[ObjSon_name]["Pos"] = cvtInt6loc2HexOffset(row.ObjNewPos)

        # update the action
        old_pos = cvtInt6loc2HexOffset(row.ObjPos)
        new_pos = cvtInt6loc2HexOffset(row.ObjNewPos)
        direction = self.which_side(old_pos, new_pos)

        self.init_acts()
        self.acts[bop_name][direction] = 1

    def Get_bop_name(self, ObjID):
        bop_name = None
        for key in bops_dic:
            if self.board[key]["ID"] == ObjID:
                bop_name = key
        if not bop_name:
            raise Exception('program error')

        return bop_name

    # def Updata_score(self, row, SoldierID):
    #     TarID = row.TarID


    def get_score(self, row, SoldierID):
        TarID = row.TarID
        csv_object_ObjValue = self.csv_object.loc[(self.csv_object["Filename"] == row.Filename) & (self.csv_object["ID"] == TarID)]
        
        if not csv_object_ObjValue.empty:
            ObjValue = csv_object_ObjValue.ObjValue.astype(int).values
        else:
            raise Exception
        if SoldierID != 0:
            SoldierValue = (self.csv_object.loc[
                (self.csv_object["Filename"] == row.Filename) & (self.csv_object["ID"] == SoldierID)].ObjValue).astype(int).values
        else:
            SoldierValue = 0
        kill_num = row.TarLost
        result = kill_num * (int(ObjValue) + int(SoldierValue))
        # print("result:", result)

        Tar_name = self.Get_bop_name(TarID)
        Tar_color =Tar_name.split('_')[0]
        # Tar_color =row.GameColor
        if Tar_color == 'red':
            self.score["red_lost"] += result
            # self.score["red_value_remain"] -= result
            # self.score["blue_shoot"] = TarID
            self.score["blue_shoot"] += result
            self.score["blue_total"] += result
        else:
            # self.score["red_shoot"] = TarID
            self.score["red_shoot"] += result
            self.score["red_total"] += result
            # self.score["blue_value_remain"] -= result
            self.score["blue_lost"] += result

    def which_side(self, old_hex, to_hex):
        """Tuple of 1, 2, or 6(same hex) hexsides passed through on way to_hex.
        Accurate for distances up to about 10000 hexes.
        :return: Tuple of integer hexsides.
        """

        if old_hex == to_hex:
            raise Exception
        d_x = old_hex[0] - to_hex[0]
        d_y = old_hex[1] - to_hex[1]
        # Same hex column.
        if d_x == 0:
            if d_y < 0:
                return 'east'
            else:
                return 'west'
        # Even / odd columns need a nudge.
        if (old_hex[0] % 2) and not (to_hex[0] % 2):
            d_y += .5
        if not (old_hex[0] % 2) and (to_hex[0] % 2):
            d_y -= .5
        # Hexes are wider than tall.
        d_y *= -20.0  # Negative cause math origin is lower left not upper left.
        d_x *= 17.32
        angle = round(math.degrees(math.atan2(d_y, d_x)), 2)
        if angle > 0 and angle < 60:
            answer = 'north_east'
        elif angle > 60 and angle < 120:
            answer = 'east'
        elif angle > 120 and angle < 180:
            answer = 'south_east'
        elif angle < 0 and angle > -60:
            answer = 'north_west'
        elif angle < -60 and angle > -120:
            answer = 'west'
        elif angle < -120 and angle > -180:
            answer = 'south_west'
        elif angle == 60:
            answer = 'east'
        elif angle == 0:
            answer = 'north_east'
        elif angle == -60:
            answer = 'north_west'
        elif angle == -120:
            answer = 'west'
        elif abs(angle) == 180:
            answer = 'south_east'
        elif angle == 120:
            answer = 'west'
        else:

            raise Exception
        return answer
