import pandas as pd
import numpy as np
import os, json

import copy
from multiprocessing import Pool
from scipy import sparse
from game_state import GameState
from state_to_feature import StateToFeature
from util.util import cvtHexOffset2Int6loc
# from plot.plot_3d import plot_3d, plot_2d

MapHeight, MapLength = 66, 51

import warnings

warnings.filterwarnings("ignore")


def load_files():
    preprocess_file_head = '../data/preprocess_data/'
    replay_file_head = '../data/replay_data/urban_terrain/'
    map_file_head = '../data/map/'
    csv_roomrecord = pd.read_csv(os.path.join(preprocess_file_head + "select_preprocess.csv"))
    csv_object = pd.read_csv(os.path.join(replay_file_head + "object.csv"))
    LOS = pd.read_csv(map_file_head + 'LOS.csv').set_index(['bop_type', 'grid_pos'])
    game_map = pd.read_csv(map_file_head + 'map83.csv')
    return csv_roomrecord, csv_object, LOS, game_map

""" 
    提取地图信息函数
    map_terrain: 地图地势
    map_urban: 地图居民位置
    map_road: 地图铁路
    map_normal:其他信息
"""
def extract_map_indicators(game_map):
    map_terrain = np.zeros((MapHeight, MapLength))
    map_urban = np.zeros_like(map_terrain)
    map_road = np.zeros_like(map_terrain)
    map_normal = np.zeros_like(map_terrain)
    for row in range(MapHeight):
        for col in range(MapLength):
            Int6loc = cvtHexOffset2Int6loc(row, col)
            map_terrain[row, col] = int(game_map[game_map['MapID'] == Int6loc]['GroundID'])
            cond = int(game_map[game_map['MapID'] == Int6loc]['Cond'])
            grid_type = int(game_map[game_map['MapID'] == Int6loc]['GridType'])
            if cond == 7 or (cond == 0 and grid_type == 3):
                map_urban[row, col] = 1
            if grid_type == 2:
                map_road[row, col] = 1
            if cond == 0 and grid_type == 0:
                map_normal[row, col] = 1
    # 高度进行最大最小归一化处理
    map_terrain = (map_terrain - map_terrain.min()) / (map_terrain.max() - map_terrain.min())
    return map_terrain, map_urban, map_road, map_normal


def StateFeature_vec_ten(stageid, bops_state, LOS, enemy_color=1, *map_args):
    # enemy_color = 1   敌方棋子颜色为蓝色
    flag = True
    if not bops_state:
        flag = False
    spatial_states_np = []
    global_states_np = []
    acts_np = []
    score_np = []
    
    game_state = StateToFeature(LOS, enemy_color)
    for state in bops_state:
        game_state.update(state)
        spatial_state_temp = game_state.to_tensor(*map_args)
        global_state_temp = game_state.to_vector()
        # acts_temp = game_state.acts_to_np()
        score_temp = game_state.to_effect(stageid, enemy_color)      

        result_step = []
        if enemy_color:
            result_step.append(np.zeros((1, MapHeight, MapLength)))
        else:
            result_step.append(np.ones((1, MapHeight, MapLength)))
        result_step.append(spatial_state_temp)
        spatial_state_temp = np.concatenate(result_step)
        spatial_states_np.append(spatial_state_temp)
        global_states_np.append(global_state_temp)
        # acts_np.append(acts_temp)
        score_np.append(score_temp)

    spatial_states_np = np.asarray(spatial_states_np)
    global_states_np = np.asarray(global_states_np)
    # acts_np = np.asarray(acts_np)
    score_np = np.asarray(score_np)

    return flag, spatial_states_np, global_states_np, score_np


def get_parameter(csv_roomrecord, csv_object, LOS, map_height, map_urban, map_road, map_normal):
    csv_roomrecord_groupby = csv_roomrecord.groupby(["Filename"])

    parameter = [(Filename, one_game_group, csv_object, LOS, map_height, map_urban, map_road, map_normal) for
            (Filename), one_game_group in csv_roomrecord_groupby]
    
    return parameter


def multi_parameter(P):
    return extract_one_game(P[0], P[1], P[2], P[3], P[4], P[5], P[6], P[7])


def extract_one_game(Filename, one_game_group, csv_object, LOS, *map_args, color="red"):
    '''
    extract_one_game:函数进行每个推演室特征提取
    Filename:记录select_preprocess.csv表格中Filename列数据
    one_game_group:从select_preprocess.csv表格中提取每场比赛的所有记录
    csv_object:得到object.csv中所有的数据[15516 rows x 79 columns]
    LOS:得到LOS.csv中所有的数据[6732 rows x 3366 columns]
    *map_args:传参地图所有tensor格式数据map_height, map_urban, map_road, map_normal[66x51]
    '''
    parts = Filename[0].split()

    Filename_str = '_'.join(item for item in parts)  # 2018530151256_01-05-03-03-02_推演室9
    Filename = '_'.join(Filename)

    if one_game_group.iloc[0]["JmResult"] >= 0:
        game_result = 'redwin'
    else:
        game_result = 'bluewin'
    save_file_name = Filename_str + '_' + game_result
    
    game_state = GameState(Filename, csv_object)
    bops, score, acts = game_state.board, game_state.score, game_state.acts
    
    one_game_group = one_game_group.sort_values(by=["DateAndTime"], axis=0, ascending=[True])

    stage_groupby = one_game_group.groupby(["StageID"])
    
    states_tensor_red = []
    states_vector_red = []
    states_act_red = []
    states_value_red = []
    
    states_tensor_blue = []
    states_vector_blue = []
    states_act_blue = []
    states_value_blue = []
    
    for stageid, group in stage_groupby:
        group1 = group.sort_values(by=["TimeID", "ObjStep"], axis=0, ascending=[False, False])

        stageid = stageid[0]

        game_state.initCancelKeep()
        bops["stage"] = stageid
        
        game_state.score["red_win"] = group.iloc[0]["JmResult"]

        board_one_stage = []
        score_one_stage = []
        acts_one_stage = []
        
        # row为每个stage中所有棋子动作记录
        for indexs, row in group1.iterrows():
            row_to_state(game_state, row, stageid, one_game_group)
            board_one_stage.append(copy.deepcopy(game_state.board))
            score_one_stage.append(copy.deepcopy(game_state.score))
            acts_one_stage.append(copy.deepcopy(game_state.acts))
            game_state.fill_no_act()
        board_one_stage.append(copy.deepcopy(game_state.board))
        score_one_stage.append(copy.deepcopy(game_state.score))
        acts_one_stage.append(copy.deepcopy(game_state.acts))
        
       
        
        # 采集红方预测蓝方棋子信息
        bops_red = zip(board_one_stage, score_one_stage, acts_one_stage)
        flag_red, tensor_one_stage_red, vector_one_stage_red, value_one_stage_red= StateFeature_vec_ten(stageid, bops_red, LOS,
                                                                                                    1, *map_args)
        # 判断数据中是否存在非数值类型
        if not np.issubdtype(np.asarray(tensor_one_stage_red).dtype, np.number):
            raise ValueError("tensor_one_stage_red 中存在非数值类型元素")

        if not np.issubdtype(np.asarray(vector_one_stage_red).dtype, np.number):
            raise ValueError("vector_one_stage_red 中存在非数值类型元素") 
        # 检查是否存在无穷大值
        if np.isinf(np.asarray(tensor_one_stage_red)).any():
            raise ValueError("tensor_one_stage_red 中存在无穷大值")
        
        if np.isinf(np.asarray(vector_one_stage_red)).any():
            raise ValueError("vector_one_stage_red 中存在无穷大值")


        if flag_red:
            states_tensor_red.append(tensor_one_stage_red)
            
            states_vector_red.append(vector_one_stage_red)
            # print("states_vector_red", states_vector_red.shape)
            # states_act_red.append(acts_one_stage_red)
            states_value_red.append(value_one_stage_red)
                
                
            
        # 采集蓝方预测红方棋子信息
        bops_blue = zip(board_one_stage, score_one_stage, acts_one_stage)
        flag_blue, tensor_one_stage_blue, vector_one_stage_blue, value_one_stage_blue = StateFeature_vec_ten(stageid, bops_blue,
                                                                                                        LOS, 0, *map_args)
        if not np.issubdtype(np.asarray(tensor_one_stage_blue).dtype, np.number):
            print(tensor_one_stage_blue)
            raise ValueError("tensor_one_stage_blue 中存在非数值类型元素")
        if np.isinf(np.asarray(tensor_one_stage_blue)).any():
            print(tensor_one_stage_blue)
            raise ValueError("tensor_one_stage_blue 中存在无穷大值")
        
        if not np.issubdtype(np.asarray(vector_one_stage_blue).dtype, np.number):
            print(vector_one_stage_blue)
            raise ValueError("vector_one_stage_blue 中存在非数值类型元素")

        if np.isinf(np.asarray(vector_one_stage_blue)).any():
            print(vector_one_stage_blue)
            raise ValueError("vector_one_stage_blue 中存在无穷大值")


        if flag_blue:
            states_tensor_blue.append(tensor_one_stage_blue)
            states_vector_blue.append(vector_one_stage_blue)
            # states_act_blue.append(acts_one_stage_blue)
            states_value_blue.append(value_one_stage_blue)
            # print("effectiveness", effectiveness)
    
    

    spatial_states_np_blue = np.concatenate(states_tensor_blue)
    spatial_states_np_blue = spatial_states_np_blue.reshape([spatial_states_np_blue.shape[0], -1])
    global_states_np_blue = np.concatenate(states_vector_blue)
    # acts_np_blue = np.concatenate(states_act_blue)
    value_np_blue = np.concatenate(states_value_blue)
    

    spatial_states_np_red = np.concatenate(states_tensor_red)
    spatial_states_np_red = spatial_states_np_red.reshape([spatial_states_np_red.shape[0], -1])
    global_states_np_red = np.concatenate(states_vector_red)
    # acts_np_red = np.concatenate(states_act_red)
    value_np_red = np.concatenate(states_value_red)

    # 对效能值数据进行归一化处理
    min_blue = value_np_blue.min(axis=0, keepdims=True)
    max_blue = value_np_blue.max(axis=0, keepdims=True)
    value_np_blue_norm = (value_np_blue - min_blue) / (max_blue - min_blue + 1e-8)

    min_red = value_np_red.min(axis=0, keepdims=True)
    max_red = value_np_red.max(axis=0, keepdims=True)
    value_np_red_norm = (value_np_red - min_red) / (max_red - min_red + 1e-8)

    print("---------------------",Filename_str,"--------------------")
    print('spatial_states_np_blue', spatial_states_np_blue.shape)
    print('global_states_np_blue', global_states_np_blue.shape)
    # print('acts_np_blue', acts_np_blue.shape)
    print('score_np_blue', value_np_blue_norm.shape)

    
    print('spatial_states_np_red', spatial_states_np_red.shape)
    print('global_states_np_red', global_states_np_red.shape)
    # print('acts_np_red', acts_np_red.shape)
    print('score_np_red', value_np_red_norm.shape)
    
 


    vs1 = 'red2blue'
    vs2 = 'blue2red'

    path = os.path.join('../data/feature_data/')
    if not os.path.isdir(path):
        os.makedirs(path)

    
    sparse.save_npz(os.path.join(path,
                                save_file_name + '_' + vs1 + '@G'), sparse.csr_matrix(global_states_np_red))
    sparse.save_npz(os.path.join(path,
                                save_file_name + '_' + vs1 + '@S'), sparse.csr_matrix(spatial_states_np_red))
    # sparse.save_npz(os.path.join(path,
    #                             save_file_name + '_' + vs1 + '@A'), sparse.csr_matrix(acts_np_red))
    sparse.save_npz(os.path.join(path,
                                save_file_name + '_' + vs1 + '@E'), sparse.csr_matrix(value_np_red_norm))

    sparse.save_npz(os.path.join(path,
                                save_file_name + '_' + vs2 + '@G'), sparse.csr_matrix(global_states_np_blue))
    sparse.save_npz(os.path.join(path,
                                save_file_name + '_' + vs2 + '@S'), sparse.csr_matrix(spatial_states_np_blue))
    # sparse.save_npz(os.path.join(path,
    #                             save_file_name + '_' + vs2 + '@A'), sparse.csr_matrix(acts_np_blue))
    sparse.save_npz(os.path.join(path,
                                save_file_name + '_' + vs2 + '@E'), sparse.csr_matrix(value_np_blue_norm))





def extract_agent_indicators(csv_roomrecord, csv_object, LOS, *args):
    parameter = get_parameter(csv_roomrecord, csv_object, LOS, *args)
    process = Pool(20)  # 设置20个进程, 每场比赛最多20个阶段, 每个阶段含有许多棋子动作
    process.map(multi_parameter, parameter)
    print("pool map done...")
    process.close()
    process.join()



def row_to_state(game_state, row, stageid, one_game_group):
    bop_name = game_state.Get_bop_name(row.ObjID)
    if row.ObjNewPos == row.ObjPos:
        if row.ObjInto != 0:
            game_state.UpdateOn(row, bop_name)

        if row.ObjOut != 0:
            game_state.UpdateOut(row, bop_name)

        if row.AttackID != 0:
            game_state.Attack(row, bop_name, stageid, one_game_group)

        if row.CityTake != 0:
            game_state.Occupy(row, bop_name)

        if row.ObjHide != 0:
            game_state.to_Hide(row, bop_name)

        if row.ObjPass == 1 and row.objname != '步兵小队':
            game_state.to_Pass(row, bop_name)

        if row.ObjPass == 0 and row.ObjHide == 0 and row.ObjKeepCancel == 0 and row.ObjRound > 1:
            game_state.to_move(row, bop_name)

        if row.ObjKeepCancel != 0:
            game_state.KeepCancel(row, bop_name)

    elif row.ObjNewPos != row.ObjPos:
        game_state.Move(row, bop_name)

    else:
        raise Exception


if __name__ == '__main__':
    # 加载数据
    csv_roomrecord, csv_object, LOS, game_map = load_files()
    # 提取地图四种特征信息
    map_terrain, map_urban, map_road, map_normal = extract_map_indicators(game_map)
    
    """
        数据分类:--静态数据-六角格作为地图最小结构单位, 由唯一的 ID 进行标识,
                包含七个特征: ID, 坐标, 高程, 地形, 道路情况, 铁路情况, 居民情况
                --动态数据-兵棋中算子, 包含各种属性
                
    """
    extract_agent_indicators(csv_roomrecord, csv_object, LOS, map_terrain, map_urban, map_road, map_normal)
    print('The data processing completed')
