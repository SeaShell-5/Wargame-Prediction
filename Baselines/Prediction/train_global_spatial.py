from __future__ import print_function

import os
import json
import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from scipy import sparse
import time
import logging
from datetime import datetime
from util.util import plot_position, metric, all_true, plot_predict_heatmap
# from sklearn.metrics import mean_absolute_error, mean_squared_error


class TrainGlobalSpacial(object):
    def __init__(self):
        pass

    @staticmethod
    def train(model, env, args):
        # save_pic_dir = os.path.join('../../data/result/{}'.format(args.name))
        # if not os.path.isdir(save_pic_dir):
        #     os.makedirs(save_pic_dir)

        all_init_flag = False
        STEPS = 10

        record_MSE, record_MAE, record_RMSE, record_step = [], [], [], []


        #################################### TRAIN ####################################
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

        gpu_id = args.gpu_id

        with torch.cuda.device(gpu_id):

            model = model.cuda() if gpu_id >= 0 else model
        
        model.train()
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.00001)

        epoch = 0
        save = args.save_intervel
        # save_pic = args.save_intervel / STEPS
        env_return = env.n_step(all_init=True)
        if env_return is not None:
            (states_S, states_G, scores_E), require_init, _, see_n_replays = env_return
            

        with torch.cuda.device(gpu_id):
            states_S = torch.from_numpy(states_S).float()
            # print(states_S.shape)
            states_G = torch.from_numpy(states_G).float()
            # print(states_G.shape)
            # actions_gt = torch.from_numpy(actions_gt).float()  #.squeeze()
            # print(actions_gt.shape)
            scores_E = torch.from_numpy(scores_E).float() 
            # print(scores_E.shape)
            see_n_replays = torch.from_numpy(see_n_replays.astype(float)).float()
            if gpu_id >= 0:
                states_S = states_S.cuda()
                states_G = states_G.cuda()
                # actions_gt = actions_gt.cuda()
                scores_E = scores_E.cuda()
                see_n_replays = see_n_replays.cuda()



        while True:
            values = model(Variable(states_G), Variable(states_S), require_init)
            values_mse = F.mse_loss(values, scores_E)
            mae, rmse, _ = metric(values, scores_E)
            model.zero_grad()
            values_mse.backward() 
            optimizer.step()

            if all_true(require_init):
                model.detach()
            if env.epoch > epoch:
                epoch = env.epoch
                for p in optimizer.param_groups:
                    p['lr'] *= 0.6

            if env.step_count() % STEPS == 0:  # 每10步记录一次详细日志
                values_mse_np = values_mse.detach().cpu().numpy().tolist()
                mae_np = mae.detach().cpu().numpy().tolist()
                rmse_np = rmse.detach().cpu().numpy().tolist()
                print(f"Step {env.step_count()}: MSE={values_mse.item():.5f}, "
                          f"MAE={mae_np:.5f}, RMSE={rmse_np:.5f}")
                print(f"Values范围: [{values.min().item():.5f}, {values.max().item():.5f}], "
                          f"Scores范围: [{scores_E.min().item():.5f}, {scores_E.max().item():.5f}]")
                print(f"MSE Loss:{values_mse.item():.5f}")
                record_MSE.append(values_mse.detach().cpu().numpy().tolist())
                record_MAE.append(mae_np)
                record_RMSE.append(rmse_np)
                record_step.append(env.step_count())
            else:
                # 简单打印，不记录到日志
                print(f"MSE: {values_mse.item():.5f}")
                print(f"MAE={mae:.5f}, RMSE={rmse:.5f}")
            

            ####################### NEXT BATCH ###################################
            env_return = env.n_step(all_init=all_init_flag)
            if env_return is not None:
                # (raw_states_S, raw_states_G, raw_rewards), require_init, raw_see_n_replays = env_return
                (raw_states_S, raw_states_G, raw_scores_E), require_init, _, raw_see_n_replays = env_return
                states_G = states_G.copy_(torch.from_numpy(raw_states_G).float())
                states_S = states_S.copy_(torch.from_numpy(raw_states_S).float())
                scores_E = scores_E.copy_(torch.from_numpy(raw_scores_E).float())
                # actions_gt = actions_gt.copy_(torch.from_numpy(raw_rewards).float())
                see_n_replays = see_n_replays.copy_(torch.from_numpy(raw_see_n_replays.astype(float)).float())
            if env.step_count() > save or env_return is None:
                
                save = env.step_count() + args.save_intervel
                torch.save(model.state_dict(),
                           os.path.join(args.model_path, 'model_iter_{}.pth'.format(env.step_count())))
                

            if env_return is None:
                env.close()
                break
        # dic = {'record_loss': record_loss,
        #        'record_loss_step': record_loss_step,
        #        'record_acc': record_acc,
        #        'record_acc_step': record_acc_step}
        dic = {'record_MSE': record_MSE,
                'record_MAE': record_MAE,
                'record_RMSE': record_RMSE,
                'record_step': record_step}
        with open(os.path.join(args.save_path, args.net_name, '{}_result.json'.format(args.module)), 'w', encoding='utf-8') as json_file:
            json.dump(dic, json_file, ensure_ascii=False)

    @staticmethod
    def test(model, env, args):
        ######################### SAVE RESULT ############################
        all_init_flag = True
        record_MSE, record_MAE, record_RMSE, record_step = [], [], [], []
        STEPS = 10
        each_game_dic = {}
        ######################### TEST ###################################
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

        gpu_id = args.gpu_id
        with torch.cuda.device(gpu_id):
            model = model.cuda() if gpu_id >= 0 else model
        model.eval()

        env_return = env.n_step(test_mode=True, all_init=all_init_flag)
        if env_return is not None:
            (states_S, states_G, scores_E), require_init, game_name_temp, _= env_return
        with torch.cuda.device(gpu_id):
            states_G = torch.from_numpy(states_G).float()
            states_S = torch.from_numpy(states_S).float()
            scores_E = torch.from_numpy(scores_E).float()
            # actions_gt = torch.from_numpy(actions_gt).float()
            if gpu_id >= 0:
                states_G = states_G.cuda()
                states_S = states_S.cuda()
                scores_E = scores_E.cuda()
                # actions_gt = actions_gt.cuda()

        while True:
            values = model(Variable(states_G), Variable(states_S), require_init)
            mae, rmse, mse = metric(values, scores_E)
            
            mse = np.squeeze(mse.data.cpu().numpy())
            mae = np.squeeze(mae.data.cpu().numpy())
            rmse = np.squeeze(rmse.data.cpu().numpy())

            if env.step_count() % STEPS == 0:  # 每10步记录一次详细日志
                print(f"Step {env.step_count()}: MSE={mse:.5f}, "
                          f"MAE={mae:.5f}, RMSE={rmse:.5f}")
                print(f"Values范围: [{values.min().item():.5f}, {values.max().item():.5f}], "
                          f"Scores范围: [{scores_E.min().item():.5f}, {scores_E.max().item():.5f}]")
                record_MSE.append(mse)
                record_MAE.append(mae)
                record_RMSE.append(rmse)
                record_step.append(env.step_count())
                
        #     ########################### NEXT BATCH #############################################
            env_return = env.n_step(test_mode=True, all_init=all_init_flag)

            if env_return is not None:
                (raw_states_S, raw_states_G, raw_scores_E), require_init, game_name, _ = env_return
                # (raw_states_S, raw_states_G, raw_actions), require_init, see_n_piece = env_return
                states_G = states_G.copy_(torch.from_numpy(raw_states_G).float())
                states_S = states_S.copy_(torch.from_numpy(raw_states_S).float())
                # actions_gt = actions_gt.copy_(torch.from_numpy(raw_actions).float())
                scores_E = scores_E.copy_(torch.from_numpy(raw_scores_E).float())
                if game_name != game_name_temp:
                    each_game_dic[game_name_temp] = {'record_MSE': record_MSE.copy(),
                                                    'record_MAE': record_MAE.copy(),
                                                    'record_RMSE': record_RMSE.copy(),
                                                    'record_MAE_mean': np.mean(record_MAE),
                                                    'record_MAE_mean': np.mean(record_MAE),
                                                    'record_RMSE_mean': np.mean(record_RMSE),}
                    
                    game_name_temp = game_name
                    record_MSE.clear()
                    record_MAE.clear()
                    record_RMSE.clear()

            else:
                env.close()
                break
        return each_game_dic
        # return value_pre_per_replay, value_gt_per_replay

    @staticmethod
    def test_pro(model, env, args):
        ######################### SAVE RESULT ############################
        all_init_flag = True
        record_MSE, record_MAE, record_RMSE, record_step = [], [], [], []
        STEPS = 10
        each_game_dic = {}
        rel_pre = {}
        ######################### TEST ###################################
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

        gpu_id = args.gpu_id
        with torch.cuda.device(gpu_id):
            model = model.cuda() if gpu_id >= 0 else model
        model.eval()

        env_return = env.n_step(test_mode=True, all_init=all_init_flag)
        if env_return is not None:
            (states_S, states_G, scores_E), require_init, game_name_temp, _= env_return
        with torch.cuda.device(gpu_id):
            states_G = torch.from_numpy(states_G).float()
            states_S = torch.from_numpy(states_S).float()
            scores_E = torch.from_numpy(scores_E).float()
            # actions_gt = torch.from_numpy(actions_gt).float()
            if gpu_id >= 0:
                states_G = states_G.cuda()
                states_S = states_S.cuda()
                scores_E = scores_E.cuda()
                # actions_gt = actions_gt.cuda()

            ############################ PLOT ##########################################
        while True:
            values = model(Variable(states_G), Variable(states_S), require_init)

            mae, rmse, mse = metric(values, scores_E)
            scores_E_np = scores_E.tolist().copy()
            mse = mse.tolist()
            mae = mae.tolist()
            rmse = rmse.tolist()
            

            if env.step_count() % STEPS == 0:  # 每10步记录一次详细日志
                print(f"Step {env.step_count()}: MSE={mse:.5f}, "
                          f"MAE={mae:.5f}, RMSE={rmse:.5f}")
                print(f"Values范围: [{values.min().item():.5f}, {values.max().item():.5f}], "
                          f"Scores范围: [{scores_E.min().item():.5f}, {scores_E.max().item():.5f}]")
                record_MSE.append(mse)
                record_MAE.append(mae)
                record_RMSE.append(rmse)
                # record_step.append(env.step_count())
            ########################### NEXT BATCH ############################################
            env_return = env.n_step(test_mode=True, all_init=all_init_flag)

            if env_return is not None:
                (raw_states_S, raw_states_G, raw_scores_E), require_init, game_name, _ = env_return
                # (raw_states_S, raw_states_G, raw_actions), require_init, see_n_piece = env_return
                states_G = states_G.copy_(torch.from_numpy(raw_states_G).float())
                states_S = states_S.copy_(torch.from_numpy(raw_states_S).float())
                # actions_gt = actions_gt.copy_(torch.from_numpy(raw_actions).float())
                scores_E = scores_E.copy_(torch.from_numpy(raw_scores_E).float())
                if game_name != game_name_temp:
                    
                    each_game_dic[game_name_temp] = {
                        'record_MSE': record_MSE.copy(),
                        'record_MAE': record_MAE.copy(),
                        'record_RMSE': record_RMSE.copy()
                    }
                    
                    rel_pre[game_name_temp] = {
                        'rel': scores_E_np,
                        'pre': values.tolist().copy()
                    }
                    game_name_temp = game_name
                    record_MSE.clear()
                    record_MAE.clear()
                    record_RMSE.clear()
                    

            else:
                
                env.close()
                break
        return each_game_dic, rel_pre

    


    # @staticmethod
    # def predect(model, env, args):

    #     all_init_flag = True
    #     action_pre_per_replay = [[[]] for _ in range(6)]
    #     action_gt_per_replay = [[[]] for _ in range(6)]
    #     ######################### TEST ###################################
    #     torch.manual_seed(args.seed)
    #     torch.cuda.manual_seed(args.seed)

    #     gpu_id = args.gpu_id
    #     with torch.cuda.device(gpu_id):
    #         model = model.cuda() if gpu_id >= 0 else model
    #     model.eval()

    #     env_return = env.n_step(test_mode=True, all_init=all_init_flag)
    #     if env_return is not None:
    #         ((states_S, states_G, actions_gt), all_feature), require_init, see_n_piece = env_return
    #         S = np.asarray(all_feature.S).squeeze()
    #         G = np.asarray(all_feature.G).squeeze()
    #         A = np.asarray(all_feature.A).squeeze()
    #         print(S.shape, G.shape, A.shape)
    #     with torch.cuda.device(gpu_id):
    #         states_G = torch.from_numpy(states_G).float()
    #         states_S = torch.from_numpy(states_S).float()
    #         actions_gt = torch.from_numpy(actions_gt).float()
    #         if gpu_id >= 0:
    #             states_G = states_G.cuda()
    #             states_S = states_S.cuda()
    #             actions_gt = actions_gt.cuda()
    #     states_tensor = []
    #     states_vector = []
    #     acts = []
    #     while True:
    #         actions = model(Variable(states_G), Variable(states_S), require_init)
    #         ############################ PLOT ##########################################

    #         actions_np = np.asarray([action.data.cpu().numpy().reshape(6, env.frame_size[0], env.frame_size[1])
    #                                  for action in actions]).squeeze()

    #         # actions_gt_np = np.asarray([np.argmax(action_gt.cpu().numpy().reshape(args.n_replays, 6, -1), axis=2)
    #         #                             for action_gt in actions_gt]).squeeze(0)
    #         # print(actions_np.shape, actions_gt_np.shape)
    #         see_n_piece = see_n_piece.squeeze()
    #         if args.net_name != 'RES_tensor_vector' and all_true(require_init):
    #             model.detach()
    #         for i, see_not in enumerate(see_n_piece):
    #             if not see_not:
    #                 S[7 + i, :, :] = actions_np[i, :, :]

    #         if all_true(require_init) and len(states_tensor) > 0:
    #             spatial_states_np = np.concatenate(states_tensor).reshape([len(states_vector), -1])
    #             vector_states_np = np.vstack(states_vector)
    #             acts_np = np.vstack(acts)
    #             print(spatial_states_np.shape, vector_states_np.shape, acts_np.shape)
    #             states_tensor = []
    #             states_vector = []
    #             acts = []

    #             sparse.save_npz(os.path.join(args.save_data_path, path[0].split('/')[-1][:-4]),
    #                             sparse.csr_matrix(spatial_states_np))
    #             sparse.save_npz(os.path.join(args.save_data_path, path[1].split('/')[-1][:-4]),
    #                             sparse.csr_matrix(vector_states_np))
    #             sparse.save_npz(os.path.join(args.save_data_path, path[2].split('/')[-1][:-4]),
    #                             sparse.csr_matrix(acts_np))

    #         states_tensor.append(S)
    #         states_vector.append(G)
    #         acts.append(A)
    #         path = env.path
    #         ########################### NEXT BATCH #############################################
    #         env_return = env.n_step(test_mode=True, all_init=all_init_flag)

    #         if env_return is not None:
    #             ((raw_states_S, raw_states_G, raw_actions), all_feature), require_init, see_n_piece = env_return
    #             S = np.asarray(all_feature.S).squeeze()
    #             G = np.asarray(all_feature.G).squeeze()
    #             A = np.asarray(all_feature.A).squeeze()

    #             states_G = states_G.copy_(torch.from_numpy(raw_states_G).float())
    #             states_S = states_S.copy_(torch.from_numpy(raw_states_S).float())
    #             actions_gt = actions_gt.copy_(torch.from_numpy(raw_actions).float())
    #         else:
    #             spatial_states_np = np.concatenate(states_tensor).reshape([len(states_vector), -1])
    #             vector_states_np = np.vstack(states_vector)
    #             acts_np = np.vstack(acts)

    #             sparse.save_npz(os.path.join(args.save_data_path, path[0].split('/')[-1][:-4]),
    #                             sparse.csr_matrix(spatial_states_np))
    #             sparse.save_npz(os.path.join(args.save_data_path, path[1].split('/')[-1][:-4]),
    #                             sparse.csr_matrix(vector_states_np))
    #             sparse.save_npz(os.path.join(args.save_data_path, path[2].split('/')[-1][:-4]),
    #                             sparse.csr_matrix(acts_np))
    #             env.close()
    #             break
    #     return True
