from __future__ import print_function
from math import log
import json, pickle
import argparse
import torch, sys
import os
import logging
sys.path.append("../..")
# from Baselines.Prediction.test import show_test_result, show_test_pro_result
from data_loader.BatchEnvEnemy import *
from Baselines.Net.WarTransGRU import WarTransGRU

from Baselines.Prediction.train_spatial import TrainSpatial
from Baselines.Prediction.train_global_spatial import TrainGlobalSpacial


ArgsELPNet = {'midplanes1': 15,
                'midplanes2': 5,
                'outplanes': 1,
                'out_vector': 100,
                'gnn_hidden': 3366,
                'in_rnn': 512,
                'out_rnn': 128}

torch.cuda.empty_cache()


def get_env(name):
    if name == 'WarTransGRU':
        env = Env_WarTransGRU()
    else:
        raise Exception('error')
    return env


def resume_training(model, args):
    model_latest_file = os.path.join(args.model_path, 'model_latest.pth')
    if os.path.isfile(model_latest_file):
        model.load_state_dict(torch.load(model_latest_file))
        print('Resume the training from {}'.format(model_latest_file))
    else:
        print('Resume missing, there are not files!')


def model_train(env, args):
    if args.net_name in ['WarTransGRU']:
        if args.module == 'full':
            model = WarTransGRU(env.n_channels, ArgsELPNet['midplanes1'], ArgsELPNet['midplanes2'], ArgsELPNet['outplanes'], env.n_features,
                            ArgsELPNet['out_vector'], ArgsELPNet['in_rnn'], ArgsELPNet['out_rnn'],
                            env.frame_size[0] * env.frame_size[1],  
                            Trans_module=True, GRU_module=True, input_steps=args.input_steps,
                            out_steps=args.out_steps)
        elif args.module == 'trans':
            model = WarTransGRU(env.n_channels, ArgsELPNet['midplanes1'], ArgsELPNet['midplanes2'], ArgsELPNet['outplanes'], env.n_features,
                            ArgsELPNet['out_vector'], ArgsELPNet['in_rnn'], ArgsELPNet['out_rnn'],
                            env.frame_size[0] * env.frame_size[1],  
                            Trans_module=True, GRU_module=False, input_steps=args.input_steps,
                            out_steps=args.out_steps)
        elif args.module == 'gru':
            model = WarTransGRU(env.n_channels, ArgsELPNet['midplanes1'], ArgsELPNet['midplanes2'], ArgsELPNet['outplanes'], env.n_features,
                            ArgsELPNet['out_vector'], ArgsELPNet['in_rnn'], ArgsELPNet['out_rnn'],
                            env.frame_size[0] * env.frame_size[1], 
                            Trans_module=False, GRU_module=True, input_steps=args.input_steps,
                            out_steps=args.out_steps)
        elif args.module == 'base':
            model = WarTransGRU(env.n_channels, ArgsELPNet['midplanes1'], ArgsELPNet['midplanes2'], ArgsELPNet['outplanes'], env.n_features,
                            ArgsELPNet['out_vector'], ArgsELPNet['in_rnn'], ArgsELPNet['out_rnn'],
                            env.frame_size[0] * env.frame_size[1], 
                            Trans_module=False, GRU_module=False, input_steps=args.input_steps,
                            out_steps=args.out_steps)               
        else:
            raise Exception('error')       


        if args.resume:
            resume_training(model, args)
        TrainGlobalSpacial.train(model, env, args)

    else:
        raise Exception('error')


def model_test(env, path, args):

    if args.net_name in ['WarTransGRU']:
        if args.module == 'full':
            trans_module = True
            GRU_module = True
        elif args.module == 'trans':
            trans_module = True
            GRU_module = False
        elif args.module == 'gru':
            trans_module = False
            GRU_module = True
        elif args.module == 'base':
            trans_module = False
            GRU_module = False 
        else:
            raise Exception('error')
        
        model = WarTransGRU(env.n_channels, ArgsELPNet['midplanes1'], ArgsELPNet['midplanes2'], ArgsELPNet['outplanes'], env.n_features,
                            ArgsELPNet['out_vector'], ArgsELPNet['in_rnn'], ArgsELPNet['out_rnn'],
                            env.frame_size[0] * env.frame_size[1],  
                            Trans_module=trans_module, GRU_module=GRU_module, input_steps=args.input_steps,
                            out_steps=args.out_steps)
        model.load_state_dict(torch.load(path))
        if args.phrase == 'test':
            result = TrainGlobalSpacial.test(model, env, args)
        elif args.phrase == 'test_pro':
            result = TrainGlobalSpacial.test_pro(model, env, args)
        elif args.phrase == 'produe_feature_data':
            result = TrainGlobalSpacial.predect(model, env, args)

    else:
        raise Exception('error')
    return result


def next_path(model_folder, paths):
    models = {int(os.path.basename(model).split('.')[0].split('_')[-1])
              for model in os.listdir(model_folder) if 'latest' not in model}
    models_not_process = models - paths
    if len(models_not_process) == 0:
        return None ,None
    models_not_process = sorted(models_not_process, reverse=True)
    paths.add(models_not_process[0])

    return os.path.join(model_folder, 'model_iter_{}.pth'.format(models_not_process[0])), 'model_iter_{}'.format(models_not_process[0])


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Enemy position prediction by the features of tensor: Wargame')
    parser.add_argument('--net_name', type=str, default='WarTransGRU',
                        help='the net is: War-TransGRU, for wargame task(eg. Prediction)')
    parser.add_argument('--piece_name', type=str, default='all',
                        help='the name of the chess pieces: tank1, tank2, car1, car2, soldier1, soldier2, all'),
    parser.add_argument('--replays_path', default='../../data/train_test',
                        help='Path for training, and test set')
    parser.add_argument('--race', default='red', help='Which race? (default: red)')
    parser.add_argument('--enemy_race', default='blue', help='Which the enemy race? (default: blue)')
    parser.add_argument('--phrase', type=str, default='train',
                        help='train | test | test_one_game(default: train)')
    parser.add_argument('--gpu_id', default=0, type=int, help='Which GPU to use [-1 indicate CPU] (default: 0)')
    parser.add_argument('--module', type=str, default='full', 
                        help='full--Transformer and GRU, trans--Transformer module, gru--GRU module, base--no module')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate (default: 0.0001)')
    parser.add_argument('--seed', type=int, default=1, help='Random seed (default: 1)')
    parser.add_argument('--n_steps', type=int, default=20, 
                        help='# Total number of input and output time steps (= input_steps + out_steps)')
    parser.add_argument('--input_steps', type=int, default=15, help='Training time steps (default: 15)')
    parser.add_argument('--out_steps', type=int, default=5, help='predict time steps (default:5 )')
    parser.add_argument('--load_steps', type=int, default=1,
                        help='# the intervel steps of the frame of data (default: 1)')
    parser.add_argument('--n_replays', type=int, default=32, help='# of replays (default: 32)')
    parser.add_argument('--n_epochs', type=int, default=15, help='# of epoches (default: 10)')
    parser.add_argument('--save_intervel', type=int, default=50000,
                        help='Frequency of model saving (default: 50000)')
    parser.add_argument('--resume', type=bool, default=False,
                        help='Resume the training (default: False)')
    args = parser.parse_args()

    # args.name = args.net_name
    args.save_path = os.path.join('../', args.phrase)
    args.model_path = os.path.join('../train', args.net_name, args.module)
    print("model save path: ", args.model_path)
    piece_name_to_offset = {'all': 0, 'tank1': 0, 'tank2': 1, 'car1': 2, 'car2': 3, 'soldier1': 4, 'soldier2': 5}

    args.piece_offset = piece_name_to_offset[args.piece_name]

    print('------------ Options -------------')
    for k, v in sorted(vars(args).items()):
        print('{}: {}'.format(k, v))
    print('-------------- End ----------------')

    
    if args.phrase == 'train':
        if not os.path.isdir(args.save_path):
            os.makedirs(args.save_path)
        if not os.path.isdir(args.model_path):
            os.makedirs(args.model_path)
        
        with open(os.path.join(args.save_path, args.net_name,'{}_config'.format(args.module)), 'w') as f:
            f.write(json.dumps(vars(args)))
        

        env = get_env(args.net_name)

        path_replays = os.path.join(os.path.join(args.replays_path, '{}.json'.format(args.phrase)))
        print("path_replays:", path_replays)
        root = os.path.join('../')
        env.init(path_replays, root, args)

        model_train(env, args)  
        print('train ending')
    elif args.phrase == 'test':
        test_result_path = os.path.join(args.save_path, args.net_name, args.module)
        if not os.path.isdir(test_result_path):
            os.makedirs(test_result_path)

        paths = set()

        while True:
            path, model_name = next_path(args.model_path, paths)
            if path is not None:
                print('[{}]Testing {} ...'.format(len(paths), path))

                env = get_env(args.net_name)
                path_replays = os.path.join(os.path.join(args.replays_path, '{}.json'.format(args.phrase)))
                root = os.path.join('../')
                args.n_epochs = 1
                args.n_replays = 1
                env.init(path_replays, root, args)

                dic = model_test(env, path, args)
                # path_list.append(path)
                with open(os.path.join(test_result_path, '{}_result'.format(model_name)), 'wb') as f:
                    f.write(pickle.dumps(dic))
            else:
                print('Test ending')
                break
    elif args.phrase == 'test_pro':
        test_pro_result_path = os.path.join(args.save_path, args.net_name, args.module)
        if not os.path.isdir(test_pro_result_path):
            os.makedirs(test_pro_result_path)

        if args.net_name == 'WarTransGRU':
            number = 550176
        else:
            raise Exception('error')

        path = os.path.join(args.model_path, 'model_iter_{}.pth'.format(number))

        print('Testing {} ...'.format(path))
        env = get_env(args.net_name)
        path_replays = os.path.join(os.path.join(args.replays_path, '{}.json'.format('test')))
        root = os.path.join('../')
        args.n_epochs = 1
        args.n_replays = 1
        env.init(path_replays, root, args)

        index_dic, rel_pre_dic = model_test(env, path, args)

        with open(os.path.join(test_pro_result_path, '{}.json'.format(args.phrase)), 'w', encoding='utf-8') as json_file:
            json.dump(index_dic, json_file, ensure_ascii=False)

        with open(os.path.join(test_pro_result_path, '{}_value.json'.format(args.phrase)), 'w', encoding='utf-8') as json_file:
            json.dump(rel_pre_dic, json_file, ensure_ascii=False)

        print('Test_pro ending')

    else:
        print("input not train/test/test_pro")
        raise Exception('error')


if __name__ == '__main__':
    main()
