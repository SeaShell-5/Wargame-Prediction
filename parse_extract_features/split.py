import os
import json
import glob
import numpy as np

# 记录数据特征保留的位置与比例
FLAGS = {'root': '../data/feature_data/',
         'save_path': '../data/train_test/',
         'ratio': '7:3',
         'seed': 1}



def save(replays, prefix, folder):
    print('{}/{}: {}'.format(folder, prefix, len(replays)))
    with open(os.path.join(folder, prefix + '.json'), 'w') as f:
        json.dump(replays, f)

# ratio是比例，feature_directory是特征向量路径，save_directory保存路径
def split(ratio, feature_directory, save_directory):
    result_redwin = []
    result_bluewin = []
    file_recorded = []
    
    for file in glob.glob(os.path.join(feature_directory, '*@S.npz')):
        file_name = os.path.split(file)[-1]
        file1 = file_name.split('_')[0] + '_' + file_name.split('_')[1] + '_' + file_name.split('_')[2]
        # print(file_name)
        if file1 in file_recorded:
            # print('skip')  # skip the file have recorded
            continue

        who_win = file_name.split('_')[3]
        
        vs = file_name.split('_')[4][:8]
        
        if who_win == 'redwin':
            replay_path_dict = {}
            parsed_replays_info_red = {}
            parsed_replays_info_blue = {}
            if vs == 'red2blue':
                parsed_replays_info_red['state_S'] = os.path.join(feature_directory, file_name)
                parsed_replays_info_red['state_G'] = os.path.join(feature_directory,
                                          file1 + '_' + who_win + '_' + 'red2blue@G.npz')
                # parsed_replays_info_red['state_A'] = os.path.join(feature_directory,
                #                           file1 + '_' + who_win + '_' + 'red2blue@A.npz')
                parsed_replays_info_red['state_E'] = os.path.join(feature_directory,
                                          file1 + '_' + who_win + '_' + 'red2blue@E.npz')
                
                parsed_replays_info_blue['state_S'] = os.path.join(feature_directory, file_name)
                parsed_replays_info_blue['state_G'] = os.path.join(feature_directory,
                                          file1 + '_' + who_win + '_' + 'blue2red@G.npz')
                # parsed_replays_info_blue['state_A'] = os.path.join(feature_directory,
                #                           file1 + '_' + who_win + '_' + 'blue2red@A.npz')
                parsed_replays_info_blue['state_E'] = os.path.join(feature_directory,
                                          file1 + '_' + who_win + '_' + 'blue2red@E.npz')
                
                file_recorded.append(file1)
                replay_path_dict['red'] = parsed_replays_info_red
                replay_path_dict['blue'] = parsed_replays_info_blue
            elif vs == 'blue2red':
                parsed_replays_info_blue['state_S'] = os.path.join(feature_directory, file_name)
                parsed_replays_info_blue['state_G'] = os.path.join(feature_directory,
                                          file1 + '_' + who_win + '_' + 'blue2red@G.npz')
                # parsed_replays_info_blue['state_A'] = os.path.join(feature_directory,
                #                           file1 + '_' + who_win + '_' + 'blue2red@A.npz')
                parsed_replays_info_blue['state_E'] = os.path.join(feature_directory,
                                          file1 + '_' + who_win + '_' + 'blue2red@E.npz')

                parsed_replays_info_red['state_S'] = os.path.join(feature_directory,
                                          file1 + '_' + who_win + '_' + 'red2blue@S.npz')
                parsed_replays_info_red['state_G'] = os.path.join(feature_directory,
                                          file1 + '_' + who_win + '_' + 'red2blue@G.npz')
                # parsed_replays_info_red['state_A'] = os.path.join(feature_directory,
                #                           file1 + '_' + who_win + '_' + 'red2blue@A.npz')
                parsed_replays_info_red['state_E'] = os.path.join(feature_directory,
                                          file1 + '_' + who_win + '_' + 'red2blue@E.npz')


                file_recorded.append(file1)
                replay_path_dict['blue'] = parsed_replays_info_blue
                replay_path_dict['red'] = parsed_replays_info_red
            else:
                Exception('error')
            result_redwin.append(replay_path_dict)
        elif who_win == 'bluewin':
            replay_path_dict = {}
            parsed_replays_info_red = {}
            parsed_replays_info_blue = {}
            if vs == 'red2blue':
                parsed_replays_info_red['state_S'] = os.path.join(feature_directory, file_name)
                parsed_replays_info_red['state_G'] = os.path.join(feature_directory,
                                          file1 + '_' + who_win + '_' + 'red2blue@G.npz')
                # parsed_replays_info_red['state_A'] = os.path.join(feature_directory,
                #                           file1 + '_' + who_win + '_' + 'red2blue@A.npz')
                parsed_replays_info_red['state_E'] = os.path.join(feature_directory,
                                          file1 + '_' + who_win + '_' + 'red2blue@E.npz')
                
                parsed_replays_info_blue['state_S'] = os.path.join(feature_directory,
                                          file1 + '_' + who_win + '_' + 'blue2red@S.npz')
                parsed_replays_info_blue['state_G'] = os.path.join(feature_directory,
                                          file1 + '_' + who_win + '_' + 'blue2red@G.npz')
                # parsed_replays_info_blue['state_A'] = os.path.join(feature_directory,
                #                           file1 + '_' + who_win + '_' + 'blue2red@A.npz')
                parsed_replays_info_blue['state_E'] = os.path.join(feature_directory,
                                          file1 + '_' + who_win + '_' + 'blue2red@E.npz')

                file_recorded.append(file1)
                replay_path_dict['red'] = parsed_replays_info_red
                replay_path_dict['blue'] = parsed_replays_info_blue
            elif vs == 'blue2red':
                parsed_replays_info_blue['state_S'] = os.path.join(feature_directory, file_name)
                
                parsed_replays_info_blue['state_G'] = os.path.join(feature_directory,
                                          file1 + '_' + who_win + '_' + 'blue2red@G.npz')
                # parsed_replays_info_blue['state_A'] = os.path.join(feature_directory,
                #                           file1 + '_' + who_win + '_' + 'blue2red@A.npz')
                parsed_replays_info_blue['state_E'] = os.path.join(feature_directory,
                                          file1 + '_' + who_win + '_' + 'blue2red@E.npz')

                parsed_replays_info_red['state_S'] = os.path.join(feature_directory,
                                          file1 + '_' + who_win + '_' + 'red2blue@S.npz')
                parsed_replays_info_red['state_G'] = os.path.join(feature_directory,
                                          file1 + '_' + who_win + '_' + 'red2blue@G.npz')
                # parsed_replays_info_red['state_A'] = os.path.join(feature_directory,
                #                           file1 + '_' + who_win + '_' + 'red2blue@A.npz')
                parsed_replays_info_red['state_E'] = os.path.join(feature_directory,
                                          file1 + '_' + who_win + '_' + 'red2blue@E.npz')

                file_recorded.append(file1)
                replay_path_dict['blue'] = parsed_replays_info_blue
                replay_path_dict['red'] = parsed_replays_info_red
            else:
                Exception('error')
            result_bluewin.append(replay_path_dict)
        else:
            raise Exception('error')

    if not os.path.isdir(save_directory):
        os.makedirs(save_directory)
    np.random.seed(1)
    train_end_redwin = int(len(result_redwin) * ratio[0])

    train_end_bluewin = int(len(result_bluewin) * ratio[0])
    print("game numer:", len(result_redwin) + len(result_bluewin))
    print("redwin game numer:", len(result_redwin))
    print("bluewin game numer:", len(result_bluewin))

    result_train = result_redwin[:train_end_redwin] + result_bluewin[:train_end_bluewin]
    result_test = result_redwin[train_end_redwin:] + result_bluewin[train_end_bluewin:]
    # np.random.shuffle(result_train)
    # np.random.shuffle(result_test)
    save(result_train, 'train', save_directory)
    save(result_test, 'test', save_directory)
    print("save_directory", save_directory)

def main():

    np.random.seed(FLAGS['seed'])
    ratio = np.asarray([float(i) for i in FLAGS['ratio'].split(':')])
    ratio /= np.sum(ratio)
    
    feature_directory = os.path.join(FLAGS['root'])
    save_directory = os.path.join(FLAGS['save_path'])
    print("***:",ratio, feature_directory, save_directory)

    split(ratio, feature_directory, save_directory)


if __name__ == '__main__':
    main()
