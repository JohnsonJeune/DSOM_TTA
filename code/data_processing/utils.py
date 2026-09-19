import torch
import os
from sklearn.metrics import f1_score
import numpy as np
import matplotlib.pyplot as plt
from data_processing.sliding_window import sliding_window
from device import DEVICE
def get_sample_weights(y, weights):
    '''
    to assign weights to each sample
    '''
    label_unique = np.unique(y) # get all unique label values (sorted in ascending order)
    sample_weights = []
    for val in y:
        idx = np.where(label_unique == val) # find the index of the current label in the unique label list
        sample_weights.append(weights[idx])
    return sample_weights


def opp_sliding_window_w_d(data_x, data_y, d, ws, ss): # window size, step size
    data_x = sliding_window(data_x,(ws,data_x.shape[1]),(ss,1))
    data_y = np.asarray([[i[-1]] for i in sliding_window(data_y,ws,ss)])
    data_d = np.asarray([[i[-1]] for i in sliding_window(d, ws, ss)])
    data_x = data_x.astype(np.float32)
    data_y = data_y.reshape(len(data_y)).astype(np.uint8)
    data_d = data_d.reshape(len(data_d)).astype(np.uint8)


    # remove 0
    unique_y, counts_y = np.unique(data_y, return_counts=True) 
    # print(unique_y)
    # print(counts_y, 'before')
    save_index = np.argwhere(data_y>0).flatten()
    data_x = data_x[save_index]
    data_y = data_y[save_index]
    data_y = data_y-1
    data_d = data_d[save_index]


    unique_y, counts_y = np.unique(data_y, return_counts=True) 
    # print(unique_y)
    # print(counts_y,'after')



    return data_x, data_y, data_d


def plot(result_csv):
    data = np.loadtxt(result_csv, delimiter=',')
    plt.figure()
    plt.plot(range(1, len(data[:, 0]) + 1), data[:, 0], color='blue', label='train')
    plt.plot(range(1, len(data[:, 1]) + 1), data[:, 1], color='red', label='test')
    plt.legend()
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Accuracy (%)', fontsize=14)
    plt.title('Training and Test Accuracy', fontsize=20)
    plt.savefig(result_csv[:-4]+'_figure.png')


def log_result(args, result_txt, best):
    log_args(args, result_txt)
    with open(result_txt, 'a') as f:
        f.write('e_acc: '+str(best[0])[:6]
                + '\t e_miF: '+str(best[1])[:6] + '\t e_maF: '+str(best[2])[:6]
                + '\t f_acc: '+str(best[3])[:6] + '\t f_miF: '+str(best[4])[:6]
                + '\t f_maF: '+str(best[5])[:6]
                + '\t best_iter: '+str(best[6]) + '\n\n')

def log_result_concise(args, result_txt, best):
    log_args(args, result_txt)
    with open(result_txt, 'a') as f:
        f.write('e_acc: '+str(best[0])[:6] + '\t best_iter: '+str(best[1]) + '\n\n')
    concise_txt = result_txt[:-4] + '_pure.txt'
    with open(concise_txt, 'a') as ff:
        ff.write(str(best[0])[:6] + '\n')

def print_args(args):
    out_str = ''
    for argument, value in sorted(vars(args).items()):
        out_str += str(argument) + ':' + str(value) + '_'
    return out_str

def log_args(args, result_txt):
    with open(result_txt, 'a') as f:
        f.write('\nSetting: Model: ' + args.now_model_name + ' Target: ' + args.target_domain + '\n')
        for argument, value in sorted(vars(args).items()):
            f.write('{}:{} '.format(argument, value))

def set_name(args):
    cur_path = 'results/' + args.dataset + '/' + args.now_model_name + '/' + args.target_domain + '/'
    dir_name = 'results/' + args.dataset + '/' + args.now_model_name + '/' + args.target_domain
    result_csv = cur_path + str(args.n_epoch) + '_' + str(args.batch_size) + '_' + args.now_model_name +'.csv'

    result_txt = cur_path + 'best_result_'+args.dataset + '.txt'
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    if not os.path.exists(cur_path):
        os.makedirs(cur_path[:-1])
    if not os.path.isfile(result_csv):
        with open(result_csv, 'w') as my_empty_csv:
            pass
    return result_csv, result_txt, dir_name


def get_scale_matrix(M, N):
    s1 = torch.ones((N, 1)) * 1.0 / N
    s2 = torch.ones((M, 1)) * -1.0 / M
    return torch.cat((s1, s2), 0)

def mmd_custorm(sample, decoded, sigma=[1]): # 0.1, 1, 10
    X = torch.cat((decoded, sample), 0)
    XX = torch.matmul(X, X.t())
    X2 = torch.sum(X * X, 1, keepdim=True)
    exp = XX - 0.5 * X2 - 0.5 * X2.t()

    M = decoded.size()[0]
    N = sample.size()[0]
    s = get_scale_matrix(M, N)
    S = torch.matmul(s, s.t())

    loss = 0
    for v in sigma:
        kernel_val = torch.exp(exp / v)
        kernel_val = kernel_val.to(DEVICE)
        loss += torch.sum(S * kernel_val)

    loss_mmd = torch.sqrt(loss)
    return loss_mmd


def measure_event_frame(predicted_label_segment, lengths_varying_segment, true_label_segment):
    """
    this function returns the correct measurements (both frame- and event-level) for chunk-based prediction on activity
    notice that 'macro' option in sklearn does not return the desired weighted maF; therefore 'weighted' option is used instead
    """
    event_acc, event_miF, event_maF, frame_acc, frame_miF, frame_maF = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    n_event = true_label_segment.size(0)
    event_acc = ((predicted_label_segment == true_label_segment).sum()).double() * 100 / n_event
    event_miF = f1_score(true_label_segment, predicted_label_segment, average='micro') * 100
    event_maF = f1_score(true_label_segment, predicted_label_segment, average='weighted') * 100

    # create frame-based vectors
    n_frame = sum(lengths_varying_segment)
    predicted_label_frame, true_label_frame = torch.LongTensor(), torch.LongTensor()
    predicted_label_frame = torch.cat([torch.cat((predicted_label_frame, predicted_label_segment[i].repeat(lengths_varying_segment[i],1)), dim=0) for i in range(n_event)])
    assert predicted_label_frame.shape[0] == n_frame

    true_label_frame = torch.cat([torch.cat((true_label_frame, true_label_segment[i].repeat(lengths_varying_segment[i],1)), dim=0) for i in range(n_event)])
    assert true_label_frame.shape[0] == n_frame

    frame_acc = ((predicted_label_frame == true_label_frame).sum()).double() * 100 / n_frame.double()
    frame_miF = f1_score(true_label_frame, predicted_label_frame, average='micro') * 100
    frame_maF = f1_score(true_label_frame, predicted_label_frame, average='weighted') * 100

    event_acc = event_acc.item()
    frame_acc = frame_acc.item()
    return event_acc, event_miF, event_maF, frame_acc, frame_miF, frame_maF

import shutil
import os
import glob
import numpy as np
from collections import Counter
from sklearn.preprocessing import StandardScaler


def download_dataset(dataset_name, file_url, dataset_dir):
    '''
        Dataset download

        Args:
        dataset_name (str): name of the dataset
        file_url (str): download link of the dataset
        dataset_dir (str): directory where the dataset is stored after download

        Returns:
        None: this function returns nothing; if the dataset already exists it returns directly, otherwise it performs the download and extraction
    '''
    # check whether the source data exists; if it does, return without downloading anything
    if os.path.exists(dataset_dir):
        return 

    print('\n==================================================[ %s dataset download ]===================================================\n'%(dataset_name))
    print('url download address is: %s\n'%(file_url))

    # get the root directory of the dataset storage path
    dir_path = dataset_dir.split('/')[0]
    # the unimib dataset cannot be downloaded directly, so it is uploaded to github and accessed via clone
    if dataset_name == 'UniMiB-SHAR' and file_url[-4:] == '.git':
        # if the clone directory of this dataset already exists, remove it first
        if os.path.exists(os.path.join(dir_path, dataset_name)):
            shutil.rmtree(os.path.join(dir_path, dataset_name))
        # clone the dataset using the git clone command
        os.system('git clone %s %s/%s' % (file_url, dir_path, dataset_name))

    else: # other datasets
        # define the storage path of the dataset archive file
        dataset_file_path = os.path.join(dir_path, 'dataset.zip')
        # download the dataset using the wget command
        os.system(f"wget -O {dataset_file_path} {file_url}")

        # extract the dataset
        while glob.glob(os.path.join(dir_path, '*.zip')):
            # iterate over all zip files
            for file in glob.glob(os.path.join(dir_path, '*.zip')):
                # try different extraction formats
                for format in ["zip", "tar", "gztar", "bztar", "xztar"]:
                    try:
                        # try to extract the file
                        shutil.unpack_archive(filename=file, extract_dir=dir_path, format=format)
                        break
                    except:
                        continue
                # delete the already extracted archive file
                os.remove(file)

    print()

    # check whether the dataset has been fully downloaded
    if not os.path.exists(dataset_dir):
        quit('dataset download failed, please check the url and network and retry')