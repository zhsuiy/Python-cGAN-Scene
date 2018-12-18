import argparse, os, torch
from cGANScene import CGANScene
from CGAN import CGAN
from cWGAN import CWGAN
from cWGAN_GP import CWGAN_GP
from CSAGAN import CSAGAN
from cGANScenePerLabel import CGANScenePerLabel
from global_var import *
from utils import init_globals, load_dataset, load_dist_gt
import sys
import time

"""parsing and configuration"""
def parse_args():
    desc = "Pytorch implementation of GAN collections"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--gan_type', type=str, default='CGAN',
                        choices=['CGAN', 'CWGAN', 'CWGAN_GP', 'CGANScene', 'CSAGAN', 'CGANScenePerLabel'],
                        help='The type of GAN')
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'scene'],
                        help='The name of dataset')
    #parser.add_argument('--split', type=str, default='', help='The split flag for svhn and stl10')
    parser.add_argument('--dataset_path', type=str, default='data', help='Directory name to read dataset from')
    parser.add_argument('--epoch', type=int, default=100, help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=64, help='The size of batch')
    parser.add_argument('--input_dim', type=int, default=410, help='The dimension of input feature vector')
    parser.add_argument('--save_dir', type=str, default='models', help='Directory name to save the model')
    parser.add_argument('--result_dir', type=str, default='results', help='Directory name to save the generated images')
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory name to save training logs')
    parser.add_argument('--lrG', type=float, default=0.0002)
    parser.add_argument('--lrD', type=float, default=0.0002)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--gpu_mode', type=bool, default=False)
    parser.add_argument('--class_num', type=int, default=10, help='The number of different classes')
    parser.add_argument('--z_dim', type=int, default=100, help='The dimension of noise z')
    parser.add_argument('--critic', type=int, default=1, help='The frequency of training discriminator')
    parser.add_argument('--clipping',  type=float, default=0.01, help='The clipping value for WGAN')
    parser.add_argument('--metric_out_dir', type=str, default='metric_output', help='Directory name to save fake/real data for metric evaluation')
    parser.add_argument('--metric_sample_size', type=int, default=200, help='The number of samples for metric calculation')
    parser.add_argument('--metric_knn_k', type=int, default=1, help='K value of KNN classifier metric')
    parser.add_argument('--batch_mode', type=bool, default=False, help='If turn on the batch mode')
    parser.add_argument('--attention', type=bool, default=False, help='If use self attention module')
    parser.add_argument('--net_complex', type=int, default=256, help='The starting dimension of nets')
    parser.add_argument('--gamma_sparsity', type=float, default=0, help='The gamma for penalising sparsity')
    parser.add_argument('--dataset_noise', type=bool, default=False, help='Whether to add noise to real datasets')
    parser.add_argument('--filter_distance', type=bool, default=False, help='Whether to only calculate valid positions')
    parser.add_argument('--gumbel_temp', type=float, default=0.1, help='The temperature for gumbel softmax')
    parser.add_argument('--output_metric_hard', type=bool, default=False, help='Whether to use one-hot when measuring metrics')
    parser.add_argument('--WGAN_Loss', type=bool, default=False, help='Whether to use wgan loss')
    parser.add_argument('--WGAN_GP', type=bool, default=False, help='Whether to use wgan_gp')
    parser.add_argument('--WGAN_GP_lambda', type=float, default=10, help='Lambda fro gradient penalty')
    #parser.add_argument('--benchmark_mode', type=bool, default=True)

    return check_args(parser.parse_args())

"""checking arguments"""
def check_args(args):
    # --save_dir
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # --result_dir
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    # --result_dir
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    # --metric_dir
    if not os.path.exists(args.metric_out_dir):
        os.makedirs(args.metric_out_dir)

    # --epoch
    try:
        assert args.epoch >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')

    return args


"""main"""
def main():
    # parse arguments
    arg = ' '.join(sys.argv[1:]).replace('CUDA_VISIBLE_DEVICES=2', '').replace('--gpu_mode True','').replace('--dataset scene','')\
        .replace('--class_num 4', '') \
        .replace('--gan_type CGANScene', '') \
        .replace('--gan_type CWGAN_GP', '').replace('--gan_type CWGAN','').replace('--gan_type CGAN','').replace('--gan_type CSAGAN','')\
        .replace('--batch_mode True','').replace('--','').replace(' ','-')
    print(arg)
    #GLV.config = arg
    GLV.config = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    args = parse_args()
    if args is None:
        exit()
    init_globals(args)
    load_dataset(args)
    load_dist_gt()
    if args.gan_type == 'CGAN':
        cgan = CGAN(args)
    elif args.gan_type == 'CWGAN':
        cgan = CWGAN(args)
    elif args.gan_type == 'CWGAN_GP':
        cgan = CWGAN_GP(args)
    elif args.gan_type == 'CSAGAN':
        cgan = CSAGAN(args)
    elif args.gan_type == 'CGANScene':
        cgan = CGANScene(args)
    elif args.gan_type == 'CGANScenePerLabel':
        cgan = CGANScenePerLabel(args)
    else:
        raise Exception("[!] There is no option for " + args.gan_type)
    #cgan = CGAN(args)
    #cgan = CGANMNIST(args)
    #cgan = CWAN(args)
    cgan.train()
    print(" [*] Training finished!")

    # visualize learned generator
    #gan.visualize_results(args.epoch)
    print(" [*] Testing finished!")

if __name__ == '__main__':
    main()