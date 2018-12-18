from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import os, imageio
import scipy.misc
from global_var import *
from global_var import GLV
from metric import init_mnistdataloader, init_scenedataloader
from torchvision.utils import save_image
import torch, pickle
import torch.nn.functional as F

cuda = True if torch.cuda.is_available() else False
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# feature length of furniture colors and small objects respectively
fl_furcolor = GLV.fl_furcolor
fl_smallobj = GLV.fl_smallobj
feature_size = GLV.feature_size
describe_word_num = GLV.describe_word_num
fur_type_num = GLV.fur_type_num


def init_globals(args):
    GLV.metric_sample_size = args.metric_sample_size
    GLV.metric_knn_k = args.metric_knn_k
    GLV.data_loader = None
    GLV.class_num = args.class_num
    GLV.batch_size = args.batch_size
    GLV.dataset_path = args.dataset_path
    GLV.epochs = args.epoch
    GLV.z_dim = args.z_dim
    GLV.save_dir = args.save_dir
    GLV.result_dir = args.result_dir
    GLV.log_dir = args.log_dir
    GLV.model_name = args.gan_type
    GLV.dataset = args.dataset
    GLV.metric_out_dir = args.metric_out_dir
    GLV.gpu_mode = args.gpu_mode
    GLV.n_critic = args.critic
    GLV.batch_mode = args.batch_mode
    GLV.attention = args.attention
    GLV.net_complex = args.net_complex
    GLV.gamma_sparsity = args.gamma_sparsity
    GLV.dataset_noise = args.dataset_noise
    GLV.filter_distance = args.filter_distance
    GLV.gumbel_temp = args.gumbel_temp
    GLV.output_metric_hard = args.output_metric_hard
    GLV.clipping = args.clipping
    GLV.WGAN_Loss = args.WGAN_Loss
    GLV.WGAN_GP = args.WGAN_GP
    GLV.WGAN_GP_lambda = args.WGAN_GP_lambda
    GLV.lrG = args.lrG
    GLV.lrD = args.lrD
    GLV.beta1 = args.beta1
    GLV.beta2 = args.beta2


def load_dist_gt():
    furniture_dist = {}
    furniture_types = GLV.furniture_types
    cluster_num = GLV.cluster_num
    adjs = GLV.adjs
    for i in range(len(furniture_types)):
        furniture_dist[furniture_types[i]] = np.zeros((cluster_num, len(adjs)))

    smallobj_types = GLV.smallobj_types
    dec_dist = {}
    for i in range(len(smallobj_types)):
        dec_dist[smallobj_types[i]] = np.zeros(len(adjs))

    with open(GLV.dist_file) as fp:
        lines = fp.readlines()

    fur_mode = True
    dec_mode = False
    f = ''
    d = ''
    for i in range(len(lines)):
        line = lines[i].strip()
        if line.startswith('Furniture type') and fur_mode:
            f = line.split(':')[1]
        elif line.startswith('Decoration type'):
            fur_mode = False
            dec_mode = True
            d = line.split(':')[1]
        elif fur_mode:
            c_i = int(line.split(':')[0])
            probs = (line.split(':')[1].split(' '))[0: len(adjs)]
            for pi in range(len(probs)):
                if f in GLV.furniture_types:
                    furniture_dist[f][c_i, pi] = float(probs[pi])
        elif dec_mode:
            probs = (line.split(' '))[0: len(adjs)]
            for pi in range(len(probs)):
                if d in GLV.smallobj_types:
                    dec_dist[d][pi] = float(probs[pi])
        else:
            print('I am very curious of when will reach here..')

    # normalization of furniture_distribution
    GLV.fur_dist_gt = {}
    for k, v in furniture_dist.items():
        GLV.fur_dist_gt[k] = np.zeros(v.shape)
        for i in range(len(adjs)):
            probs = v[:, i]
            GLV.fur_dist_gt[k][:, i] = linear_norm(furniture_dist[k][:, i])

    # normalization of decoration_dist
    GLV.dec_dist_gt = {}
    for k, v in dec_dist.items():
        GLV.dec_dist_gt[k] = linear_norm(v)


def linear_norm(x):
    if x.sum() > 0:
        return x/x.sum()
    else:
        return x


def load_dataset(args):
    if args.dataset == 'mnist':
        GLV.img_size = 28
        GLV.data_loader = DataLoader(datasets.MNIST('./data/mnist', train=True, download=True,
                                                     transform=transforms.Compose([
                                                         transforms.Resize(GLV.img_size),
                                                         transforms.ToTensor(),
                                                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                                     ])), batch_size=GLV.batch_size, shuffle=True)
        GLV.input_dim = GLV.img_size * GLV.img_size
        GLV.H = GLV.W = GLV.img_size
        init_mnistdataloader(train=True, class_num=GLV.class_num, imgSize=GLV.img_size, batch_size=GLV.batch_size)

    elif args.dataset == 'scene':
        feature_path = GLV.dataset_path + '/features.txt'
        label_path = GLV.dataset_path + '/labels.txt'
        GLV.data_loader = DataLoader(SceneFurColorSmallObjDataset(feature_path, label_path, noise=GLV.dataset_noise),
                                         batch_size=GLV.batch_size,
                                         shuffle=True)
        data = GLV.data_loader.__iter__().__next__()[0]
        GLV.input_dim = data.shape[1]  # feature dimension
        GLV.H = 20
        GLV.W = 24
        init_scenedataloader(feature_path=feature_path, label_path=label_path,
                             class_num=GLV.class_num, input_dim=GLV.input_dim, batch_size=GLV.batch_size)

    else:
        raise Exception("[!] There is no dataset of " + GLV.dataset)


def adjust_lr(optimizer, epoch):
    lr = init_lr * (0.1 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def normal_init(m, mean, std):
    if isinstance(m, nn.Linear):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


def visualize_results(G, sample_num, epoch, n_row, sample_z_, sample_y_, fix=True, final=False):
    # self.G.eval()

    path = GLV.result_dir + '/' + GLV.dataset + '/' + GLV.model_name
    if final:
        path = os.path.join(GLV.save_dir, GLV.dataset, GLV.model_name, GLV.config)

    if not os.path.exists(path):
        os.makedirs(path)

    # Sample noise
    z = torch.rand((n_row ** 2, GLV.z_dim))  # 均匀噪声
    # z = torch.randn((n_row**2, self.z_dim))  # 高斯噪声
    if GLV.gpu_mode:
        z = z.cuda()
    # z = Variable(FloatTensor(np.random.normal(0, 1, (n_row ** 2, self.z_dim))))
    # Get labels ranging from 0 to n_classes for n rows
    # labels = np.array([num for _ in range(n_row) for num in range(n_row)])
    # labels = Variable(LongTensor(labels))
    # gen_imgs = self.G(z, self.sample_y_)
    if fix:
        if GLV.attention:
            gen_imgs, _, _ = G(sample_z_, sample_y_)
        else:
            gen_imgs = G(sample_z_, sample_y_)

    else:
        #sample_y_ = self.sample_y_
        # sample_y_ = torch.zeros(self.batch_size, self.class_num).scatter_(1,
        #                                                                  torch.randint(0, self.class_num - 1, (
        #                                                                  self.batch_size, 1)).type(
        #                                                                  torch.LongTensor), 1)
        #if self.gpu_mode:
        #    sample_y_ = sample_y_.cuda()
        if GLV.attention:
            gen_imgs, _, _ = G(z, sample_y_)
        else:
            gen_imgs = G(z, sample_y_)

    if GLV.gpu_mode:
        data = gen_imgs.cpu().data
    else:
        data = gen_imgs.data

    data = data.numpy()
    if GLV.dataset == 'scene':
        diff = GLV.H * GLV.W - len(data[0])
        b = np.zeros((len(data), len(data[0])+diff))
        b[:, :-diff] = data
        data = b

    data = FloatTensor(data.reshape(sample_num, -1, GLV.H, GLV.W))
    #data = (data + 1) / 2
    save_image(data,
               path + '/' + GLV.model_name + '_epoch%03d' % epoch + '.png',
                nrow=n_row, normalize=True)


def adjust_lrD(optimizer, epoch):
    lr = 0.1 * (0.1 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def save_features(G, sample_num, sample_z_, sample_y_, epoch, fix=True, process=True, final=False):
    path = GLV.result_dir + '/' + GLV.dataset + '/' + GLV.model_name
    if final:
        path = os.path.join(GLV.save_dir, GLV.dataset, GLV.model_name, GLV.config)
    if not os.path.exists(path):
        os.makedirs(path)

    # Sample noise
    z = torch.rand((sample_num, GLV.z_dim))   # 均匀噪声
    # z = torch.randn((n_row**2, self.z_dim))  # 高斯噪声
    if GLV.gpu_mode:
        z = z.cuda()
    # z = Variable(FloatTensor(np.random.normal(0, 1, (n_row ** 2, self.z_dim))))
    # Get labels ranging from 0 to n_classes for n rows
    # labels = np.array([num for _ in range(n_row) for num in range(n_row)])
    # labels = Variable(LongTensor(labels))
    # gen_imgs = self.G(z, self.sample_y_)
    if fix:
        if GLV.attention:
            features, _, _ = G(sample_z_, sample_y_)
        else:
            features = G(sample_z_, sample_y_)
    else:
        # sample_y_ = self.sample_y_
        # sample_y_ = torch.zeros(self.batch_size, self.class_num).scatter_(1,
        #                                                                  torch.randint(0, self.class_num - 1, (
        #                                                                  self.batch_size, 1)).type(
        #                                                                  torch.LongTensor), 1)
        # if self.gpu_mode:
        #    sample_y_ = sample_y_.cuda()
        if GLV.attention:
            features, _, _ = G(z, sample_y_)
        else:
            features = G(z, sample_y_)

    if GLV.gpu_mode:
        data = features.cpu().data
    else:
        data = features.data


    if process:
        export_features(data.numpy(),
                        path + '/' + GLV.model_name + '_epoch%03d-processed' % epoch + '.txt',
                        process=process)
    else:
        export_features(data.numpy(),
                        path + '/' + GLV.model_name + '_epoch%03d-org' % epoch + '.txt',
                        process=process)


def dist_gt_plot(hist):
    path = os.path.join(GLV.save_dir, GLV.dataset, GLV.model_name)
    if GLV.batch_mode:
        path = path + '/' + GLV.config

    if not os.path.exists(path):
        os.makedirs(path)

    model_name = GLV.model_name
    x = range(len(hist['dist_gt']))

    y1 = hist['dist_gt']

    plt.plot(x, y1, label='Distance to target distribution')

    plt.xlabel('Iter')
    plt.ylabel('Distance')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    path = os.path.join(path, model_name + '_dist_gt.png')

    plt.savefig(path)

    plt.close()



def loss_plot(hist):
    path = os.path.join(GLV.save_dir, GLV.dataset, GLV.model_name)
    if GLV.batch_mode:
        path = path + '/' + GLV.config

    if not os.path.exists(path):
        os.makedirs(path)

    model_name = GLV.model_name
    x = range(len(hist['D_loss']))

    y1 = hist['D_loss']
    y2 = hist['G_loss']

    plt.plot(x, y1, label='D_loss')
    plt.plot(x, y2, label='G_loss')

    plt.xlabel('Iter')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    path = os.path.join(path, model_name + '_loss.png')

    plt.savefig(path)

    plt.close()

def D_Prob_G_plot(hist):
    path = os.path.join(GLV.save_dir, GLV.dataset, GLV.model_name)
    if GLV.batch_mode:
        path = path + '/' + GLV.config

    if not os.path.exists(path):
        os.makedirs(path)

    model_name = GLV.model_name
    x = range(len(hist['D_prob_fake']))

    y1 = hist['D_prob_fake']
    y2 = hist['D_prob_real']

    plt.plot(x, y1, label='D_prob_fake')
    plt.plot(x, y2, label='D_prob_real')

    plt.xlabel('Iter')
    plt.ylabel('Prob')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    path = os.path.join(path, model_name + '_D_Prob_fake_real.png')

    plt.savefig(path)

    plt.close()

def metric_plot(scores):
    # dim 0： epochs, dim 1: class, dim 2: metric, emd, mmd, 1-nn.acc, 1-nn.acc_real, 1-nn.acc_fake
    path = os.path.join(GLV.save_dir, GLV.dataset, GLV.model_name)
    if GLV.batch_mode:
        path = path + '/' + GLV.config

    if not os.path.exists(path):
        os.makedirs(path)

    model_name = GLV.model_name
    [x, class_n, metric_n] = scores.shape
    #x = range(len(scores)) # number of scores
    x = range(x)
    #c = scores[0].length()
    scores = scores.mean(1) # 把所有类进行平均

    wass_dis = scores[:, 0]
    mmd = scores[:, 1]
    knn_acc = scores[:, 2]
    knn_acc_r = scores[:, 3]
    knn_acc_f = scores[:, 4]

    plt.plot(x, wass_dis, label='Wasserstein Distance')
    plt.plot(x, mmd, label='MMD')
    plt.plot(x, knn_acc, label='1-NN Accuracy')
    plt.plot(x, knn_acc_r, label='1-NN Accuracy (real)')
    plt.plot(x, knn_acc_f, label='1-NN Accuracy (fake)')

    plt.xlabel('Epoch')
    plt.ylabel('Score')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    path = os.path.join(path, model_name + '_metric.png')

    plt.savefig(path)

    plt.close()

def generate_animation():
    path = GLV.result_dir + '/' + GLV.dataset + '/' + GLV.model_name + '/' + GLV.model_name
    num = GLV.epochs
    images = []
    for e in range(num):
        #img_name = path + '_epoch%03d' % (e+1) + '.png'
        img_name = path + '_epoch%03d' % (e + 1) + '.png'
        images.append(imageio.imread(img_name))
    imageio.mimsave(path + '_generate_animation.gif', images, fps=5)

def save_images(images, size, image_path):
    return imsave(images, size, image_path)

def imsave(images, size, path):
    image = np.squeeze(merge(images, size))
    return scipy.misc.imsave(path, image)

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    if (images.shape[3] in (3,4)):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3]==1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
        return img
    else:
        raise ValueError('in merge(images,size) images parameter ''must have dimensions: HxW or HxWx3 or HxWx4')


def convert2softmax(x):
    ret = torch.zeros(x.shape)
    batch_n = x.shape[0]
    cluster_num = GLV.cluster_num

    fc, sm = x[:, :fl_furcolor], x[:, fl_furcolor:feature_size]
    fc = fc.reshape(batch_n, -1, cluster_num)
    fc = fc.softmax(dim=2)
    fc = fc.reshape(batch_n, -1)

    #sm = (sm > 0).type(FloatTensor)

    ret[:, :feature_size] = torch.cat([fc, sm], 1)
    if GLV.gpu_mode:
        ret = ret.cuda()

    return ret


def convert2gumbelsoftmax(x, temperature=1):
    ret = torch.zeros(x.shape)
    batch_n = x.shape[0]
    cluster_num = GLV.cluster_num

    fc, sm = x[:, :fl_furcolor], x[:, fl_furcolor:feature_size]
    #fc = fc.reshape(batch_n, -1, cluster_num)
    for i in range(cluster_num):
        fc[:, i*fur_type_num: (i+1)*fur_type_num] = F.gumbel_softmax(fc[:, i*fur_type_num: (i+1)*fur_type_num],
                                                                      tau=temperature)

    sm = sm.sigmoid()
    #fc = fc.reshape(batch_n, -1)

    # sm = (sm > 0).type(FloatTensor)

    ret[:, :feature_size] = torch.cat([fc, sm], 1)
    if GLV.gpu_mode:
        ret = ret.cuda()

    return ret

def filter_types(x, label):
    #l = np.where(label.data.numpy() == 1)
    if GLV.gpu_mode:
        data = x.cpu().data
    else:
        data = x.data
    converted_data = np.zeros((len(data), len(data[0])))
    data = data.numpy()

    fc, sm = data[:, :fl_furcolor], data[:, fl_furcolor:feature_size]
    fc = fc.reshape(len(data), -1, GLV.cluster_num)
    for i in range(len(x)):
        if GLV.gpu_mode:
            l = np.where(label[i].cpu().data.numpy() == 1)[0]
        else:
            l = np.where(label[i].data.numpy() == 1)[0]
        if l == 0 or l == 1:
            fc[i, GLV.living_only_idx, :] = 0
        else:
            fc[i, GLV.bedroom_only_idx, :] = 0
    fc = fc.reshape(len(data), -1)
    # sm = np.float32(sm > 0.5)
    converted_data[:, :fl_furcolor] = fc
    converted_data[:, fl_furcolor: feature_size] = sm
    data = torch.FloatTensor(converted_data)

    return data


# process为true时对聚类直接进行处理
def export_features(features, path, process=True):
    N, F = features.shape[0], feature_size
    cluster_num = GLV.cluster_num
    furniture_types = GLV.furniture_types
    smallobj_types = GLV.smallobj_types
    with open(path, 'w') as fp:
        if process:
            for i in range(0, N):
                fc, sm = features[i][:fl_furcolor], features[i][fl_furcolor:feature_size]
                fc = fc.reshape(-1, cluster_num)
                cluster_index = fc.argmax(axis=1)
                #sm = np.ceil(sm)
                sm = np.round(sm)  # since sm \in (0, 1)
                fp.write(str(i % describe_word_num) +'\n')
                fp.write('Furniture_color\n')
                for fi in range(0, len(furniture_types)):
                    fp.write('%s:%d ' % (furniture_types[fi], cluster_index[fi]))
                fp.write('\n')
                fp.write('Decorations\n')
                for si in range(0, len(smallobj_types)):
                    #if sm[si] > 0:
                    if sm[si*2] > sm[si*2 + 1]:
                        fp.write('%s:%d ' % (smallobj_types[si],sm[si]))
                fp.write('\n')
        else:
            for i in range(0, N):
                fc, sm = features[i][:fl_furcolor], features[i][fl_furcolor:feature_size]
                fc = fc.reshape(-1, cluster_num)
                fp.write(str(i % describe_word_num) + '\n')
                fp.write('Furniture_color\n')
                for fi in range(0, len(furniture_types)):
                    data = ['{:.6f}'.format(x) for x in fc[fi]]
                    fp.write('%s:%s\n' % (furniture_types[fi], data))
                fp.write('Decorations\n')
                for si in range(0, len(smallobj_types)):
                    fp.write('%s:%f ' % (smallobj_types[si], sm[si*2]))
                    #fp.write('%s:%d ' % (smallobj_types[si], sm[si]))
                fp.write('\n')

def save(G, D, train_hist, score_tr):
    save_dir = os.path.join(GLV.save_dir, GLV.dataset, GLV.model_name)
    if GLV.batch_mode:
        save_dir = save_dir + '/' + GLV.config

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    torch.save(G.state_dict(), os.path.join(save_dir, GLV.model_name + '_G.pkl'))
    torch.save(D.state_dict(), os.path.join(save_dir, GLV.model_name + '_D.pkl'))

    with open(os.path.join(save_dir, GLV.model_name + '_history.pkl'), 'wb') as f:
        pickle.dump(train_hist, f)

    np.save('%s/metric_score_tr.npy' % save_dir, score_tr)

    # save configurations
    with open(os.path.join(save_dir, 'config.txt'), 'wb') as f:
        str = GLV.get_GLV_str()
        f.write(str.encode())



def load(G, D):
    save_dir = os.path.join(GLV.save_dir, GLV.dataset, GLV.model_name)

    G.load_state_dict(torch.load(os.path.join(save_dir, GLV.model_name + '_G.pkl')))
    D.load_state_dict(torch.load(os.path.join(save_dir, GLV.model_name + '_D.pkl')))
    return G, D

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)  # only difference

class SceneFurColorSmallObjDataset(Dataset):
    def __init__(self, feature_dir, label_dir, noise=False):
        self.features = []
        self.labels = []
        self.indices = []
        furniture_types = GLV.furniture_types
        smallobj_types = GLV.smallobj_types
        cluster_num = GLV.cluster_num
        index = 0
        f_cmpl_flag = False
        with open(feature_dir) as f:
            for line in f.read().splitlines():
                if len(line.split()) == 0: continue
                if line.split()[0][0].isdigit():
                    f_cmpl_flag = False
                    index = int(line)
                    feature = [0 for x in range(GLV.feature_size)] # 420要比fl_furcolor + fl_smallobj大，才可以显示
                    #feature = [0 for x in range(fl_furcolor + fl_smallobj)]
                elif line.startswith('Furniture_color'):
                    fur_color = line.split()[1:]
                    cur_fur_types = []
                    cur_fur_ci = []
                    for color_cluster in fur_color:
                        cur_fur_types.append(color_cluster.split(':')[0])
                        cur_fur_ci.append(int(color_cluster.split(':')[1]))

                    #for color_cluster in fur_color
                    for fur in furniture_types:
                        if fur in cur_fur_types:
                            i = cur_fur_types.index(fur)
                            ci = cur_fur_ci[i]
                    #for i in range(len(fur_color)):
                        #fur = cur_fur_types[i]   # color_cluster.split(':')[0]
                        #ci = cur_fur_ci[i]       # int(color_cluster.split(':')[1])
                        #if fur in furniture_types:
                            if noise:  # 不能直接使用数据集的分布，因为数据是随机的，只要保证当前cluster最大
                                # random
                                #vec = np.random.normal(size=cluster_num)
                                vec = np.random.dirichlet(np.ones(cluster_num) / 10., size=1).transpose()
                                #vec = softmax(vec)
                                # swap max with vec[ci]
                                max_v = vec.max()
                                max_i = vec.argmax()
                                tmp = vec[ci]
                                vec[ci] = max_v
                                vec[max_i] = tmp

                                feature[furniture_types.index(fur) * cluster_num:
                                        furniture_types.index(fur) * cluster_num + cluster_num] = vec
                            else:
                                feature[furniture_types.index(fur)*cluster_num + ci] = 1
                        else:
                            vec = np.zeros(cluster_num)
                            vec[np.random.randint(cluster_num)] = 1
                            #vec = np.ones(cluster_num) * 1.0/cluster_num
                            feature[furniture_types.index(fur) * cluster_num:
                                                furniture_types.index(fur) * cluster_num + cluster_num] = vec

                        #else:  # 对于不在的，就均匀分布
                        #    #if noise:
                        #    vec = np.zeros(cluster_num)
                        #    vec[np.random.randint(cluster_num)] = 1
                        #    #vec = np.ones(cluster_num) * 1.0/cluster_num
                        #    feature[furniture_types.index(fur) * cluster_num:
                        #                    furniture_types.index(fur) * cluster_num + cluster_num] = vec


                elif line.startswith('Decorations'):
                    small_obj = line.split()[1:]
                    for obj in smallobj_types:
                        if obj in small_obj:
                            if noise:
                                feature[fl_furcolor + smallobj_types.index(obj)*2] = np.random.uniform(0.8, 1)
                                feature[fl_furcolor + smallobj_types.index(obj) * 2 + 1] = 1.0 - feature[fl_furcolor + smallobj_types.index(obj) * 2]
                            else:
                                feature[fl_furcolor + smallobj_types.index(obj)*2] = 1  # used to be +=1
                                feature[fl_furcolor + smallobj_types.index(obj)*2 + 1] = 0
                        else:
                            if noise:
                                feature[fl_furcolor + smallobj_types.index(obj)*2] = np.random.uniform(0, 0.2)
                                feature[fl_furcolor + smallobj_types.index(obj) * 2 + 1] = 1.0 - feature[fl_furcolor + smallobj_types.index(obj) * 2]
                            else:
                                feature[fl_furcolor + smallobj_types.index(obj)*2] = 0  # used to be +=1
                                feature[fl_furcolor + smallobj_types.index(obj)*2 + 1] = 1

                    f_cmpl_flag = True
                if f_cmpl_flag:
                    self.features.append(feature)
                    self.indices.append(index)

        with open(label_dir) as f:
            for line in f.read().splitlines():
                if len(line.split()) == 2:
                    index = int(line.split()[0])
                    label = int(line.split()[1])
                    self.labels.append(label)
        self.features = np.array(self.features)


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]



