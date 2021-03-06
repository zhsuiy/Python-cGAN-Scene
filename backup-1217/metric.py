import math
import os
import timeit
import math

import numpy as np
import ot
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from sklearn.metrics.regression import mean_squared_error
import pdb
# from tqdm import tqdm
from scipy.stats import entropy
from numpy.linalg import norm
from scipy import linalg
from global_var import *
import utils

cuda = True if torch.cuda.is_available() else False
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

fl_furcolor = GLV.fl_furcolor
fl_smallobj = GLV.fl_smallobj
feature_size = GLV.feature_size


def init_mnistdataloader(train=True, class_num=10, imgSize=28, batch_size=32):
    GLV.mnist_data = {}
    for i in range(class_num):
        GLV.mnist_data[i] = []

    data_loader = DataLoader(datasets.MNIST('./data/mnist', train=train, download=True,
                                            transform=transforms.Compose([
                                                transforms.Resize(imgSize),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                            ])), batch_size=batch_size)

    for iter, (x_, y_) in enumerate(data_loader):  # x_是feature, y_是label
        # step += 1
        # batch_size = x_.shape[0]  # 有可能尾部的size不一样
        x_ = x_.view(-1, imgSize**2)  # 针对MNIST数据集
        for y in range(class_num):
            y_index = np.where(y_ == y)
            GLV.mnist_data[y].extend(x_[y_index])


def init_scenedataloader(feature_path, label_path, class_num=4, input_dim=-1, batch_size=32):
    GLV.scene_data = {}
    for i in range(class_num):
        GLV.scene_data[i] = []
    #data_loader = DataLoader(utils.SceneFurColorSmallObjDataset(feature_path, label_path),
    #                     batch_size=batch_size,
    #                     shuffle=True)
    data_loader = GLV.data_loader

    for iter, (x_, y_) in enumerate(data_loader):  # x_是feature, y_是label
        # step += 1
        # batch_size = x_.shape[0]  # 有可能尾部的size不一样
        x_ = x_.view(-1, input_dim)  # scene
        for y in range(class_num):
            y_index = np.where(y_ == y)
            GLV.scene_data[y].extend(x_[y_index])



class MNISTConditional(Dataset):
    def __init__(self, label=0):
        self.features = GLV.mnist_data[label]
        self.labels = np.full((len(self.features)), label)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class SceneConditional(Dataset):
    def __init__(self, label=0):
        self.features = GLV.scene_data[label]
        self.labels = np.full((len(self.features)), label)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def giveName(iter):  # 7 digit name.
    ans = str(iter)
    return ans.zfill(7)


def sampleFake(label, dataset, netG, nz, sampleSize, batchSize, saveFolder, class_num, gpu_mode, save_images=False):
    #print('sampling fake images ...')

    if save_images:
        if not os.path.exists(saveFolder):
            os.makedirs(saveFolder)
        if not os.path.exists(saveFolder + str(label) + '/'):
            os.makedirs(saveFolder + str(label) + '/')

    sample_z = torch.rand(sampleSize, nz)
    sample_y = (torch.ones(sampleSize, 1) * label).type(torch.LongTensor)  # 全为label的y向量
    y_vec = torch.zeros(sampleSize, class_num)
    y_vec.scatter_(1, sample_y.view(sampleSize, 1), 1)
    if gpu_mode:
        sample_z, y_vec = sample_z.cuda(), y_vec.cuda()
    if GLV.attention:
        fake, _, _ = netG(sample_z, y_vec)
    else:
        fake = netG(sample_z, y_vec, training=False)

    if gpu_mode:
        data = fake.cpu().data
    else:
        data = fake.data

    if save_images:
        dim = int(fake.size()[1]**0.5)
        for iter in range(0, len(data)):
            vutils.save_image(fake.data[iter].view(1, dim, dim).mul(0.5).add(
                0.5), saveFolder + str(label) + '/' + giveName(iter) + ".png")

    #data = data.numpy()
    if dataset=='scene':

        if GLV.output_metric_hard:
            converted_data = np.zeros((len(data), len(data[0])))
            data = data.numpy()
            fc, sm = data[:,:fl_furcolor], data[: ,fl_furcolor:feature_size]
            fc = fc.reshape(len(data), -1, GLV.cluster_num)
            cluster_index = fc.argmax(axis=2)   # 取出值最大的cluster index
            cluster_index = torch.LongTensor(cluster_index.reshape(len(fc), len(fc[0]), 1))
            fi = torch.zeros(fc.shape)          # 其余位置均为0
            fi.scatter_(2, cluster_index, 1)    # 在cluster位置为1
            #sm = np.float32(sm > 0.5)  ## used to be > 0 when using tanh

            if label == 0 or label == 1:
                fi[:, GLV.living_only_idx, :] = 0
            else:
                fi[:, GLV.bedroom_only_idx, :] = 0

            if gpu_mode:
                fi = fi.view(len(data),-1).cpu().data.numpy()
            else:
                fi = fi.view(len(data), -1).data.numpy()

            converted_data[:,:fl_furcolor] = fi
            converted_data[:, fl_furcolor: feature_size] = sm
            data = torch.FloatTensor(converted_data)
        else:
            converted_data = np.zeros((len(data), len(data[0])))
            data = data.numpy()
            fc, sm = data[:, :fl_furcolor], data[:, fl_furcolor:feature_size]
            fc = fc.reshape(len(data), -1, GLV.cluster_num)
            if label == 0 or label == 1:
                fc[:, GLV.living_only_idx, :] = 0
            else:
                fc[:, GLV.bedroom_only_idx, :] = 0
            fc = fc.reshape(len(data), -1)
            #sm = np.float32(sm > 0.5)
            converted_data[:, :fl_furcolor] = fc
            converted_data[:, fl_furcolor: feature_size] = sm
            data = torch.FloatTensor(converted_data)


    #return np.concatenate(data.numpy())
    return data


def sampleTrue(label, dataset, imageSize, dataroot, sampleSize, batchSize, saveFolder, save_images=False):
    #print('sampling real images ...')
    # saveFolder = saveFolder + '0/'

    workers = 4
    if dataset == 'mnist':
        dataloader = DataLoader(MNISTConditional(label),
                                batch_size=batchSize,
                                shuffle=True)
    elif dataset == 'scene':
        dataloader = DataLoader(SceneConditional(label),
                                batch_size=batchSize,
                                shuffle=True)
    else:
        raise Exception("[!] There is no dataset of " + dataset)

    feature_r = []
    if not os.path.exists(saveFolder):
        os.makedirs(saveFolder)
    if not os.path.exists(saveFolder + str(label) + '/'):
        os.makedirs(saveFolder + str(label) + '/')

    iter = 0
    for i, (features, labels) in enumerate(dataloader):
        for j in range(0, len(features)):
            if save_images:
                vutils.save_image(features[j].view(1, imageSize, imageSize).mul(0.5).add(
                    0.5), saveFolder + str(label) + '/' + giveName(iter) + ".png")
            iter += 1
           #feature_r.append(features[j].view(-1, imageSize**2))
            feature_r.append(features[j])
            if iter >= sampleSize:
                break
        if iter >= sampleSize:
            break
        #feature_r.append(features)

    #feature_r = torch.cat(feature_r, 0)
    feature_r = torch.stack(feature_r).type(torch.FloatTensor)
    return feature_r

def distanceL1(X, Y):
    nX = X.size(0)
    nY = Y.size(0)
    X = X.view(nX, -1)
    Y = Y.view(nY, -1)
    M = torch.zeros(nX, nY)
    M = (X[:, None, :] - Y[None, ...]).abs().sum(dim=2)
    return M


def distance(X, Y, sqrt, filter=False):
    nX = X.size(0)
    nY = Y.size(0)
    X = X.view(nX, -1)
    Y = Y.view(nY, -1)
    M = torch.zeros(nX, nY)
    #M_ = torch.zeros(nX, nY)

    if GLV.dataset == 'scene' and filter:
        fc_x = X[:, :fl_furcolor]
        fc_x = fc_x.reshape(nX, -1, GLV.cluster_num)
        sum = (fc_x.sum(dim=2, keepdim=True) > 0).type(torch.FloatTensor)
        X_ = torch.ones(X.shape)
        bit = sum.expand(nX, -1, GLV.cluster_num) # 记录哪些家具是有的
        bit = bit.reshape(nX, -1)
        X_[:, :fl_furcolor] = bit

        #for i in range(nX):
        #    for j in range(nY):
        #        M[i, j] = ((X[i] - X_[i] * Y[j])**2).sum()

        #M_
        M = ((X[:, None, :] - X_[:, None, :] * Y[None, ...]) ** 2).sum(dim=2)

        #flag = (M-M_).sum()

    else:
        X2 = (X * X).sum(1).resize_(nX, 1)
        Y2 = (Y * Y).sum(1).resize_(nY, 1)


        M.copy_(X2.expand(nX, nY) + Y2.expand(nY, nX).transpose(0, 1) -
                2 * torch.mm(X, Y.transpose(0, 1)))

        del X, X2, Y, Y2

    if sqrt:
        M = ((M + M.abs()) / 2).sqrt()

    return M


def wasserstein(M, sqrt):
    if sqrt:
        M = M.abs().sqrt()
    emd = ot.emd2([], [], M.numpy())

    return emd


class Score_knn:
    acc = 0
    acc_real = 0
    acc_fake = 0
    precision = 0
    recall = 0
    tp = 0
    fp = 0
    fn = 0
    ft = 0


def knn(Mxx, Mxy, Myy, k, sqrt):
    n0 = Mxx.size(0)
    n1 = Myy.size(0)
    label = torch.cat((torch.ones(n0), torch.zeros(n1)))
    M = torch.cat((torch.cat((Mxx, Mxy), 1), torch.cat(
        (Mxy.transpose(0, 1), Myy), 1)), 0)
    if sqrt:
        M = M.abs().sqrt()
    INFINITY = float('inf')
    val, idx = (M + torch.diag(INFINITY * torch.ones(n0 + n1))
                ).topk(k, 0, False)

    count = torch.zeros(n0 + n1)
    for i in range(0, k):
        count = count + label.index_select(0, idx[i])
    pred = torch.ge(count, (float(k) / 2) * torch.ones(n0 + n1)).float()

    s = Score_knn()
    s.tp = (pred * label).sum()
    s.fp = (pred * (1 - label)).sum()
    s.fn = ((1 - pred) * label).sum()
    s.tn = ((1 - pred) * (1 - label)).sum()
    s.precision = s.tp / (s.tp + s.fp + 1e-10)
    s.recall = s.tp / (s.tp + s.fn + 1e-10)
    s.acc_t = s.tp / (s.tp + s.fn)
    s.acc_f = s.tn / (s.tn + s.fp)
    s.acc = torch.eq(label, pred).float().mean()
    s.k = k

    return s


def mmd(Mxx, Mxy, Myy, sigma):
    scale = Mxx.mean()
    Mxx = torch.exp(-Mxx / (scale * 2 * sigma * sigma))
    Mxy = torch.exp(-Mxy / (scale * 2 * sigma * sigma))
    Myy = torch.exp(-Myy / (scale * 2 * sigma * sigma))
    #mean = Mxx.mean() + Myy.mean() - 2 * Mxy.mean()
    #mmd = math.sqrt((mean + math.fabs(mean))/2)
    mmd = math.sqrt(Mxx.mean() + Myy.mean() - 2 * Mxy.mean())
    return mmd


def ent(M, epsilon):
    n0 = M.size(0)
    n1 = M.size(1)
    neighbors = M.lt(epsilon).float()
    sums = neighbors.sum(0).repeat(n0, 1)
    sums[sums.eq(0)] = 1
    neighbors = neighbors.div(sums)
    probs = neighbors.sum(1) / n1
    rem = 1 - probs.sum()
    if rem < 0:
        rem = 0
    probs = torch.cat((probs, rem * torch.ones(1)), 0)
    e = {}
    e['probs'] = probs
    probs = probs[probs.gt(0)]
    e['ent'] = -probs.mul(probs.log()).sum()

    return e


def entropy_score(X, Y, epsilons):
    Mxy = distance(X, Y, False)
    scores = []
    for epsilon in epsilons:
        scores.append(ent(Mxy.t(), epsilon))

    return scores


eps = 1e-20


def inception_score(X):
    kl = X * ((X + eps).log() - (X.mean(0) + eps).log().expand_as(X))
    score = np.exp(kl.sum(1).mean())

    return score


def mode_score(X, Y):
    kl1 = X * ((X + eps).log() - (X.mean(0) + eps).log().expand_as(X))
    kl2 = X.mean(0) * ((X.mean(0) + eps).log() - (Y.mean(0) + eps).log())
    score = np.exp(kl1.sum(1).mean() - kl2.sum())

    return score


def fid(X, Y):
    m = X.mean(0)
    m_w = Y.mean(0)
    X_np = X.numpy()
    Y_np = Y.numpy()

    C = np.cov(X_np.transpose())
    C_w = np.cov(Y_np.transpose())
    C_C_w_sqrt = linalg.sqrtm(C.dot(C_w), True).real

    score = m.dot(m) + m_w.dot(m_w) - 2 * m_w.dot(m) + \
            np.trace(C + C_w - 2 * C_C_w_sqrt)
    return np.sqrt(score)


class Score:
    emd = 0
    mmd = 0
    knn = None

def distance2gtdist(features):
    n_sample = features.size(0)



def compute_score(real, fake, k=1, sigma=1, sqrt=True):
    Mxx = distance(real, real, False)
    Mxy = distance(real, fake, False)
    Myy = distance(fake, fake, False)

    s = Score()
    s.emd = wasserstein(Mxy, sqrt)
    s.mmd = mmd(Mxx, Mxy, Myy, sigma)
    s.knn = knn(Mxx, Mxy, Myy, k, sqrt)

    return s


def compute_score_raw(netG, epoch, gpu_mode=False):

    saveFolder_r = GLV.metric_out_dir + '/' + GLV.dataset + '/real/'
    saveFolder_f = GLV.metric_out_dir + '/' + GLV.dataset + '/fake/'
    score = np.zeros(((GLV.class_num, 5))) # emd, mmd, 1-NN.acc, 1-NN.acc_r, 1-NN.acc_f, distance_gt
    #for i in range(class_num):
    #    score[i] = []

    for label in range(GLV.class_num):
        feature_r = sampleTrue(label, GLV.dataset, GLV.img_size, GLV.dataset_path,
                               GLV.metric_sample_size, GLV.batch_size,
                               saveFolder_r + 'epoch' + str(epoch).zfill(4) + '/', False)
        feature_f = sampleFake(label, GLV.dataset, netG,  GLV.z_dim, GLV.metric_sample_size, GLV.batch_size,
                               saveFolder_f + 'epoch' + str(epoch).zfill(4) + '/',
                               GLV.class_num, GLV.gpu_mode, False)

        # 只取家具颜色
        #feature_r = feature_r[:, :fl_furcolor]
        #feature_f = feature_f[:, :fl_furcolor]

        #Mxx = distance(feature_r, feature_r, False)
        #Mxy = distance(feature_r, feature_f, False, filter=GLV.filter_distance)
        #Myy = distance(feature_f, feature_f, False)

        Mxx = distanceL1(feature_r, feature_r)
        Mxy = distanceL1(feature_r, feature_f)
        Myy = distanceL1(feature_f, feature_f)

        score[label, 0] = wasserstein(Mxy/Mxy.max(), True)
        score[label, 1] = mmd(Mxx, Mxy, Myy, 1)
        knn_d = knn(Mxx, Mxy, Myy, GLV.metric_knn_k, True)
        score[label, 2] = knn_d.acc
        score[label, 3] = knn_d.acc_t
        score[label, 4] = knn_d.acc_f

        #score[label] = [emd, mmd, knn]
        #feature_r = FloatTensor(feature_r)
        #feature_f = FloatTensor(feature_f)

        # 4 feature spaces and 7 scores + incep + modescore + fid
        #score[label] = compute_score(feature_r, feature_f, 1, 1, True)
    return score

def compute_distance_gt(netG, epoch, gpu_mode=False):
    score = 0.0
    fur_dist = {}
    dec_dist = {}
    for f in GLV.furniture_types:
        fur_dist[f] = np.zeros((GLV.cluster_num, GLV.describe_word_num))
    for d in GLV.smallobj_types:
        dec_dist[d] = np.zeros(GLV.describe_word_num)

    for label in range(GLV.class_num):
        feature_f = sampleFake(label, GLV.dataset, netG, GLV.z_dim, GLV.metric_dist_size, GLV.batch_size,
                               '/', GLV.class_num, GLV.gpu_mode, False)

        if GLV.gpu_mode:
            data = feature_f.cpu().data.numpy()
        else:
            data = feature_f.data.numpy()

      #  converted_data = np.zeros((len(data), len(data[0])))

        fc, sm = data[:, :fl_furcolor], data[:, fl_furcolor:feature_size]

        fc = fc.reshape(len(data), -1, GLV.cluster_num)
        cluster_index = fc.argmax(axis=2)  # 取出值最大的cluster index
        cluster_index = torch.LongTensor(cluster_index.reshape(len(fc), len(fc[0]), 1))
        fi = torch.zeros(fc.shape)  # 其余位置均为0
        fi.scatter_(2, cluster_index, 1)  # 在cluster位置为1
        # sm = np.float32(sm > 0.5)  ## used to be > 0 when using tanh

        if label == 0 or label == 1:
            fi[:, GLV.living_only_idx, :] = 0
        else:
            fi[:, GLV.bedroom_only_idx, :] = 0

        sm = sm.reshape(len(data), -1, 2)
        sm_index = sm.argmax(axis=2)  # 取出值最大的cluster index
        sm_index = torch.LongTensor(sm_index.reshape(len(sm), len(sm[0]), 1))
        smi = torch.zeros(sm.shape)  # 其余位置均为0
        smi.scatter_(2, sm_index, 1)  # 在cluster位置为1

        if gpu_mode:
            fi = fi.cpu().data.numpy()
            smi = smi.cpu().data.numpy()
        else:
            fi = fi.data.numpy()
            smi = smi.data.numpy()

        fi = fi.sum(axis=0) / GLV.metric_dist_size
        smi = smi.sum(axis=0) / GLV.metric_dist_size

        for i in range(GLV.fur_type_num):
            fur_dist[GLV.furniture_types[i]][:,label] = fi[i]

        for i in range(GLV.fl_smallobj):
            dec_dist[GLV.smallobj_types[i]][label] = smi[i][0]

    for f in GLV.furniture_types:
        score += ((GLV.fur_dist_gt[f] - fur_dist[f])**2).sum()
    #for d in GLV.smallobj_types:
    #    score += ((GLV.dec_dist_gt[d] - dec_dist[d])**2).sum()

    return score

def calculate_dist_distance(real, fake):
    score = 0.0
    number = len(real)
    real = real.sum(dim=0)/number
    fake = fake.sum(dim=0)/number
    score = ((real - fake) ** 2).sum()
    return score.type(torch.FloatTensor)


def probability(data):
    return data.sum() / float(data.shape[0])


def probabilities_by_dimension(data):
    return np.array([probability(data[:, j]) for j in range(data.shape[1])])


def mse_probabilities_by_dimension(data_x, data_y):
    p_x_by_dimension = probabilities_by_dimension(data_x)
    p_y_by_dimension = probabilities_by_dimension(data_y)
    mse = mean_squared_error(p_x_by_dimension, p_y_by_dimension)
    return p_x_by_dimension, p_y_by_dimension, mse

def calculate_mse_prob_dim(G):
    score = 0.0

    for label in range(GLV.class_num):
        dataloader = DataLoader(SceneConditional(label),
                                batch_size=GLV.batch_size,
                                shuffle=True)
        mse = 0.0
        for i, (features, labels) in enumerate(dataloader):
            batch_size = len(features)
            real = features
            noise = torch.rand((batch_size, GLV.z_dim))
            sample_y_ = (torch.ones(batch_size, 1) * label).type(torch.LongTensor)  # 随机y
            y_vec_ = torch.zeros(batch_size, GLV.class_num)
            y_vec_.scatter_(1, sample_y_.view(batch_size, 1), 1)
            if GLV.gpu_mode:
                noise, y_vec_ = noise.cuda(), y_vec_.cuda()

            fake = G(noise, y_vec_, training=False)

            # 新加的
            fake = utils.filter_types(fake, labels)



            _, _, err = mse_probabilities_by_dimension(real, fake)
            mse += err
        score += mse
    score /= GLV.class_num

    return score

