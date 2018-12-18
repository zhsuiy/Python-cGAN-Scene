import utils
import torch, time, os
import numpy as np
import torch.nn as nn
import torch.optim as optim
from utils import normal_init, export_features, convert2softmax, convert2gumbelsoftmax
import torch.nn.functional as F
import metric
from global_var import GLV

cuda = True if torch.cuda.is_available() else False

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

class Generator(nn.Module):
    """
        Simple Generator w/ MLP
    """
    def __init__(self, noise_dim=100, output_dim=1, condition_dim=10):
        super(Generator, self).__init__()
        self.noise_dim = noise_dim
        self.output_dim = output_dim
        self.condition_dim = condition_dim

        l1 = GLV.net_complex  # 256
        l2 = l1 * 2  # 512
        l3 = l2 * 2  # 1024

        self.fc1_1 = nn.Linear(self.noise_dim, l1)
        self.fc1_1_bn = nn.BatchNorm1d(l1)
        self.fc1_2 = nn.Linear(self.condition_dim, l1)
        self.fc1_2_bn = nn.BatchNorm1d(l1)
        self.fc2 = nn.Linear(l2, l2)  # 把z和y映射后的向量concat在一起
        self.fc2_bn = nn.BatchNorm1d(l2)
        self.fc3 = nn.Linear(l2, l3)
        self.fc3_bn = nn.BatchNorm1d(l3)
        self.fc4 = nn.Linear(l3, self.output_dim)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, noise, label):
        x = F.relu(self.fc1_1_bn(self.fc1_1(noise)))
        y = F.relu(self.fc1_2_bn(self.fc1_2(label)))
        x = torch.cat([x, y], 1)
        x = F.relu(self.fc2_bn(self.fc2(x)))
        x = F.relu(self.fc3_bn(self.fc3(x)))

        if GLV.dataset == 'scene':
            #x = torch.sigmoid(self.fc4(x))
            #x = convert2softmax(x)
            x = convert2gumbelsoftmax(self.fc4(x), temperature=GLV.gumbel_temp)
        else:
            x = torch.tanh(self.fc4(x))

        return x

class Discriminator(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
    def __init__(self, input_dim=1, output_dim=1, condition_dim=10):
        super(Discriminator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.condition_dim = condition_dim

        l1 = GLV.net_complex  # 256
        l2 = l1 * 2  # 512
        l3 = l2 * 2  # 1024
        l4 = l3 * 2  # 2048

        self.fc1_1 = nn.Linear(self.input_dim, l3)
        self.fc1_2 = nn.Linear(self.condition_dim, l3)
        self.fc2 = nn.Linear(l4, l2) # concat two
        self.fc2_bn = nn.BatchNorm1d(l2)
        self.fc3 = nn.Linear(l2, l1)
        self.fc3_bn = nn.BatchNorm1d(l1)
        self.fc4 = nn.Linear(l1, self.output_dim)
        #utils.initialize_weights(self)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, input, label):
        x = F.leaky_relu(self.fc1_1(input), 0.2)
        y = F.leaky_relu(self.fc1_2(label), 0.2)
        x = torch.cat([x, y], 1)
        x = F.leaky_relu(self.fc2_bn(self.fc2(x)), 0.2)
        x = F.leaky_relu(self.fc3_bn(self.fc3(x)), 0.2)
        x = torch.sigmoid(self.fc4(x))
        return x


class CGAN(object):
    def __init__(self, args):

        self.n_row = 10
        self.sample_num = GLV.class_num * self.n_row

        # networks init
        self.G = Generator(noise_dim=GLV.z_dim, output_dim=GLV.input_dim, condition_dim=GLV.class_num)
        self.D = Discriminator(input_dim=GLV.input_dim, output_dim=1, condition_dim=GLV.class_num)

        self.G.weight_init(mean=0, std=0.02)
        self.D.weight_init(mean=0, std=0.02)
        #self.G_optimizer = optim.Adam(self.G.parameters())
        #self.D_optimizer = optim.Adam(self.D.parameters())

        if GLV.gpu_mode:
            self.G.cuda()
            self.D.cuda()
            self.BCE_loss = nn.BCELoss().cuda()
            self.L1_loss = nn.L1Loss().cuda()
        else:
            self.BCE_loss = nn.BCELoss()
            self.L1_loss = nn.L1Loss()

        self.G_optimizer = optim.Adam(self.G.parameters(), lr=args.lrG, betas=(args.beta1, args.beta2))
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=args.lrD, betas=(args.beta1, args.beta2))
        #self.D_optimizer = optim.SGD(self.D.parameters(), lr=0.1)

        print('---------- Networks architecture -------------')
        utils.print_network(self.G)
        utils.print_network(self.D)
        print('-----------------------------------------------')

        # 固定的噪声+y
        # fixed noise & condition
        self.sample_z_ = torch.zeros((self.sample_num, GLV.z_dim))
        for i in range(self.n_row):
            self.sample_z_[i * GLV.class_num] = torch.rand(1, GLV.z_dim)  # 均匀噪声
            #self.sample_z_[i * self.class_num] = torch.randn(1, self.z_dim) # 高斯噪声
            for j in range(1, GLV.class_num):
                self.sample_z_[i * GLV.class_num + j] = self.sample_z_[i * GLV.class_num]

        temp = torch.zeros((GLV.class_num, 1))
        for i in range(GLV.class_num):
            temp[i, 0] = i

        temp_y = torch.zeros((self.sample_num, 1))
        for i in range(self.n_row):
            temp_y[i * GLV.class_num: (i + 1) * GLV.class_num] = temp

        self.sample_y_ = torch.zeros((self.sample_num, GLV.class_num)).scatter_(1, temp_y.type(torch.LongTensor), 1)
        if GLV.gpu_mode:
            self.sample_z_, self.sample_y_ = self.sample_z_.cuda(), self.sample_y_.cuda()

        # results save folder
        if not os.path.isdir('MNIST_cGAN_results'):
            os.mkdir('MNIST_cGAN_results')
        if not os.path.isdir('MNIST_cGAN_results/Fixed_results'):
            os.mkdir('MNIST_cGAN_results/Fixed_results')

    def train(self):
        self.train_hist = {}
        self.train_hist['D_loss'] = [] # 记录D的Loss下降情况
        self.train_hist['G_loss'] = [] # 记录G的Loss下降情况
        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_time'] = []
        self.train_hist['D_prob_fake'] = [] # 记录D判断G所生成的样本的概率
        self.train_hist['D_prob_real'] = []  # 记录D判断G所生成的样本的概率

        # metric
        self.score_tr = []
        #self.D.train()

        print('training start!!')
        start_time = time.time()
        #step = 0  # 控制G和D训练的节奏

        for epoch in range(GLV.epochs):
            D_losses = []
            G_losses = []
            #self.G.train()
            #self.adjust_lrD(self.D_optimizer, epoch)
            if GLV.dataset != 'scene':
                if (epoch + 1) == 30:
                    self.G_optimizer.param_groups[0]['lr'] /= 10
                    self.D_optimizer.param_groups[0]['lr'] /= 10
                    print("learning rate change!")
                if (epoch + 1) == 40:
                    self.G_optimizer.param_groups[0]['lr'] /= 10
                    self.D_optimizer.param_groups[0]['lr'] /= 10
                    print("learning rate change!")

            epoch_start_time = time.time()
            for iter, (x_, y_) in enumerate(GLV.data_loader): # x_是feature, y_是label
                #step += 1
                batch_size = x_.shape[0] # 有可能尾部的size不一样
                x_ = x_.view(-1, GLV.input_dim) # 针对MNIST数据集
                x_ = x_.type(torch.FloatTensor)
                z_ = torch.rand((batch_size, GLV.z_dim))  # 均匀噪声
                #z_ = torch.randn((batch_size, self.z_dim)) # 高斯噪声
                y_vec_ = torch.zeros((batch_size, GLV.class_num)).scatter_(1, y_.type(torch.LongTensor).unsqueeze(1), 1)
                #print (batch_size)
                y_real_, y_fake_ = torch.ones(batch_size, 1), torch.zeros(batch_size, 1)

                if GLV.gpu_mode:
                    x_, z_, y_vec_, y_real_, y_fake_= x_.cuda(), z_.cuda(), y_vec_.cuda(), y_real_.cuda(), y_fake_.cuda()

                #x_.requires_grad = True
                #x_.register_hook(lambda grad: print(grad.mean()))

                # update D
                self.D_optimizer.zero_grad()

                D_real = self.D(x_, y_vec_)
                D_real_loss = self.BCE_loss(D_real, y_real_)

                G_z = self.G(z_, y_vec_) # 用噪声z生成的样本
                D_fake = self.D(G_z, y_vec_)
                D_fake_loss = self.BCE_loss(D_fake, y_fake_)

                D_loss = D_real_loss + D_fake_loss

                D_loss.backward()
                self.D_optimizer.step()

                #if step % 5 == 0:
                if ((iter + 1) % GLV.n_critic) == 0:
                    # update G network
                    self.G_optimizer.zero_grad()
                    z_ = torch.rand((batch_size, GLV.z_dim))  # 均匀噪声
                    #z_ = torch.randn((batch_size, self.z_dim))  # 高斯噪声
                    sample_y_ = (torch.rand(batch_size, 1) * GLV.class_num).type(torch.LongTensor) # 随机y
                    y_vec_ = torch.zeros(batch_size, GLV.class_num)
                    y_vec_.scatter_(1, sample_y_.view(batch_size, 1), 1)
                    if GLV.gpu_mode:
                        z_, y_vec_ = z_.cuda(), y_vec_.cuda()

                    y_vec_.requires_grad = True
                    y_vec_.register_hook(lambda grad: print("G_y_vec: ", grad.mean()))

                    G_z = self.G(z_, y_vec_)
                    D_fake = self.D(G_z, y_vec_)
                    G_loss = self.BCE_loss(D_fake, y_real_)
                    if GLV.dataset == 'scene':
                        zeros = torch.zeros(G_z.shape)
                        if GLV.gpu_mode:
                            zeros = zeros.cuda()
                        G_loss += GLV.gamma_sparsity*self.L1_loss(G_z, zeros)
                    #self.train_hist['G_loss'].append(G_loss.item())
                    #self.train_hist['D_loss'].append(D_loss.item())
                    #if GLV.gpu_mode:
                    #    self.train_hist['D_prob_fake'].append(D_fake.cpu().data.numpy().mean())
                    #    self.train_hist['D_prob_real'].append(D_real.cpu().data.numpy().mean())
                    #else:
                    #    self.train_hist['D_prob_fake'].append(D_fake.data.numpy().mean())
                    #    self.train_hist['D_prob_real'].append(D_real.data.numpy().mean())

                    G_loss.backward()
                    self.G_optimizer.step()

                #if iter % 400 == 0:
                #    self.G.eval()
                #    s = metric.compute_score_raw(self.G, epoch)
                #    self.score_tr.append(s)
                #    self.G.train()

            per_epoch_ptime = time.time() - epoch_start_time
            print(
                'Epoch: {}/{}, D Loss: {}, G Loss: {}, Epoch time: {}'.format(epoch, GLV.epochs, D_loss.item(),
                                                                            G_loss.item(), per_epoch_ptime))
                #if ((iter + 1) % 100) == 0:
                #    print("Epoch: [%2d] [%4d/%4d] D_loss: %.8f, G_loss: %.8f" %
                #          (
                #          (epoch + 1), (iter + 1), self.data_loader.dataset.__len__() // self.batch_size, D_loss.item(),
                #          G_loss.item()))
                #batches_done = epoch * len(self.data_loader) + iter
                #if batches_done % 400 == 0:
                #    with torch.no_grad():
                #        self.sample_image(n_row=10, batches_done=batches_done)
            with torch.no_grad():
                self.G.eval()
                #self.sample_image(n_row=10, batches_done=epoch)

                s = metric.compute_score_raw(self.G, epoch)
                self.score_tr.append(s)

                self.train_hist['G_loss'].append(G_loss.item())
                self.train_hist['D_loss'].append(D_loss.item())
                if GLV.gpu_mode:
                    self.train_hist['D_prob_fake'].append(torch.sigmoid(D_fake).cpu().data.numpy().mean())
                    self.train_hist['D_prob_real'].append(torch.sigmoid(D_real).cpu().data.numpy().mean())
                else:
                    self.train_hist['D_prob_fake'].append(torch.sigmoid(D_fake).data.numpy().mean())
                    self.train_hist['D_prob_real'].append(torch.sigmoid(D_real).data.numpy().mean())

                utils.visualize_results(G=self.G, sample_num=self.sample_num,
                                       epoch=(epoch + 1), n_row=self.n_row, sample_z_=self.sample_z_,
                                       sample_y_=self.sample_y_, fix=True)
                if (GLV.dataset == 'scene'):
                    utils.save_features(G=self.G, sample_num=self.sample_num, sample_z_=self.sample_z_,
                                       sample_y_=self.sample_y_, epoch=(epoch + 1),
                                       fix=True, process=True)
                    #self.save_features((epoch + 1), fix=True, process=False)
                self.G.train()

            self.train_hist['per_epoch_time'].append(per_epoch_ptime)

        self.train_hist['total_time'].append(time.time() - start_time)
        print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
                                                                        GLV.epochs, self.train_hist['total_time'][0]))
        print("Training finish!... save training results")

        utils.save(self.G, self.D, self.train_hist, self.score_tr) # 保存模型

        utils.generate_animation()

        #

        # plot statistics
        utils.loss_plot(self.train_hist)
        utils.D_Prob_G_plot(self.train_hist)
        utils.metric_plot(np.array(self.score_tr))
        utils.visualize_results(G=self.G, sample_num=self.sample_num,
                                epoch=(epoch + 1), n_row=self.n_row, sample_z_=self.sample_z_,
                                sample_y_=self.sample_y_, fix=True, final=True)
        if (GLV.dataset == 'scene'):
            utils.save_features(G=self.G, sample_num=self.sample_num, sample_z_=self.sample_z_,
                                sample_y_=self.sample_y_, epoch=(epoch + 1),
                                fix=True, process=True, final=True)
            utils.save_features(G=self.G, sample_num=self.sample_num, sample_z_=self.sample_z_,
                                sample_y_=self.sample_y_, epoch=(epoch + 1),
                                fix=True, process=False, final=True)