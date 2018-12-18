import utils
import torch, time, os, pickle
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.autograd import Variable
from torchvision.utils import save_image
from utils import normal_init, export_features
import torch.nn.functional as F
import metric

cuda = True if torch.cuda.is_available() else False

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

z_dim = 100
h_dim = 128

class Generator(nn.Module):
    """
        Simple Generator w/ MLP
    """
    def __init__(self, noise_dim=100, output_dim=1, condition_dim=10):
        super(Generator, self).__init__()
        self.noise_dim = noise_dim
        self.output_dim = output_dim
        self.condition_dim = condition_dim

        self.fc1_1 = nn.Linear(self.noise_dim, 256)
        self.fc1_1_bn = nn.BatchNorm1d(256)
        self.fc1_2 = nn.Linear(self.condition_dim, 256)
        self.fc1_2_bn = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(512, 512)  # 把z和y映射后的向量concat在一起
        self.fc2_bn = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc3_bn = nn.BatchNorm1d(1024)
        self.fc4 = nn.Linear(1024, self.output_dim)

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

        self.fc1_1 = nn.Linear(self.input_dim, 1024)
        self.fc1_2 = nn.Linear(self.condition_dim, 1024)
        self.fc2 = nn.Linear(2048, 512) # concat two
        self.fc2_bn = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 256)
        self.fc3_bn = nn.BatchNorm1d(256)
        self.fc4 = nn.Linear(256, self.output_dim)
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
        self.dataset_path = args.dataset_path
        self.epochs = args.epoch
        self.batch_size = args.batch_size
        self.z_dim = args.z_dim
        self.class_num = args.class_num
        self.save_dir = args.save_dir
        self.result_dir = args.result_dir
        self.log_dir = args.log_dir
        self.model_name = args.gan_type
        self.dataset = args.dataset

        self.gpu_mode = args.gpu_mode
        self.feature_path = self.dataset_path+'/features.txt'
        self.label_path = self.dataset_path+'/labels.txt'

        self.img_size = 28 # for MNIST dataset


        self.n_row = 10
        self.sample_num = self.class_num * self.n_row
        self.n_critic = args.critic

        if self.dataset == 'mnist':
            self.data_loader = DataLoader(datasets.MNIST('./data/mnist', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.Resize(self.img_size),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                       ])), batch_size=self.batch_size, shuffle=True)
            self.input_dim = self.img_size * self.img_size
            self.H = self.W = self.img_size
        elif self.dataset == 'scene':
            self.data_loader = DataLoader(utils.SceneFurColorSmallObjDataset(self.feature_path, self.label_path),
                                batch_size=self.batch_size,
                                shuffle=True)
            data = self.data_loader.__iter__().__next__()[0]
            self.input_dim = data.shape[1] # feature dimension
            self.H = 20
            self.W = int(np.floor(self.input_dim/self.H))

        else:
            raise Exception("[!] There is no dataset of " + self.dataset)




        # networks init
        self.G = Generator(noise_dim=self.z_dim, output_dim=self.input_dim, condition_dim=self.class_num)
        self.D = Discriminator(input_dim=self.input_dim, output_dim=1, condition_dim=self.class_num)

        self.G.weight_init(mean=0, std=0.02)
        self.D.weight_init(mean=0, std=0.02)
        #self.G_optimizer = optim.Adam(self.G.parameters())
        #self.D_optimizer = optim.Adam(self.D.parameters())

        if self.gpu_mode:
            self.G.cuda()
            self.D.cuda()
            self.BCE_loss = nn.BCELoss().cuda()
        else:
            self.BCE_loss = nn.BCELoss()

        self.G_optimizer = optim.Adam(self.G.parameters(), lr=args.lrG, betas=(args.beta1, args.beta2))
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=args.lrD, betas=(args.beta1, args.beta2))
        #self.D_optimizer = optim.SGD(self.D.parameters(), lr=0.1)

        print('---------- Networks architecture -------------')
        utils.print_network(self.G)
        utils.print_network(self.D)
        print('-----------------------------------------------')

        # 固定的噪声+y
        # fixed noise & condition
        self.sample_z_ = torch.zeros((self.sample_num, self.z_dim))
        for i in range(self.n_row):
            self.sample_z_[i * self.class_num] = torch.rand(1, self.z_dim)  # 均匀噪声
            #self.sample_z_[i * self.class_num] = torch.randn(1, self.z_dim) # 高斯噪声
            for j in range(1, self.class_num):
                self.sample_z_[i * self.class_num + j] = self.sample_z_[i * self.class_num]

        temp = torch.zeros((self.class_num, 1))
        for i in range(self.class_num):
            temp[i, 0] = i

        temp_y = torch.zeros((self.sample_num, 1))
        for i in range(self.n_row):
            temp_y[i * self.class_num: (i + 1) * self.class_num] = temp

        self.sample_y_ = torch.zeros((self.sample_num, self.class_num)).scatter_(1, temp_y.type(torch.LongTensor), 1)
        if self.gpu_mode:
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

        #self.D.train()
        print('training start!!')
        start_time = time.time()
        #step = 0  # 控制G和D训练的节奏
        for epoch in range(self.epochs):
            D_losses = []
            G_losses = []
            #self.G.train()
            #self.adjust_lrD(self.D_optimizer, epoch)
            if (epoch + 1) == 30:
                self.G_optimizer.param_groups[0]['lr'] /= 10
                self.D_optimizer.param_groups[0]['lr'] /= 10
                print("learning rate change!")
            if (epoch + 1) == 40:
                self.G_optimizer.param_groups[0]['lr'] /= 10
                self.D_optimizer.param_groups[0]['lr'] /= 10
                print("learning rate change!")

            epoch_start_time = time.time()
            for iter, (x_, y_) in enumerate(self.data_loader): # x_是feature, y_是label
                #step += 1
                batch_size = x_.shape[0] # 有可能尾部的size不一样
                x_ = x_.view(-1, self.input_dim) # 针对MNIST数据集
                x_ = x_.type(torch.FloatTensor)
                z_ = torch.rand((batch_size, self.z_dim))  # 均匀噪声
                #z_ = torch.randn((batch_size, self.z_dim)) # 高斯噪声
                y_vec_ = torch.zeros((batch_size, self.class_num)).scatter_(1, y_.type(torch.LongTensor).unsqueeze(1), 1)
                #print (batch_size)
                y_real_, y_fake_ = torch.ones(batch_size, 1), torch.zeros(batch_size, 1)

                if self.gpu_mode:
                    x_, z_, y_vec_, y_real_, y_fake_= x_.cuda(), z_.cuda(), y_vec_.cuda(), y_real_.cuda(), y_fake_.cuda()

                # update D
                self.D_optimizer.zero_grad()

                D_real = self.D(x_, y_vec_)
                D_real_loss = self.BCE_loss(D_real, y_real_)

                G_z = self.G(z_, y_vec_) # 用噪声z生成的样本
                D_fake = self.D(G_z, y_vec_)
                D_fake_loss = self.BCE_loss(D_fake, y_fake_)

                D_loss = D_real_loss + D_fake_loss
                self.train_hist['D_loss'].append(D_loss.item())

                D_loss.backward()
                self.D_optimizer.step()

                #if step % 5 == 0:
                if ((iter + 1) % self.n_critic) == 0:
                    # update G network
                    self.G_optimizer.zero_grad()
                    z_ = torch.rand((batch_size, self.z_dim))  # 均匀噪声
                    #z_ = torch.randn((batch_size, self.z_dim))  # 高斯噪声
                    sample_y_ = (torch.rand(batch_size, 1) * self.class_num).type(torch.LongTensor) # 随机y
                    y_vec_ = torch.zeros(batch_size, self.class_num)
                    y_vec_.scatter_(1, sample_y_.view(batch_size, 1), 1)
                    if self.gpu_mode:
                        z_, y_vec_ = z_.cuda(), y_vec_.cuda()
                    G_z = self.G(z_, y_vec_)
                    D_fake = self.D(G_z, y_vec_)
                    G_loss = self.BCE_loss(D_fake, y_real_)
                    self.train_hist['G_loss'].append(G_loss.item())
                    if self.gpu_mode:
                        self.train_hist['D_prob_fake'].append(D_fake.cpu().data.numpy().mean())
                        self.train_hist['D_prob_real'].append(D_real.cpu().data.numpy().mean())
                    else:
                        self.train_hist['D_prob_fake'].append(D_fake.data.numpy().mean())
                        self.train_hist['D_prob_real'].append(D_real.data.numpy().mean())

                    G_loss.backward()
                    self.G_optimizer.step()

            per_epoch_ptime = time.time() - epoch_start_time
            print(
                'Epoch: {}/{}, D Loss: {}, G Loss: {}, Epoch time: {}'.format(epoch, self.epochs, D_loss.item(),
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
                self.visualize_results((epoch + 1), n_row=self.n_row, fix=True)
                if (self.dataset == 'scene'):
                    self.save_features((epoch + 1), fix=True, process=True)
                    #self.save_features((epoch + 1), fix=True, process=False)
                self.G.train()


            self.train_hist['per_epoch_time'].append(per_epoch_ptime)

            #with torch.no_grad():
            #    self.visualize_results((epoch + 1),fix=False)

            #self.G.eval()
            #img = get_sample_image(G)
            #scipy.misc.imsave('sample/{}_epoch_{}_type1.jpg'.format('cGAN', epoch), img)
            #G.train()

        self.train_hist['total_time'].append(time.time() - start_time)
        print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
                                                                        self.epochs, self.train_hist['total_time'][0]))
        print("Training finish!... save training results")

        self.save() # 保存模型

        utils.generate_animation(self.result_dir + '/' + self.dataset + '/' + self.model_name + '/' + self.model_name,
                                 self.epochs)
        #utils.generate_anmation('./images/', self.epochs)
        utils.loss_plot(self.train_hist, os.path.join(self.save_dir, self.dataset, self.model_name), self.model_name)
        utils.D_Prob_G_plot(self.train_hist, os.path.join(self.save_dir, self.dataset, self.model_name), self.model_name)

    def adjust_lrD(self, optimizer, epoch):
        lr = 0.1 * (0.1 ** (epoch // 10))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


    def sample_image(self, n_row, epoch):
        """Saves a grid of generated digits ranging from 0 to n_classes"""
        #self.G.eval()
        # Sample noise
        z = torch.rand((n_row ** 2, self.z_dim))  # 均匀噪声
        #z = torch.randn((n_row**2, self.z_dim))  # 高斯噪声
        if self.gpu_mode:
            z = z.cuda()
        #z = Variable(FloatTensor(np.random.normal(0, 1, (n_row ** 2, self.z_dim))))
        # Get labels ranging from 0 to n_classes for n rows
        #labels = np.array([num for _ in range(n_row) for num in range(n_row)])
        #labels = Variable(LongTensor(labels))
        #gen_imgs = self.G(z, self.sample_y_)
        gen_imgs = self.G(self.sample_z_, self.sample_y_)
        if self.gpu_mode:
            data = gen_imgs.cpu().data
        else:
            data = gen_imgs.data

        data = FloatTensor(data.numpy().reshape(self.sample_num, -1, self.img_size, self.img_size))
        save_image(data, 'images/%d.png' % epoch, nrow=n_row, normalize=True)

    def save_features(self, epoch, fix=True, process=True):
        path = self.result_dir + '/' + self.dataset + '/' + self.model_name
        if not os.path.exists(path):
            os.makedirs(path)

        # Sample noise
        z = torch.rand((self.sample_num, self.z_dim))   # 均匀噪声
        # z = torch.randn((n_row**2, self.z_dim))  # 高斯噪声
        if self.gpu_mode:
            z = z.cuda()
        # z = Variable(FloatTensor(np.random.normal(0, 1, (n_row ** 2, self.z_dim))))
        # Get labels ranging from 0 to n_classes for n rows
        # labels = np.array([num for _ in range(n_row) for num in range(n_row)])
        # labels = Variable(LongTensor(labels))
        # gen_imgs = self.G(z, self.sample_y_)
        if fix:
            features = self.G(self.sample_z_, self.sample_y_)
        else:
            # sample_y_ = self.sample_y_
            # sample_y_ = torch.zeros(self.batch_size, self.class_num).scatter_(1,
            #                                                                  torch.randint(0, self.class_num - 1, (
            #                                                                  self.batch_size, 1)).type(
            #                                                                  torch.LongTensor), 1)
            # if self.gpu_mode:
            #    sample_y_ = sample_y_.cuda()
            features = self.G(z, self.sample_y_)

        if self.gpu_mode:
            data = features.cpu().data
        else:
            data = features.data


        if process:
            export_features(data.numpy(),
                            path + '/' + self.model_name + '_epoch%03d-processed' % epoch + '.txt',
                            process=process)
        else:
            export_features(data.numpy(),
                            path + '/' + self.model_name + '_epoch%03d-org' % epoch + '.txt',
                            process=process)



    def visualize_results(self, epoch, n_row, fix=True):
        #self.G.eval()

        if not os.path.exists(self.result_dir + '/' + self.dataset + '/' + self.model_name):
            os.makedirs(self.result_dir + '/' + self.dataset + '/' + self.model_name)

        # Sample noise
        z = torch.rand((n_row ** 2, self.z_dim))  # 均匀噪声
        # z = torch.randn((n_row**2, self.z_dim))  # 高斯噪声
        if self.gpu_mode:
            z = z.cuda()
        # z = Variable(FloatTensor(np.random.normal(0, 1, (n_row ** 2, self.z_dim))))
        # Get labels ranging from 0 to n_classes for n rows
        # labels = np.array([num for _ in range(n_row) for num in range(n_row)])
        # labels = Variable(LongTensor(labels))
        # gen_imgs = self.G(z, self.sample_y_)
        if fix:
            gen_imgs = self.G(self.sample_z_, self.sample_y_)
        else:
            #sample_y_ = self.sample_y_
            # sample_y_ = torch.zeros(self.batch_size, self.class_num).scatter_(1,
            #                                                                  torch.randint(0, self.class_num - 1, (
            #                                                                  self.batch_size, 1)).type(
            #                                                                  torch.LongTensor), 1)
            #if self.gpu_mode:
            #    sample_y_ = sample_y_.cuda()
            gen_imgs = self.G(z, self.sample_y_)

        if self.gpu_mode:
            data = gen_imgs.cpu().data
        else:
            data = gen_imgs.data

        data = FloatTensor(data.numpy().reshape(self.sample_num, -1, self.H, self.W))
        #data = (data + 1) / 2
        save_image(data,
                   self.result_dir + '/' + self.dataset + '/' + self.model_name + '/' + self.model_name + '_epoch%03d' % epoch + '.png',
                    nrow=n_row, normalize=True)

    def save(self):
        save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        torch.save(self.G.state_dict(), os.path.join(save_dir, self.model_name + '_G.pkl'))
        torch.save(self.D.state_dict(), os.path.join(save_dir, self.model_name + '_D.pkl'))

        with open(os.path.join(save_dir, self.model_name + '_history.pkl'), 'wb') as f:
            pickle.dump(self.train_hist, f)

    def load(self):
        save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)

        self.G.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_G.pkl')))
        self.D.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_D.pkl')))