class MyGlobal:
    def __init__(self):
        self.mnist_data = {}
        self.scene_data = {}
        self.scene_data_fur_indices = None  # 用于记录真样本中有哪些家具出现
        self.B = [0]
        # 'FloorLamp', 'FloorProxy'
        self.furniture_types = ['TvCabinet', 'TV', 'Sofa', 'SingleSofa', 'SofaPillow',
                           'CoffeeTable', 'Floor', 'Wall', 'Bed', 'Closet', 'Cabinet',
                           'NightTable', 'WallPhoto', 'Dresser', 'BedSheet', 'BedPillow', 'Window',
                           'Curtain', 'Desk', 'Chair', 'Carpet', 'SideTable']
        #self.furniture_types = ['Floor', 'Wall', 'Bed', 'Closet', 'Cabinet',
        #                   'NightTable', 'WallPhoto', 'Dresser', 'BedSheet', 'BedPillow', 'Window',
        #                   'Curtain', 'Desk', 'Chair', 'Carpet', 'SideTable']
        # 'AeolianBells', 'Angel', 'Gundam', 'ShoppingBag',
        self.smallobj_types = [ 'BalletShoe', 'Backpack', 'Basketball',
                          'Book', 'Candle', 'Cap', 'Clock', 'CrystalBall', 'Cup', 'DigitalScale',
                          'Doll', 'DVD', 'Earphone', 'EiffelTower', 'Figurine', 'Flower',
                          'Football', 'GameHandle', 'GiftBox', 'GirlShoe', 'Goblet',
                          'Handbag', 'Heart', 'Lamp', 'Laptop', 'LoudSpeaker', 'MakeUp', 'Map',
                          'MechanicalModel', 'Mirror', 'MP3', 'Notebook', 'Pad', 'Pen',
                          'Perfume', 'PhotoFrame', 'Plant', 'RemoteControl',
                          'SkaterShoe', 'Skirt', 'StackBoox', 'TeaSet', 'Telephone', 'Tennis',
                          'TennisRacket', 'TissueBox', 'TrashBin', 'Vase']
        #self.smallobj_types = ['Backpack', 'Basketball',
        #                  'Book', 'Cap', 'Clock', 'Cup',
        #                  'Doll', 'Earphone', 'EiffelTower', 'Figurine', 'Flower',
        #                  'Football', 'GameHandle', 'GiftBox', 'GirlShoe',
        #                  'Handbag', 'Heart', 'Lamp', 'Laptop', 'LoudSpeaker', 'MakeUp', 'Map',
        #                  'MechanicalModel', 'Mirror', 'Notebook', 'Pad', 'Pen',
        #                  'PhotoFrame', 'Plant',
        #                  'SkaterShoe', 'StackBoox', 'Telephone', 'Tennis',
        #                  'TennisRacket', 'TissueBox', 'TrashBin', 'Vase']
        self.adjs = ['girl', 'boy', 'romantic', 'modern']
        #self.adjs = ['girl', 'boy']
        # feature length of furniture colors and small objects respectively
        self.fl_smallobj = len(self.smallobj_types)
        self.fur_type_num = len(self.furniture_types)
        self.cluster_num = 15

        self.variable_sizes = []
        for i in range(self.fur_type_num):
            self.variable_sizes.append(self.cluster_num)
        for i in range(self.fl_smallobj):
            self.variable_sizes.append(2)

        self.fl_furcolor = self.fur_type_num * self.cluster_num
        self.feature_size = self.fl_furcolor + self.fl_smallobj*2
        self.describe_word_num = 4
        #self.describe_word_num = 2

        self.bedroom_only = ['Bed', 'BedPillow', 'BedSheet', 'Chair', 'Closet', 'Desk', 'NightTable']
        self.living_only = ['CoffeeTable', 'SingleSofa', 'Sofa', 'SofaPillow', 'TV', 'TvCabinet']

        self.bedroom_only_idx = []
        self.living_only_idx = []

        for f in self.bedroom_only:
            self.bedroom_only_idx.append(self.furniture_types.index(f))
        for f in self.living_only:
            self.living_only_idx.append(self.furniture_types.index(f))

        self.metric_knn_k = 1
        self.metric_sample_size = 20
        self.data_loader = None
        self.data_loader_condition = []
        self.input_dim = 0
        self.H = 0
        self.W = 0
        self.class_num = 10
        self.img_size = 28
        self.batch_size = 128
        self.dataset_path = 'data'
        self.lrG = 0.0002
        self.lrD = 0.0002
        self.beta1 = 0.5
        self.beta2 = 0.999
        self.epochs = 32
        self.z_dim = 100
        self.save_dir = 'models'
        self.result_dir = 'results'
        self.log_dir = 'logs'
        self.model_name = 'CGAN'
        self.dataset = 'mnist'
        self.metric_out_dir = 'metric_output'
        self.gpu_mode = False
        self.n_critic = 1 # the number of iterations of the critic per generator iteration
        self.config = 'config'
        self.batch_mode = False
        self.attention = False
        self.net_complex = 256
        self.gamma_sparsity = 10
        self.dataset_noise = False
        self.filter_distance = False
        self.gumbel_temp = 0.2
        self.output_metric_hard = False
        #self.dist_file = '20181207214033-cluster-portions.txt'
        self.dist_file = 'data/portions.txt'
        self.dec_dist_gt = None
        self.fur_dist_gt = None
        self.metric_dist_size = 500

        self.WGAN_Loss = False
        self.clipping = 0.1

        self.WGAN_GP = False
        self.WGAN_GP_lambda = 10

        # train-test
        self.train_test = ['total', 'train', 'test']
        self.org_gen = ['org', 'gen']

        self.train_set = 'train'
        self.train_type = 'gen'
        self.test_set = 'test'
        self.test_type = 'gen'

        self.filter_types = False

        self.gumbel_softmax = False

        self.softmax_argmax = False
        self.softmax_sample = False


    def get_GLV_str(self):
        config = \
            'model_name: {}\n'.format(GLV.model_name) + \
            'dataset: {}\n'.format(GLV.dataset) + \
            'train_set: {}\n'.format(GLV.train_set) + \
            'train_type: {}\n'.format(GLV.train_type) + \
            'test_set: {}\n'.format(GLV.test_set) + \
            'test_type: {}\n'.format(GLV.test_type) + \
            'filter_types: {}\n'.format(GLV.filter_types) + \
            'gpu_mode: {}\n'.format(GLV.gpu_mode) + \
            'batch_mode: {}\n'.format(GLV.batch_mode) + \
            'lrG: {}\n'.format(GLV.lrG) + \
            'lrD: {}\n'.format(GLV.lrD) + \
            'beta1: {}\n'.format(GLV.beta1) + \
            'beta2: {}\n'.format(GLV.beta2) + \
            'gamma_sparsity: {}\n'.format(GLV.gamma_sparsity) + \
            'metric_knn_k: {}\n'.format(GLV.metric_knn_k ) + \
            'metric_sample_size: {}\n'.format(GLV.metric_sample_size ) + \
            'class_num: {}\n'.format(GLV.class_num) + \
            'batch_size: {}\n'.format(GLV.batch_size) + \
            'epochs: {}\n'.format(GLV.epochs) + \
            'z_dim: {}\n'.format(GLV.z_dim) + \
            'n_critic: {}\n'.format(GLV.n_critic) + \
            'net_complex: {}\n'.format(GLV.net_complex) + \
            'WGAN_Loss: {}\n'.format(GLV.WGAN_Loss) + \
            'gumbel_softmax: {}\n'.format(GLV.gumbel_softmax) + \
            'softmax_argmax: {}\n'.format(GLV.softmax_argmax) + \
            'softmax_sample: {}\n'.format(GLV.softmax_sample) + \
            'gumbel_temp: {}\n'.format(GLV.gumbel_temp) + \
            'metric_dist_size: {}\n'.format(GLV.metric_dist_size) + \
            'clipping: {}\n'.format(GLV.clipping) + \
            'WGAN_GP: {}\n'.format(GLV.WGAN_GP) + \
            'WGAN_GP_lambda: {}\n'.format(GLV.WGAN_GP_lambda)
        return config




GLV = MyGlobal()