model = ['CGAN','CWGAN','CWGAN_GP']

commands = []

critics = [1, 2, 5, 10]
batch_size = [16, 64, 128]
#lrGs = [0.01, 0.001, 0.0001, 0.00001, 0.000001]
#lrDs = [0.01, 0.001, 0.0001, 0.00001, 0.000001]
lrGs = [0.001, 0.0001, 0.00001, 0.000001]
lrDs = [0.001, 0.0001, 0.00001, 0.000001]
lr = [0.001, 0.0001, 0.00001, 0.000001]

clipping = [0.1, 0.01, 0.001, 0.0001]


beta1 = [0, 0.5, 0.999]
beta2 = [0, 0.5, 0.999]

metric_knn_k = [1, 7]

with open('batchCGAN.sh','w') as f:
    for c in critics:
        f.write('CUDA_VISIBLE_DEVICES=2 python3 main.py '
                '--gpu_mode True --batch_mode True '
                '--epoch 100 --critic {}\n'.format(c))

with open('batchCWGAN.sh','w') as f:
    for c in critics:
        for l in lr:
            for clip in clipping:
                for k in metric_knn_k:
                    f.write('CUDA_VISIBLE_DEVICES=2 python3 main.py '
                            '--gpu_mode True --batch_mode True --gan_type CWGAN '                    
                            '--epoch 100 --critic {} '
                            '--lrG {} --lrD {} '
                            '--clipping {} --metric_knn_k {}\n'.format(c, l, l, clip, k))

with open('batchCWGAN_GP.sh','w') as f:
    for c in critics:
        for l in lr:
            for k in metric_knn_k:
                for b1 in beta1:
                    for b2 in beta2:
                        f.write('CUDA_VISIBLE_DEVICES=2 python3 main.py '
                                '--gpu_mode True --batch_mode True --gan_type CWGAN_GP '
                                '--epoch 100 --critic {} '
                                '--lrG {} --lrD {} '
                                '--metric_knn_k {} '
                                '--beta1 {} --beta2 {}\n'.format(c, l, l, k, b1, b2))