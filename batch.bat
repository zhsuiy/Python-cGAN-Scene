python main.py --dataset scene --class_num 4 --epoch 5 --gan_type CWGAN_GP --z_dim 10  --batch_size 16 --lrG 0.0001 --lrD 0.0001 --critic 5 --beta1 0 --beta2 0.9 --batch_mode True
python main.py --dataset scene --class_num 4 --epoch 5 --gan_type CWGAN_GP --z_dim 10  --batch_size 16 --lrG 0.0002 --lrD 0.0002 --critic 2 --beta1 0 --beta2 0.9 --batch_mode True
python main.py --dataset scene --class_num 4 --epoch 5 --gan_type CWGAN --z_dim 10  --batch_size 16 --lrG 0.00005 --lrD 0.00005 --critic 5 --batch_mode True
python main.py --dataset scene --class_num 4 --epoch 5 --gan_type CWGAN --z_dim 10  --batch_size 16 --lrG 0.0001 --lrD 0.0001 --critic 5 --batch_mode True
python main.py --dataset scene --class_num 4 --epoch 5 --gan_type CGAN --z_dim 10  --batch_size 16 --batch_mode True
python main.py --dataset scene --class_num 4 --epoch 5 --gan_type CGAN --z_dim 10  --batch_size 32 --batch_mode True