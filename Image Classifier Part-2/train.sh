python train.py --data_dir flowers/ --save_dir vgg16.pt --learning_rate 0.0001 --drop 0.5 0.5 --hidden_units 4096 512 --epochs 5 --arch vgg16 --gpu True

#densenet
#python train.py --data_dir flowers/ --save_dir densenet121.pt --learning_rate 0.001 --drop 0.4 0.1 --hidden_units 500 200 --epochs 1 --arch densenet121 --gpu True

#resnet50
#python train.py --data_dir flowers/ --save_dir resnet50.pt --learning_rate 0.001 --drop 0.1 0.1 --hidden_units 700 204 --epochs 5 --arch resnet50 --gpu True