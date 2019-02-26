# test vgg16
python predict.py --checkpoint vgg16.pt --img flowers/test/81/image_00946.jpg --topk 4 

# test densenet121
#python predict.py --checkpoint densenet121.pt --img flowers/test/81/image_00946.jpg --topk 4 --gpu True

# test resnet 
#python predict.py --checkpoint resnet50.pt --img flowers/test/81/image_00946.jpg --topk 4 --gpu True
