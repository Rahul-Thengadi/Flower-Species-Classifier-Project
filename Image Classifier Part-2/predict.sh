# test vgg16
python predict.py --checkpoint vgg16.pt --img flowers/test/46/image_00976.jpg --topk 4 --gpu True

# test densenet121
#python predict.py --checkpoint densenet121.pt --img flowers/test/46/image_00976.jpg --topk 4 --gpu True

# test resnet 
#python predict.py --checkpoint resnet50.pt --img flowers/test/46/image_00976.jpg --topk 4
