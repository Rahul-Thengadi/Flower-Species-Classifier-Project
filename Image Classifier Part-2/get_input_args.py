import argparse

def get_input_args():
    
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_dir', type=str, default='flowers/', help='location of datasets')
    parser.add_argument('--save_dir', type=str, default='workspace_backup', help='checkpoint directory path')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate for model')
    parser.add_argument('--drop', type=float, nargs=2, default=[0.5, 0.3], help='enter dropout')
    parser.add_argument('--hidden_units', type=int,nargs=2, default=[4096, 512], help='enter at least 2 hidden units')
    parser.add_argument('--epochs', type=int, default= 5, help='number of iterations while training')
    parser.add_argument('--gpu', type=str, default=False, help='Usase: --gpu True')
    parser.add_argument('--arch', type=str, default='vgg16', help='chosen architecture')
    parser.add_argument('--img', type=str, default= 'flowers/test/97/image_07719.jpg', help= 'path of image file')
    parser.add_argument('--checkpoint', type=str, default= 'checkpoint/vgg_train_test.pt', help='path of checkpoint')
    parser.add_argument('--topk', type= int, default=5, help= 'select top k(int) probability')
    parser.add_argument('--categeory', type= str, default='cat_to_name.json', help= 'path of json file')

    return parser.parse_args()  
