
from PIL import Image
import numpy as np
import torch 
from make_model import make_model
from torch import optim
from torchvision import transforms
import argparse

def get_input_args():
    
    
    parser = argparse.ArgumentParser()
    

    parser.add_argument('--gpu', type=str, default=False, help='Usase: --gpu True')
    parser.add_argument('--img', type=str, default= 'flowers/test/97/image_07719.jpg', help= 'path of image file')
    parser.add_argument('--checkpoint', type=str, default= 'checkpoint/vgg_train_test.pt', help='path of checkpoint')
    parser.add_argument('--topk', type= int, default=5, help= 'select top k(int) probability')
    parser.add_argument('--categeory', type= str, default='cat_to_name.json', help= 'path of json file')

    return parser.parse_args()  



def load_checkpoint(path):
    
    
    #state = torch.load(path)
    checkpoint = torch.load(path, map_location=lambda storage, loc: storage)

    drop = checkpoint['dropout']
    hidden_units = checkpoint['hidden_units']  
    arch = checkpoint['arch']
    lr = checkpoint['lr']
    epochs = checkpoint['epochs']
    state_dict = checkpoint['state_dict']
    class_to_idx = checkpoint['class_to_idx']
   
    model = make_model(arch, hidden_units, drop)
    
    model.class_to_idx = class_to_idx
    
    print("Loading ",  model.name, " checkpoint\n")
    if model.name == 'vgg16' or model.name == 'densenet121':
        model.classifier.load_state_dict(state_dict)
        optimizer = optim.Adam(model.classifier.parameters(), lr=lr)
    elif model.name == 'resnet50':
        model.fc.load_state_dict(state_dict)
        optimizer = optim.Adam(model.fc.parameters(), lr=lr)
        
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
############################################################# 
#  following code is needed if we want train model further  #
#  after loading from checkpoint since it requires gpu      #
############################################################# 
#   for state in optimizer.state.values():
#         for k, v in state.items():
#             if isinstance(v, torch.Tensor):
#                 state[k] = v.cuda()
    print(model.name, " loaded successfully\n")
    return model, optimizer


def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    img_loader = transforms.Compose([
        transforms.Resize(256), 
        transforms.CenterCrop(224), 
        transforms.ToTensor()])
    
    pil_image = Image.open(image_path)
    pil_image = img_loader(pil_image).float()
    
    np_image = np.array(pil_image)    
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np.transpose(np_image, (1, 2, 0)) - mean)/std    
    np_image = np.transpose(np_image, (2, 0, 1))
            
    return np_image


#############################################################################
##### Alternate process_image ##############################################
###########################################################################
# def process_image(image):
#     ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
#         returns an Numpy array
#     '''
#     img_loader = transforms.Compose([
#         transforms.Resize(256), 
#         transforms.CenterCrop(224), 
#         transforms.ToTensor()])
    
#     pil_image = Image.open(image)
#     pil_image = img_loader(pil_image).float()
    
#     np_image = np.array(pil_image)    
    
#     mean = np.array([0.485, 0.456, 0.406])
#     std = np.array([0.229, 0.224, 0.225])
#     np_image = (np.transpose(np_image, (1, 2, 0)) - mean)/std    
#     np_image = np.transpose(np_image, (2, 0, 1))
            
#     return np_image



def make_prediction(image_path, model, topk, device):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # TODO: Implement the code to predict the class from an image file
    print("Predicting ", topk, " flowers name for given image\n")
    image = process_image(image_path)
    
    # convert img to tensor
    image = torch.from_numpy(image)
    # fix the tensor dtype
    image = image.type(torch.FloatTensor)
   
    # copy img to proper device
    image = image.to(device)
    # adjust dimension 
    image = image.reshape((-1, image.shape[0], image.shape[1], image.shape[2]))
 
    # turn on eval mode
    model.eval()
    with torch.no_grad():
        logps = model(image)
    
    # get actual ps from logps
    ps = torch.exp(logps)
    
    # get topk ps and classes respectively
    top_p, top_class = ps.topk(topk, dim=1)
    
    # copy to cpu and convert into numpy
    top_p = top_p.cpu()
    top_class = top_class.cpu()
    top_p = top_p.numpy()
    top_class = top_class.numpy()
    
    # get mapping of class to idx
    class_to_idx = model.class_to_idx
    # reverse mapping
    idx_to_class = { v:k for k,v in class_to_idx.items()}
    # get class from its idx
    top_class = [idx_to_class[x] for x in top_class[0,:]]
    
    
    return top_p[0], top_class
