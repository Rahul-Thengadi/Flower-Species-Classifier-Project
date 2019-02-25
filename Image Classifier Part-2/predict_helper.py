
from PIL import Image
import numpy as np
import torch 
from make_model import make_model
from torch import optim

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
    model.classifier.load_state_dict(state_dict)
    print("Loading ",  model.name, " checkpoint\n")
    if model.name == 'vgg16' or model.name == 'densenet121':
        optimizer = optim.Adam(model.classifier.parameters(), lr=lr)
    elif model.name == 'resnet50':
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
    
    # TODO: Process a PIL image for use in a PyTorch model
    #print("Processing image\n")
    img = Image.open(image_path)
    img = img.resize((256,256))
    width, height = img.size
    #print(width, height)
    img = img.crop(((width-224)/2, (height-224)/2,  width - (width-224)/2, height - (height-224)/2))
    img = np.array(img)
    np_img = np.zeros((224,224,3))
    np_img = img/255
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225] 
    np_img = (np_img[:,:,:] - mean)/std
    np_img = np_img.transpose(2, 0, 1)
    
    return np_img


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
