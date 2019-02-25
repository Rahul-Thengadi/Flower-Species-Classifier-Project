from torchvision import models
from torch import nn
from collections import OrderedDict
def make_model(arch, hidden_units, drop):
    
    print("Initialising pre-trained ", arch, " model\n")

    if arch == "vgg16":
        model = models.vgg16(pretrained=True)
        model.name = 'vgg16'
        # freez the parameters from backward pass
        for param in model.parameters():
            param.requires_grad = False
        input_size = model.classifier[0].in_features 
        
    elif arch == 'densenet121':
        model = models.densenet121(pretrained=True)
        model.name = 'densenet121'
        # freez the parameters from backward pass
        for param in model.parameters():
            param.requires_grad = False
        input_size = model.classifier.in_features 
    
    elif arch == 'resnet50':
        model = models.resnet50(pretrained=True)
        model.name = 'resnet50'
        # freez the parameters from backward pass
        for param in model.parameters():
            param.requires_grad = False
        input_size = model.classifier.in_features
        
        
    out_size = 102
    
    classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(input_size, hidden_units[0])),
                                        ('relu', nn.ReLU()),
                                        ('drop1',nn.Dropout(p=drop[0])),
                                        ('fc2', nn.Linear(hidden_units[0], hidden_units[1])),
                                        ('relu', nn.ReLU()),
                                        ('drop2',nn.Dropout(p=drop[1])),
                                        ('fc3', nn.Linear(hidden_units[1], out_size)),
                                        ('output', nn.LogSoftmax(dim=1))]))
    
 
    print("Assigning classifier to ", model.name, "\n")        
    if arch == 'resnet50':
        model.fc = classifier   
    elif arch == 'vgg16' or 'densenet121':
        model.classifier = classifier
    
    return model
    