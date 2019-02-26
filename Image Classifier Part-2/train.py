from train_helper import get_input_args
from train_helper import load_data
from make_model import make_model
from train_helper import train_model
from train_helper import compute_accuracy
from train_helper import save_checkpoint
from workspace_utils import active_session
import torch
from torch import optim


# command line arg
in_arg = get_input_args()

data_dir = in_arg.data_dir
save_dir = in_arg.save_dir
arch = in_arg.arch
lr = in_arg.learning_rate
hidden_units = in_arg.hidden_units
drop = in_arg.drop
epochs = in_arg.epochs
gpu = in_arg.gpu

device = 'cpu'
if gpu:
    if torch.cuda.is_available():
        device = 'cuda'
        print("GPU mode enabled\n")
    else: 
        print("Device doesn't support CUDA\n")
        exit(0)
else:
    device = 'cpu'
    print("Further training will be done on cpu, switch to GPU\n")

print("Selected Device: ", device, "\n")

# load the datasets
data, loader =  load_data(data_dir)

# make model
model = make_model(arch, hidden_units, drop)
model.to(device)

# set optimizer state according to arch 
if model.name == 'vgg16' or model.name == 'densenet121':
    optimizer = optim.Adam(model.classifier.parameters(), lr=lr)
elif model.name == 'resnet50':
    optimizer = optim.Adam(model.fc.parameters(), lr=lr)
    
# train model, get new state of optimizer in order to save it in checkpoint
trained_model, optimizer = train_model(model, optimizer, epochs, device, data, loader)

# check accuracy
compute_accuracy(trained_model, loader, device)

# save model
save_checkpoint(model , arch, optimizer, lr, epochs, hidden_units, drop, save_dir)


