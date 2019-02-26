from predict_helper import get_input_args
from predict_helper import load_checkpoint
from predict_helper import process_image  
from predict_helper import make_prediction
import torch   
import json


in_arg = get_input_args()

device = 'cpu'
  
img_path = in_arg.img
checkpoint_path = in_arg.checkpoint
cat_to_name_path = in_arg.categeory
topk = in_arg.topk
gpu = in_arg.gpu

# handle device to use
if gpu:
    if torch.cuda.is_available():
        device = 'cuda'
    else: 
        print("Device doesn't support CUDA")
        exit(0)
else:
    device = 'cpu'
    
print("prediction will be done on ", device, "\n")

# load the model & get the optimizer( optimizer if want to train further)
model, optimizer = load_checkpoint(checkpoint_path)

model.to(device)

ps, classes = make_prediction(img_path, model, topk, device)

# read file
with open(cat_to_name_path, 'r') as f:
    cat_to_name = json.load(f) 
    
# make dict of categeory:probabilty
cat_ps = {cat_to_name[idx]:ps[classes.index(idx)] for idx in classes}
classes =[k for k in cat_ps.keys()]

print("list of top ps: ", ps)
print("list of top classes: ", classes, "\n\n\n")


print('flowers                                     probability')
print("--------------------------------------------------------")
for k in cat_ps:
    print("{flower:40}{probability}".format(flower=k, probability=cat_ps[k]))
