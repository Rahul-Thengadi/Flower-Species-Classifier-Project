import torch
from torch import nn
from torchvision import models, datasets, transforms
from workspace_utils import keep_awake
from torch import optim


def load_data(data_dir):
    
    print("Loading data from datasets\n")
    
    train_dir = data_dir + 'train/'
    valid_dir = data_dir + 'valid/'
    test_dir = data_dir + 'test/'
    
    train_transform = transforms.Compose([transforms.RandomRotation(30),
                                    transforms.RandomResizedCrop(224),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                         [0.229, 0.224, 0.225])])

    validtest_transform = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    train_data = datasets.ImageFolder(train_dir,transform = train_transform)
    valid_data = datasets.ImageFolder(valid_dir, transform = validtest_transform)
    test_data = datasets.ImageFolder(test_dir, transform = validtest_transform)
    
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True) 
                                              
    data = {'train_data': train_data, 'valid_data': valid_data, 'test_data': test_data}
    loader = {'trainloader': trainloader, 'validloader': validloader, 'testloader': testloader}
                                              
    return data, loader                                    
                                        




def train_model(model, optimizer, epochs, device, data, loader):
    print("Training ", model.name, " model\n\n")
    
    data, loader =  data, loader
    train_data = data['train_data']
    trainloader = loader['trainloader']
    validloader = loader['validloader']
    optimizer = optimizer
    criterion = nn.NLLLoss()
     
    train_loss = 0
    print_every = 5
    steps = 0
    
    for epoch in range(epochs):
        for images, labels in trainloader:
            steps += 1
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            logps = model(images)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if steps % print_every == 0:
                model.eval()
                valid_loss = 0
                accuracy = 0
                with torch.no_grad():
                    for images, labels in validloader:
                        images, labels = images.to(device), labels.to(device)
                       
                        logps = model(images)
                        ps = torch.exp(logps)
                        loss = criterion(logps, labels)
                        valid_loss += loss.item()
                        
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor))

                    print(f"Epoch {epoch+1}/{epochs}.. "
                          f"Train loss: {train_loss/print_every:.3f}.. "
                          f"Validation loss: {valid_loss/len(validloader):.3f}.. "
                          f"Test accuracy: {accuracy/len(validloader):.3f}")
                
                train_loss = 0
                model.train()
    print("")            
    model.class_to_idx = train_data.class_to_idx            
    return model, optimizer
        


def compute_accuracy(model, loader, device):
    
    print("\n\n\nAccuracy of ", model.name, " model on test dataset: \n")
    
    testloader = loader['testloader']
    criterion = nn.NLLLoss()
    epochs = 5
    steps = 0
    print_every = 5
    model.eval()
    with torch.no_grad():
        for e in range(epochs):
            test_loss = 0
            accuracy = 0
            
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)

                logps = model(images)
                loss = criterion(logps, labels)
                test_loss += loss.item()

                ps = torch.exp(logps)

                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor))


            print("Epoch: {}/{}.. ".format(e+1, epochs),
                  "Test Loss: {:.3f}.. ".format(test_loss/len(testloader)),
                  "Test Accuracy: {:.3f}".format(accuracy/len(testloader)))    
                                        
                                        


def save_checkpoint(model , arch, optimizer, lr, epochs, hidden_units, drop, path):
    print("\nSaving ", model.name , " state and hyperparameters\n")
    checkpoint = {
                   'dropout': drop,
                   'hidden_units': hidden_units,   
                   'arch': arch,
                   'lr': lr,
                   'epochs': epochs,
                   'state_dict': model.classifier.state_dict(),
                   'class_to_idx': model.class_to_idx,
                   'optimizer_state_dict': optimizer.state_dict()
                 }
    torch.save(checkpoint, path)
    print("State & Hyperparameters of ", model.name , " saved\n")