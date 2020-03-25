import time
import copy
import torch
import numpy as np

def _process_train_function(phase, epoch, model, writer, dataloaders, scheduler, 
                            epoch_loss, epoch_acc, running_loss, running_corrects,
                            best_acc, best_model_wts, save_model_path):
    if phase == 'train':
        if scheduler != None:
            scheduler.step()

    epoch_loss[phase].append(running_loss / len(dataloaders[phase].dataset))
    epoch_acc[phase].append(float(running_corrects) / len(dataloaders[phase].dataset))

    print('{} Loss: {:.4f} Acc: {:.4f}'.format(
        phase, epoch_loss[phase][-1], epoch_acc[phase][-1]))

    #----for tensorboard----#
    writer.add_scalar('Loss/{}'.format(phase), np.array([epoch_loss[phase][-1]]), np.array([epoch]))
    writer.add_scalar('Accuracy/{}'.format(phase), np.array([epoch_acc[phase][-1]]), np.array([epoch]))
    #-----------------------#

    if phase == 'val' and epoch_acc[phase][-1] > best_acc:
        best_acc = epoch_acc[phase][-1]
        best_model_wts = copy.deepcopy(model.state_dict())
        torch.save(model.state_dict(), save_model_path + "best_model.pth")
    return best_acc, best_model_wts

def train_model(model, dataloaders, purpose, criterion, optimizer, scheduler, 
                num_epochs, epoch_loss, epoch_acc, save_model_path, device, writer):
    since = time.time()
    
    model.to(device)
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = 100000000.0
        
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('----------')
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
                
            running_loss = 0.0
            running_corrects = 0
                        
            if purpose == 'Classification':
#                 for inputs, labels in tqdm(dataloaders[phase]):
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        
                        loss = criterion(outputs, labels)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                        
#                         if math.isnan(loss.item()) == True or math.isinf(loss.item()) == True or loss.item()>50:
#                             print('something wrong.')
#                             print('output', outputs)
#                             print('labels', labels)

                        running_loss += loss.item() * inputs.size(0)
                        running_corrects += torch.sum(preds == labels.data)
                    
                best_acc, best_model_wts = _process_train_function(phase, epoch, model, writer, dataloaders, scheduler, 
                                                                   epoch_loss, epoch_acc, running_loss, running_corrects,
                                                                   best_acc, best_model_wts, save_model_path)

            elif purpose == 'Regression':
#                 for inputs, labels in tqdm(dataloaders[phase]):    
                for inputs, labels in dataloaders[phase]:                
                    inputs = inputs.to(device)
                    labels_float = labels.float() / (class_num - 1)  #normalize
                    labels_float = labels_float.to(device)
                    labels_float = labels_float.view(-1, 1)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        outputs = outputs.float()
                        loss = criterion(outputs, labels_float)
    
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                        running_loss += loss.item() * inputs.size(0)
                        outputs = outputs.to(torch.device('cpu')) * (class_num - 1)
                        outputs = outputs.round().int()
                        outputs = outputs.view(len(labels))
                        running_corrects += torch.sum(outputs == labels.data.int())
                
                best_acc, best_model_wts = _process_train_function(phase, epoch, model, viz, dataloaders, scheduler, 
                                                                   epoch_loss, epoch_acc, running_loss, running_corrects,
                                                                   best_acc, best_model_wts, save_model_path)

        print()
        
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
        
    model.load_state_dict(best_model_wts)
    return model