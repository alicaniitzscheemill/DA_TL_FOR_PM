import CNN

import os
import sys



from pathlib import Path

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(1, os.path.join(sys.path[0], '..'))
import Dataloader2


def plot_learning_curves(loss_list, accuracy_list):
    """
    Plot the training and validation accuracy and loss curves

    INPUT:
    @loss_list: dictionary containing the validation, training loss for each epoch. Lists can be accesses with keys "train", "val"
    @accuracy_list: dictionary containing the validation, training accuracy for each epoch. Lists can be accesses with keys "train", "val"
    """

    
    fig1 = plt.figure()
    plt.title('Loss')
    plt.plot(loss_list['train'], 'bo-', label = 'train', linewidth=1,markersize=0.1)
    plt.plot(loss_list['val'], 'ro-', label = 'val', linewidth=1,markersize=0.1)
    plt.legend()
    plt.show()
    plt.savefig('CNN_NO_DOMAIN_ADAPTION_LOSS')



    fig2 = plt.figure()

    plt.plot(accuracy_list['train'], 'bo-', label = 'train', linewidth=1,markersize=0.1)
    plt.plot(accuracy_list['val'], 'ro-', label = 'val', linewidth=1,markersize=0.1)
    plt.legend()
    plt.show()
    plt.savefig('CNN_NO_DOMAIN_ADAPTION_ACCURACY')






def train(model, train_loader, num_epochs, lr):
    
    #define loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    #collect loss for each batch
    loss_collected = 0
    loss_list = {}
    loss_list['train']=[]
    loss_list['val']=[]

    #collect accuracy for each batch
    correct_prediction_collected = 0
    accuracy_list={}
    accuracy_list['train']=[]
    accuracy_list['val']=[]


    #information for plotting
    train_train_loader = train_loader["train"]
    val_train_loader = train_loader["val"]
    len_train_loader={}
    len_train_loader['train'] = len(train_train_loader)
    len_train_loader['val'] = len(val_train_loader)


    #Tensorboard writer
    writer_graph = SummaryWriter('runs/Dataloader2/graph')
    writer_train = SummaryWriter('runs/Dataloader2/train')
    writer_val = SummaryWriter('runs/Dataloader2/val')
    writer = {}
    writer["train"] = writer_train
    writer["val"] = writer_val


    # TENSORBOARD SHOW MODEL
    examples = iter(val_train_loader)
    example_data, example_targets = examples.next()
    writer_graph.add_graph(model, example_data.float())


    # Train and Validate the model
    for epoch in range(num_epochs):


        for phase in ["train", "val"]:
            for i, (window, labels) in enumerate(train_loader[phase]):
                
                if phase == "val":
                    
                    model.train(False) #no training
                    
                    with torch.no_grad():
                        ########Forward pass########
                        outputs = model(window.float())
                    
                        #collect loss
                        loss = criterion(outputs, labels)
                        loss_collected += loss
                        
                else:
                    
                    model.train(True) #training 
                    
                    ########Forward pass########
                    outputs = model(window.float())

                    #collect loss
                    loss = criterion(outputs, labels)
                    loss_collected += loss

                    ########Backward pass########
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                #collect accuracy
                output = outputs.argmax(dim=1)
                correct_prediction_collected += (output == labels).sum().item()
                (output == labels).sum().item()



                #plot information during training
                #if (i+1) % 20 == 0:
                #    print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

            # TENSORBOARD TRACK TRAINING CURVES (LOSS, ACCURACY)
            running_loss = loss_collected / (len_train_loader[phase] * output.size(0))
            if phase == "train":
                running_loss = running_loss.detach()
            loss_list[phase].append(running_loss)
            
            
            running_accuracy = correct_prediction_collected / (len_train_loader[phase] * output.size(0))
            accuracy_list[phase].append(running_accuracy)
            
            writer[phase].add_scalar(f'training loss', running_loss, epoch)
            writer[phase].add_scalar(f'accuracy', running_accuracy, epoch)
                    
            correct_prediction_collected = 0
            loss_collected = 0.0
            
        # PRINT EPOCH INFO    
        print(f"Epoch {epoch+1}/{num_epochs} successfull")


    plot_learning_curves(loss_list, accuracy_list)




def test(model, test_loader):
    with torch.no_grad():
        classes = ['BSD_11', 'BSD_21', 'BSD_31', 'BSD_P1']
        
        #collect information about labels, predictions
        n_correct = 0
        n_samples = 0
        n_class_correct = [0 for i in range(4)]
        n_class_samples = [0 for i in range(4)]
        n_class_samples_out = [0 for i in range(4)]
        
        #iterate through bateches in test_loader
        for i, (window, labels) in enumerate(test_loader["test"]):
            print(window)
            #make predictions for each batch
            outputs = model(window.float())
            #for each element in batch check if prediction is correct and collect total and correct predictions and labels
            for i in range(batch_size):
                if len(labels)==4:
                    label = labels[i]
                    output = torch.argmax(outputs[i])
                    if label == output:
                        n_correct+=1
                        n_class_correct[label]+=1
                    
                    n_samples+=1
                    n_class_samples[label]+=1
                    n_class_samples_out[output]+=1
                else:
                    break
        
        #calculate total accuracy
        acc = 100.0 * n_correct / n_samples
        print(f'Accuracy of the network: {acc} %')
        
        #calculate class accuracy
        for i in range(4):
            acc = 100.0 * n_class_correct[i] / n_class_samples[i]
            print(f'Accuracy of {classes[i]}: {acc} %') 



####### MAIN ########


window_size = 1024
overlap_size = 0
#features_of_interest = ['S:x_bottom', 'S:y_bottom', 'S:z_bottom', 'S:x_nut', 'S:y_nut', 'S:z_nut', 'S:x_top', 'S:y_top', 'S:z_top', 'S:Nominal_rotational_speed[rad/s]', 'S:Actual_rotational_speed[µm/s]', 'S:Actual_position_of_the_position_encoder(dy/dt)[µm/s]', 'S:Actual_position_of_the_motor_encoder(dy/dt)[µm/s]']
list_of_train_BSD_states = ["1", "2", "3", "4", "10", "11", "12", "13", "19", "20", "21", "22"]
list_of_test_BSD_states = ["5", "6", "7", "9", "14", "15", "16", "18", "23", "24", "25", "27"]
data_path = Path(os.getcwd()).parents[1]
data_path = os.path.join(data_path, "data")
dataloader_split_train = 0.8
dataloader_split_test = 0.8
batch_size = 4

#test_loader = Dataloader.create_dataloader(data_path, list_of_test_BSD_states, window_size, overlap_size, features_of_interest, "test", dataloader_split_test, batch_size)

#train_loader = Dataloader.create_dataloader(data_path, list_of_train_BSD_states, window_size, overlap_size, features_of_interest, "train", dataloader_split_train, batch_size)

##############
features_of_interest = ['C:s_ist/X', 'C:s_soll/X', 'C:s_diff/X', 'C:v_(n_ist)/X', 'C:v_(n_soll)/X', 'C:P_mech./X', 'C:Pos._Diff./X',
    'C:I_ist/X', 'C:I_soll/X', 'C:x_bottom', 'C:y_bottom', 'C:z_bottom', 'C:x_nut', 'C:y_nut', 'C:z_nut',
    'C:x_top', 'C:y_top', 'C:z_top', 'D:s_ist/X', 'D:s_soll/X', 'D:s_diff/X', 'D:v_(n_ist)/X', 'D:v_(n_soll)/X',
    'D:P_mech./X', 'D:Pos._Diff./X', 'D:I_ist/X', 'D:I_soll/X', 'D:x_bottom', 'D:y_bottom', 'D:z_bottom',
    'D:x_nut', 'D:y_nut', 'D:z_nut', 'D:x_top', 'D:y_top', 'D:z_top']
vel_cut_off_value = 0
test_loader = Dataloader2.create_dataloader(data_path, list_of_test_BSD_states, window_size, overlap_size, vel_cut_off_value, features_of_interest, "test", dataloader_split_test, batch_size)

train_loader = Dataloader2.create_dataloader(data_path, list_of_train_BSD_states, window_size, overlap_size, vel_cut_off_value, features_of_interest, "train", dataloader_split_train, batch_size)
input_size = 36
##############


#input_size = 13
output_size = 4
num_epochs = 50
lr = 0.008

model = CNN.CNN(input_size, output_size)


train(model, train_loader, num_epochs, lr)
test(model, test_loader)


"""

window_size = 1024
overlap_size = 0
features_of_interest = ['S:x_bottom', 'S:y_bottom', 'S:z_bottom', 'S:x_nut', 'S:y_nut', 'S:z_nut', 'S:x_top', 'S:y_top', 'S:z_top', 'S:Nominal_rotational_speed[rad/s]', 'S:Actual_rotational_speed[µm/s]', 'S:Actual_position_of_the_position_encoder(dy/dt)[µm/s]', 'S:Actual_position_of_the_motor_encoder(dy/dt)[µm/s]']
list_of_train_BSD_states = ["1", "2", "3", "4", "10", "11", "12", "13", "19", "20", "21", "22"]
list_of_test_BSD_states = ["5", "6", "7", "9", "14", "15", "16", "18", "23", "24", "25", "27"]
data_path = Path(os.getcwd()).parents[1]
data_path = os.path.join(data_path, "data")
dataloader_split_train = 0.8
dataloader_split_test = 0.8
batch_size = 4

test_loader = Dataloader.create_dataloader(data_path, list_of_test_BSD_states, window_size, overlap_size, features_of_interest, "test", dataloader_split_test, batch_size)

train_loader = Dataloader.create_dataloader(data_path, list_of_train_BSD_states, window_size, overlap_size, features_of_interest, "train", dataloader_split_train, batch_size)


input_size = 13
output_size = 4
num_epochs = 50
lr = 0.008

model = CNN.CNN(input_size, output_size)


train(model, train_loader, num_epochs, lr)
test(model, test_loader)


"""