import os
import csv
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader, Subset
import torch

from torch.utils.tensorboard import SummaryWriter


from skimage.util.shape import view_as_windows
import warnings




import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F



def load_data(data, window_size, overlap_size):
    """
    Split data in windows of equal size with overlap
    
    INPUT:
    @data: data numpy array of shape [elements per file, features]
    @window: number of elements per window
    @overlap_size: defines the overlapping elements between consecutive windows
    
    OUTPUT
    @data: data numpy array of shape [number_of_windows, elements per window, features]
    """

    
    if window_size==overlap_size:
        raise Exception("Overlap arg must be smaller than length of windows")
    S = window_size - overlap_size
    nd0 = ((len(data)-window_size)//S)+1
    if nd0*S-S!=len(data)-window_size:
        warnings.warn("Not all elements were covered")
    return view_as_windows(data, (window_size,data.shape[1]), step=S)[:,0,:,:]





def del_nan_element(data_with_nan):
    """
    Delete all elements in the data which have any nan valued feature
    
    INPUT:
    @data_with_nan: data numpy array containing nan_values
    
    OUTPUT
    @data_with_nan: data numpy array inlcuding just elements per window which do have no nan_vaues in any feature
    """
    nan_val = np.isnan(data_with_nan) #mask for all nan_elements as 2d array [elements_per_window, features]
    nan_val = np.any(nan_val,axis = 1) #mask for all nan_rows as 1d array [elements_per_window]
    return data_with_nan[nan_val==False]





def create_folder_dictionary(list_of_train_BSD_states, list_of_test_BSD_states, data_path):
    """
    Create a dictionaty for testing and training containing folder names as keys and files as values
    
    INPUT:
    @list_of_train_BSD_states: list containing the training BSD states as string
    @list_of_test_BSD_states: list containing the testing BSD states as string
    @data_path: data directory containing folders for each BSD state
    
    OUTPUT
    @training_folders: dictionary folders and keys for training
    @testing_folders: dictionary folders and keys for testing
    """
    
    data_path = data_path
    state_dictionary = {
        "1":"NR01_20200317_PGS_31_BSD_31",
        "2":"NR02_20200423_PGS_31_BSD_21",
        "3":"NR03_20200424_PGS_31_BSD_11",
        "4":"NR04_20200424_PGS_31_BSD_P1",
        "5":"NR05_20200930_PGS_31_BSD_22",
        "6":"NR06_20201001_PGS_31_BSD_12",
        "7":"NR07_20201001_PGS_31_BSD_32",
        "8":"NR08_20200918_PGS_31_BSD_33",
        "9":"NR09_20200917_PGS_31_BSD_P2",
        "10":"NR10_20200502_PGS_21_BSD_31",
        "11":"NR11_20200429_PGS_21_BSD_21",
        "12":"NR12_20200429_PGS_21_BSD_11",
        "13":"NR13_20200428_PGS_21_BSD_P1",
        "14":"NR14_20200731_PGS_21_BSD_22",
        "15":"NR15_20200901_PGS_21_BSD_12",
        "16":"NR16_20200908_PGS_21_BSD_32",
        "17":"NR17_20200717_PGS_21_BSD_33",
        "18":"NR18_20200714_PGS_21_BSD_P2",
        "19":"NR19_20200505_PGS_11_BSD_31",
        "20":"NR20_20200507_PGS_11_BSD_21",
        "21":"NR21_20200508_PGS_11_BSD_11",
        "22":"NR22_20200508_PGS_11_BSD_P1",
        "23":"NR23_20200511_PGS_11_BSD_22",
        "24":"NR24_20200512_PGS_11_BSD_12",
        "25":"NR25_20200512_PGS_11_BSD_32",
        "26":"NR26_20200513_PGS_11_BSD_33",
        "27":"NR27_20200513_PGS_11_BSD_P2",
    }
    
    
    training_folders = {}
    testing_folders = {}
    
    for train_element in list_of_train_BSD_states:
        training_folders[state_dictionary[train_element]]=os.listdir(os.path.join(data_path,state_dictionary[train_element]))
    for test_element in list_of_test_BSD_states:
        testing_folders[state_dictionary[test_element]]=os.listdir(os.path.join(data_path,state_dictionary[test_element]))
    
    return training_folders, testing_folders





def get_features(path):
    """
    Creates a list of all feature names
    INPUT:
    @path: path to any BSD file since the features are the same for all files
    
    OUTPUT
    @features: list of features:
    ['C:s_ist/X', 'C:s_soll/X', 'C:s_diff/X', 'C:v_(n_ist)/X', 'C:v_(n_soll)/X', 'C:P_mech./X', 'C:Pos._Diff./X',
    'C:I_ist/X', 'C:I_soll/X', 'C:x_bottom', 'C:y_bottom', 'C:z_bottom', 'C:x_nut', 'C:y_nut', 'C:z_nut',
    'C:x_top', 'C:y_top', 'C:z_top', 'D:s_ist/X', 'D:s_soll/X', 'D:s_diff/X', 'D:v_(n_ist)/X', 'D:v_(n_soll)/X',
    'D:P_mech./X', 'D:Pos._Diff./X', 'D:I_ist/X', 'D:I_soll/X', 'D:x_bottom', 'D:y_bottom', 'D:z_bottom',
    'D:x_nut', 'D:y_nut', 'D:z_nut', 'D:x_top', 'D:y_top', 'D:z_top', 'S:x_bottom', 'S:y_bottom', 'S:z_bottom',
    'S:x_nut', 'S:y_nut', 'S:z_nut', 'S:x_top', 'S:y_top', 'S:z_top', 'S:Nominal_rotational_speed[rad/s]',
    'S:Actual_rotational_speed[µm/s]', 'S:Actual_position_of_the_position_encoder(dy/dt)[µm/s]',
    'S:Actual_position_of_the_motor_encoder(dy/dt)[µm/s]']
    """
    
    with open(path, 'r') as file:
        csvreader = csv.reader(file)
        features = next(csvreader)
    return features








def concatenate_data_from_BSD_state(folders, data_path, features_of_interest, window_size, overlap_size):
    """
    Concatenates all the windowed data from each file to one big torch array
    INPUT:
    @folders: dictionary containing folders (as keys) and files (as values) to downloaded
    @data_path: data directory containing folders for each BSD state
    @features_of_interest: list of features which should be included for training
    @window_size: number of elements per widow
    
    OUTPUT:
    @n_samples: number of total elements from all included files
    @x_data: torch array containing all the data elements 
    @y_data: torch array containing the labels for all elements
    """
    
    
    # arrays to collect data and label
    x_data_concatenated = None
    y_data_concatenated = None
    
    
    iterator = 0
    first = True
    
    
    for BSD_path in folders.keys(): #folder path
        for file_path in folders[BSD_path]: #file path 
            path_BSD_file = os.path.join(data_path, BSD_path, file_path) # concatenate the data_path, folder and file path
            #in first iteration get a list if all features
            if first == True:
                features = get_features(path_BSD_file)
            
            data_BSD_file = np.genfromtxt(path_BSD_file, dtype = np.dtype('d'), delimiter=',')[1:,:] #write csv in numpy
            feature_index_list = np.where(np.isin(features, features_of_interest)) #get index for all features of interest
            data_BSD_file = data_BSD_file[:,feature_index_list] #slice numpy array such that just features of interest are included
            data_BSD_file = np.squeeze(data_BSD_file, axis = 1) # one unnecessary extra dimension was created while slicing
            data_BSD_file = del_nan_element(data_BSD_file) #delete all elements with any nan feature
            data_BSD_file = load_data(data_BSD_file, window_size, overlap_size) #window the data
            data_BSD_file = np.swapaxes(data_BSD_file,1,2) #swap axes for CNN
            
            
            #rewrite labels as BSD_condition_1 = 0, BSD_condition_2 = 1, BSD_condition_3 = 2, BSD_condition_P1 = 3
            label = BSD_path[-2] #take the first number of the BSD state for class label
            if label == "P":
                label = int(3)
            else:
                label =int(int(label)-1)
            
            
            
            #concatenate the data from each file in one numpy array
            if  first == True: #overwrite variable
                x_data_concatenated = np.copy(data_BSD_file)
                y_data_concatenated = np.copy(np.asarray([label]*np.shape(data_BSD_file)[0]))
                first = False
            else: #concatenate data numpy arrays
                x_data_concatenated = np.concatenate((x_data_concatenated, data_BSD_file), axis=0)
                y_data_concatenated = np.concatenate((y_data_concatenated,np.asarray([label]*np.shape(data_BSD_file)[0])), axis=0)
            
            iterator +=1
            print(f"{iterator}/{len(folders.keys())*len(folders[list(folders.keys())[0]])} folders downloaded")
            print(f"downloaded folder: {BSD_path}/{file_path}")
            print(f"Shape of collected datafram: X_shape: {np.shape(x_data_concatenated)}, Y_shape: {np.shape(y_data_concatenated)}")
    
    #generate torch array
    n_samples = np.shape(x_data_concatenated)[0]
    x_data = torch.from_numpy(x_data_concatenated)
    y_data = torch.from_numpy(y_data_concatenated)
    
    return n_samples, x_data, y_data




class TimeSeriesData(Dataset):
    """
    Class for creating dataset using PyTorch data primitive Dataset. An instance of this class can be used in the 
    PyTorch data primitive Dataloader
    
    The following patameters can be adjusted:
    @windwo_size: Size of window which is used as Input in CNN
    @feature_of_interest: List of all features which should be used in the CNN
    @list_of_train_BSD_states: List of BSD states which should be used for training. Be careful at least 4 BSD
    states representing the 4 different classes should be included for the training
    @list_of_test_BSD_states: List of BSD states which should be used for testing
    """
    
    
    def __init__(self, domain):
        window_size = 1024
        overlap_size = 300
        features_of_interest =     ['S:x_bottom', 'S:y_bottom', 'S:z_bottom',
    'S:x_nut', 'S:y_nut', 'S:z_nut', 'S:x_top', 'S:y_top', 'S:z_top', 'S:Nominal_rotational_speed[rad/s]',
    'S:Actual_rotational_speed[µm/s]', 'S:Actual_position_of_the_position_encoder(dy/dt)[µm/s]',
    'S:Actual_position_of_the_motor_encoder(dy/dt)[µm/s]']
        
        
        
    #['C:s_ist/X', 'C:s_soll/X', 'C:s_diff/X', 'C:v_(n_ist)/X', 'C:v_(n_soll)/X', 'C:P_mech./X', 'C:Pos._Diff./X',
    #'C:I_ist/X', 'C:I_soll/X', 'C:x_bottom', 'C:y_bottom', 'C:z_bottom', 'C:x_nut', 'C:y_nut', 'C:z_nut',
    #'C:x_top', 'C:y_top', 'C:z_top', 'D:s_ist/X', 'D:s_soll/X', 'D:s_diff/X', 'D:v_(n_ist)/X', 'D:v_(n_soll)/X',
    #'D:P_mech./X', 'D:Pos._Diff./X', 'D:I_ist/X', 'D:I_soll/X', 'D:x_bottom', 'D:y_bottom', 'D:z_bottom',
    #'D:x_nut', 'D:y_nut', 'D:z_nut', 'D:x_top', 'D:y_top', 'D:z_top','S:x_bottom', 'S:y_bottom', 'S:z_bottom',
    #'S:x_nut', 'S:y_nut', 'S:z_nut', 'S:x_top', 'S:y_top', 'S:z_top', 'S:Nominal_rotational_speed[rad/s]',
    #'S:Actual_rotational_speed[µm/s]', 'S:Actual_position_of_the_position_encoder(dy/dt)[µm/s]',
    #'S:Actual_position_of_the_motor_encoder(dy/dt)[µm/s]']
        number_of_files_per_BDS_state = 10
        
        #list_of_train_BSD_states = ["BSD_31", "BSD_21", "BSD_11", "BSD_P1"]
        #list_of_test_BSD_states = ["BSD_32", "BSD_22", "BSD_12", "BSD_P2"]
        
        list_of_train_BSD_states = ["1", "2", "3", "4", "10", "11", "12", "13", "19", "20", "21", "22"]
        list_of_test_BSD_states = ["5", "6", "7", "9", "14", "15", "16", "18", "23", "24", "25", "27"]
        
        data_path = os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)), "data")
        
        training_folders, testing_folders = create_folder_dictionary(list_of_train_BSD_states, list_of_test_BSD_states, data_path)
        folders_domain = {}
        folders_domain["test"] = testing_folders
        folders_domain["train"] = training_folders
        
        
        self.n_samples, self.x_data, self.y_data = concatenate_data_from_BSD_state(folders_domain[domain], data_path, features_of_interest, window_size, overlap_size)
        
                  
    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples






class CNN(nn.Module):
    def __init__(self, input_size, output_size,hidden_size,num_layers):
        super(CNN, self).__init__()
        
        """
        formula [(W−K+2P)/S]+1.
        """
        self.conv1 = nn.Conv1d(input_size, 64, kernel_size=2, stride=1)#input: 1024
        self.conv2 = nn.Conv1d(64,32,kernel_size=1, stride = 1, padding=1)#input: [(1025-2+2*0)/1]+1 = 1023
        self.batch1 =nn.BatchNorm1d(32)#input: [(1023-1+2*1)/1]+1 = 1025
        self.conv3 = nn.Conv1d(32,32,kernel_size=1, stride = 1, padding=1) #input:1025
        self.batch2 =nn.BatchNorm1d(32)#input: [(1025-2+0)/1]+1 = 1027
        #self.LSTM = nn.LSTM(input_size=1027, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        #self.fc1 = nn.Linear(32*hidden_size, output_size)
        self.fc1 = nn.Linear(32*1027, output_size)

    def forward(self, x):
        x = F.selu(self.conv1(x)) #conv1
        x = self.conv2(x) #conv2
        x = F.selu(self.batch1(x)) #batch1
        x = self.conv3(x) #conv3
        x = F.selu(self.batch2(x)) #batch2
        #x, h = self.LSTM(x) 
        x = torch.reshape(x,(x.shape[0],x.shape[1]*x.shape[2])) #flatten
        x = self.fc1(x) #linear1
        output = x
        
        return output










############################    DATASET     ############################

#DATASET TEST (domain = "test")
dataset_test = TimeSeriesData("test")
#DATASET TRAIN (domain = "train")
dataset_train = TimeSeriesData("train")










############################    MODEL   ############################
input_size = 13
output_size = 4
hidden_size = 1000
num_layers = 2

model = CNN(input_size, output_size,hidden_size, num_layers)










############################    DATALOADER   ############################

# define train/val dimensions
train_size = int(0.8 * len(dataset_train))
validation_size = len(dataset_train) - train_size

#split dataset randomly
training_dataset, validation_dataset = torch.utils.data.random_split(dataset_train, [train_size, validation_size])

#define batch size for dataloader
batch_size = 4

#dataloader
train_loader = DataLoader(dataset=training_dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=2)


val_loader = DataLoader(dataset=validation_dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=2)



test_loader = DataLoader(dataset=dataset_test,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=2)


data_loader = {}
data_loader['train'] = train_loader
data_loader['val'] = val_loader










############################    TENSORFLOW INIT   ############################

writer_graph = SummaryWriter('runs/Dataloader2/graph')
writer_train = SummaryWriter('runs/Dataloader2/train')
writer_val = SummaryWriter('runs/Dataloader2/val')
writer = {}
writer["train"] = writer_train
writer["val"] = writer_val











############################    TRAINING   ############################

#define training params
num_epochs = 400
learning_rate = 0.008

#define loss and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

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
n_total_steps = len(train_loader)
len_data_loader={}
len_data_loader['train'] = len(train_loader)
len_data_loader['val'] = len(val_loader)


# TENSORBOARD SHOW MODEL
examples = iter(val_loader)
example_data, example_targets = examples.next()

writer_graph.add_graph(model, example_data.float())


# Train and Validate the model
for epoch in range(num_epochs):


    for phase in ["train", "val"]:
        for i, (window, labels) in enumerate(data_loader[phase]):
            
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


            #plot information during training
            #if (i+1) % 20 == 0:
            #    print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

        # TENSORBOARD TRACK TRAINING CURVES (LOSS, ACCURACY)
        running_loss = loss_collected / len_data_loader[phase]
        if phase == "train":
            running_loss = running_loss.detach()
        loss_list[phase].append(running_loss)
        
        
        running_accuracy = correct_prediction_collected / len_data_loader[phase] / output.size(0)
        accuracy_list[phase].append(running_accuracy)
        
        writer[phase].add_scalar(f'training loss', running_loss, epoch)
        writer[phase].add_scalar(f'accuracy', running_accuracy, epoch)
                
        correct_prediction_collected = 0
        loss_collected = 0.0
        
    # PRINT EPOCH INFO    
    print(f"Epoch {epoch+1}/{num_epochs} successfull")












############################    TRAINING PLOT   ############################

fig1 = plt.figure()
plt.title('Loss')
plt.plot(loss_list['train'], 'bo-', label = 'train', linewidth=1,markersize=0.1)
plt.plot(loss_list['val'], 'ro-', label = 'train', linewidth=1,markersize=0.1)
plt.legend()
plt.show()
plt.savefig('CNN_NO_DOMAIN_ADAPTION_LOSS')



fig2 = plt.figure()

plt.plot(accuracy_list['train'], 'bo-', label = 'train', linewidth=1,markersize=0.1)
plt.plot(accuracy_list['val'], 'ro-', label = 'val', linewidth=1,markersize=0.1)
plt.legend()
plt.show()
plt.savefig('CNN_NO_DOMAIN_ADAPTION_ACCURACY')












############################    TESTING ON OTHER DOMAIN   ############################


with torch.no_grad():
    classes = ['BSD_11', 'BSD_21', 'BSD_31', 'BSD_P1']
    
    #collect information about labels, predictions
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(4)]
    n_class_samples = [0 for i in range(4)]
    n_class_samples_out = [0 for i in range(4)]
    
    #iterate through bateches in test_loader
    for window, labels in test_loader:
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