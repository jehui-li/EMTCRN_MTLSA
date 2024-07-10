import numpy as np
from utils import load_DEAP
from sklearn.model_selection  import train_test_split
from torch.utils.data import TensorDataset
import torch
from sklearn.utils import shuffle
import os


DATA_DIR = "/content/drive/MyDrive/data_preprocessed_python/data_preprocessed_python"



def baseline_removal(data):
    """ 
    calculate the baseline signal per second 
    then subtract that baseline from the signal
    """
    # duration of the baseline
    baseline_dur = 3 
    # signal's sampling rate
    sampling_rate = 128
    preprocessed = []
    # loop through the data array (n_instance, n_channels, n_samples)
    for ins in range(data.shape[0]):
        preprocessed_ins = []
        for c in range(data.shape[1]):
            signal = data[ins, c]
            # get all 3 second baseline segment and split in to 3 1-second segments
            all_baseline = np.split(signal[:baseline_dur*sampling_rate], 3)
            signal = signal[baseline_dur*sampling_rate:]
            # calculate the per second mean baseline
            baseline_per_second = np.mean(all_baseline, axis = 0)
            # print(baseline_per_second.shape)
            baseline_to_remove = np.tile(baseline_per_second, int(len(signal)/sampling_rate))
            signal_baseline_removed = signal - baseline_to_remove
    
            signal_split = signal_baseline_removed.reshape(-1, 3*128)
            
            preprocessed_ins.append(signal_split)
        
        preprocessed.append(preprocessed_ins)
        

    return np.array(preprocessed).transpose(0, 2, 1, 3)


def dataset_prepare(segment_duration = 3, n_subjects = 1, load_all = True, single_subject = False, return_dataset = False, sampling_rate = 128):
    data = load_DEAP(DATA_DIR, n_subjects = n_subjects, single_subject=single_subject,load_all = load_all)    

    # s1, s1_labels, s1_names = data
    s1, labels_val, labels_arousal, s1_names = data
    # all_labels shape: (1280,)
    labels_val = np.repeat(labels_val.reshape(-1, 1), 20)
    labels_arousal = np.repeat(labels_arousal.reshape(-1, 1), 20)
    print("labels_val shape", labels_val.shape)
    print("labels_arousal shape", labels_arousal.shape)
    # preprocesed labels shape:  (25600,)

    s1_preprocessed = baseline_removal(s1)
    b, s, c, n = s1_preprocessed.shape
    s1_preprocessed = s1_preprocessed.reshape(b*s, c, segment_duration, sampling_rate).transpose(0, 2, 1, 3)

    # preprocesed data shape: (25600, 3, 32, 128)
    if single_subject:
      X_train_val, X_value_val, y_train_val, y_value_val = train_test_split(s1_preprocessed, labels_val, test_size = 0.2, stratify = labels_val, shuffle = True, random_state = 29)
      X_train_arousal, X_val_arousal, y_train_arousal, y_val_arousal = train_test_split(s1_preprocessed, labels_arousal, test_size = 0.2, stratify = labels_arousal, shuffle = True, random_state = 29)

      X_train_val_new, X_test_val, y_train_val_new, y_test_val = train_test_split(X_train_val, y_train_val, test_size = 0.2, stratify = labels_val, shuffle = True, random_state = 29)
      X_train_arousal_new, X_test_arousal, y_train_arousal_new, y_test_arousal = train_test_split(X_train_arousal, y_train_arousal, test_size = 0.2, stratify = y_train_arousal, shuffle = True, random_state = 29)
    
    if return_dataset:
      train_x_val = torch.Tensor(X_train_val) # transform to torch tensor
      train_y_val = torch.Tensor(y_train_val)
      test_x_val = torch.Tensor(X_test_val) # transform to torch tensor
      test_y_val = torch.Tensor(y_test_val)
      train_dataset_val = TensorDataset(train_x_val, train_y_val.long()) # create your datset
      test_dataset_val = TensorDataset(test_x_val, test_y_val.long())

      train_x_arousal = torch.Tensor(X_train_arousal_new) # transform to torch tensor
      train_y_arousal = torch.Tensor(y_train_arousal_new)
      test_x_arousal = torch.Tensor(X_test_arousal) # transform to torch tensor
      test_y_arousal = torch.Tensor(y_test_arousal)
      train_dataset_arousal = TensorDataset(train_x_arousal, train_y_arousal.long()) # create your datset
      test_dataset_arousal = TensorDataset(test_x_arousal, test_y_arousal.long())

      return train_dataset_val, test_dataset_val, train_dataset_arousal, test_dataset_arousal
    X_train_val, X_value_val, y_train_val, y_value_val = train_test_split(s1_preprocessed, labels_val, test_size = 0.2, stratify = labels_val, shuffle = True, random_state = 29)
    X_train_arousal, X_val_arousal, y_train_arousal, y_val_arousal = train_test_split(s1_preprocessed, labels_arousal, test_size = 0.2, stratify = labels_arousal, shuffle = True, random_state = 29)

    # return X_train, y_train[:, np.newaxis], X_test, y_test[:, np.newaxis]
    return X_train_val, X_value_val, y_train_val[:, np.newaxis], y_value_val[:, np.newaxis], X_train_arousal, X_val_arousal, y_train_arousal[:, np.newaxis], y_val_arousal[:, np.newaxis]
    
if __name__ == "__main__":
    train_dataset, test_dataset =  dataset_prepare()