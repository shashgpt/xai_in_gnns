from torch_geometric.data import HeteroData
import pickle
import numpy as np
import torch

def EHRData(label):
    data = HeteroData()
    inputpath = "./data/" + str(label) + "/"
    data['icd'].x = torch.LongTensor(np.load(inputpath + "icd_feat.npy")).squeeze(1)
#     print(data['icd'].x.shape) # 591
    data['ndc'].x = torch.LongTensor(np.load(inputpath + "ndc_feat.npy")).squeeze(1)
#     print(data['ndc'].x.shape) # 2042
    data['patient'].x = torch.FloatTensor(np.load(inputpath + "patient_feat.npy"))
    data['patient'].y = torch.FloatTensor(np.load(inputpath + "patient_label.npy"))
    
    diagnosis_edge_index = torch.LongTensor(np.load(inputpath + "edge_index_diagnosis.npy"))
    process_edge_index = torch.LongTensor(np.load(inputpath + "edge_index_process.npy"))
    
    data['patient', 'toprocess', 'icd'].edge_index = process_edge_index
    data['icd', 'fromprocess', 'patient'].edge_index = process_edge_index[[1,0],:]
    data['patient', 'todiagnosis', 'ndc'].edge_index = diagnosis_edge_index
    data['ndc', 'fromdiagnosis', 'patient'].edge_index = diagnosis_edge_index[[1,0],:]
    # data['patient', 'self', 'patient'].edge_index = torch.LongTensor(self_edge_index)


    train_mask = np.load(inputpath + "train_mask.npy")
    val_mask = np.load(inputpath + "val_mask.npy")
    test_mask = np.load(inputpath + "test_mask.npy")
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    
    return data
