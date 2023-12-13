import pandas as pd
import csv
import tensorflow as tf
tf.compat.v1.enable_eager_execution()
import sys
import pickle
from sklearn import model_selection
import argparse
from datetime import datetime
import numpy as np
from scipy.sparse import csr_matrix

edge_attr = []
edge_index = [[], []]
node2label = {}
node2index = {}
patient2expire = {}
node2feat = {}
     
def process_patient(infile, min_length_of_stay=0):
    
    inff = open(infile, 'r') # "/root/IDL_Project/MIMIC3/patient.csv"
    count = 0
    for line in csv.DictReader(inff):
        if count % 10000 == 0:
            sys.stdout.write('%d\r' % count)
            sys.stdout.flush()
        patient_id = str(int(float(line['SUBJECT_ID'])))
        encounter_id = str(int(float(line['HADM_ID'])))
        patient_node = patient_id + ":" + encounter_id
        expired = line['EXPIRE_FLAG'] == "1"
        patient2expire[patient_node] = expired

        if patient_node not in node2index:
            node2index[patient_node] = len(node2index)
            nodeindex = node2index[patient_node]
            node2label[nodeindex] = patient2expire[patient_node]
            node2feat[nodeindex] = [0 for _ in range(235)]
            
        count += 1
    
    inff.close()


def process_diagnosis(infile):
    inff = open(infile, 'r')
    count = 0
    missing_pid = 0
    encounter_dict = {}
    for line in csv.DictReader(inff):
        if count % 10000 == 0:
            sys.stdout.write('%d\r' % count)
            sys.stdout.flush()
        patient_id = str(int(float(line['SUBJECT_ID'])))
        encounter_id = str(int(float(line['HADM_ID'])))
        patient_node = patient_id + ":" + encounter_id
        if patient_node not in patient2expire:
            continue
        if patient_node not in node2index:
            node2index[patient_node] = len(node2index)
            nodeindex = node2index[patient_node]
            node2label[nodeindex] = patient2expire[patient_node]
            node2feat[nodeindex] = [0 for _ in range(235)]
        vi = node2index[patient_node]
        
        dx_id = "dia:" + line['NDC'].lower()
        if dx_id not in node2index:
            node2index[dx_id] = len(node2index)
        vj = node2index[dx_id]
        
        edge_attr.append("diagnosis")
        edge_index[0].append(vi)
        edge_index[1].append(vj)
        
    inff.close()


def process_treatment(infile):
    inff = open(infile, 'r')
    count = 0
    missing_eid = 0
    for line in csv.DictReader(inff):
        if count % 10000 == 0:
            sys.stdout.write('%d\r' % count)
            sys.stdout.flush()
        patient_id = str(int(float(line['SUBJECT_ID'])))
        encounter_id = str(int(float(line['HADM_ID'])))
        patient_node = patient_id + ":" + encounter_id
        if patient_node not in patient2expire:
            continue
        if patient_node not in node2index:
            node2index[patient_node] = len(node2index)
            nodeindex = node2index[patient_node]
            node2label[nodeindex] = patient2expire[patient_node]
            node2feat[nodeindex] = [0 for _ in range(235)]
        vi = node2index[patient_node]
        
        treatment_id = "proc:" + line['ICD9_CODE'].lower()
        if treatment_id not in node2index:
            node2index[treatment_id] = len(node2index)
        vj = node2index[treatment_id]
        
        edge_attr.append("process")
        edge_index[0].append(vi)
        edge_index[1].append(vj)
        
    inff.close()

def process_lab(infile):
    lab2index = {}
    
    inff = open(infile, 'r')
    count = 0
    missing_eid = 0
    for line in csv.DictReader(inff):
        if count % 10000 == 0:
            sys.stdout.write('%d\r' % count)
            sys.stdout.flush()
        patient_id = str(int(float(line['SUBJECT_ID'])))
        encounter_id = str(int(float(line['HADM_ID'])))
        patient_node = patient_id + ":" + encounter_id
        if patient_node not in node2index:
            continue
        patient_id = node2index[patient_node]
        
        lab_id = line['CHART_ITEMID'].lower()
        if lab_id not in lab2index:
            lab2index[lab_id] = len(lab2index)
        lab_index = lab2index[lab_id]
        
        node2feat[patient_id][lab_index] = float(line['CHART_VALUENUM'])

    inff.close()
    
# edge_attr = []
# edge_index = [[], []]
# node2label = {}
# node2index = {}
# patient2expire = {}
# node2feat = {}

input_path = '../../GNN_for_EHR/rawdata/mimic/'
admission_dx_file = input_path + 'patient.csv' # '/ADMISSIONS.csv'
diagnosis_file = input_path + 'medication.csv'  # '/DIAGNOSES_ICD.csv'
treatment_file = input_path + 'procedure.csv' #'/PROCEDURES_ICD.csv'
lab_file = input_path + '/lab.csv'

process_patient(admission_dx_file)
process_diagnosis(diagnosis_file)
process_treatment(treatment_file)
process_lab(lab_file)

input_path = '../../GNN_for_EHR/data/mimic/'
train_ids, val_ids, test_ids = [], [], []
with open(input_path + "train_ids.txt", "r") as f:
    for line in f.readlines():
        _id = line.rstrip()
        idx = node2index[_id]
        train_ids.append(idx)
with open(input_path + "val_ids.txt", "r") as f:
    for line in f.readlines():
        _id = line.rstrip()
        idx = node2index[_id]
        val_ids.append(idx)
with open(input_path + "test_ids.txt", "r") as f:
    for line in f.readlines():
        _id = line.rstrip()
        idx = node2index[_id]
        test_ids.append(idx)
train_mask = np.zeros(len(train_ids) + len(val_ids) + len(test_ids))
val_mask = np.zeros(len(train_ids) + len(val_ids) + len(test_ids))
test_mask = np.zeros(len(train_ids) + len(val_ids) + len(test_ids))
for _id in train_ids:
    train_mask[_id] = 1
for _id in val_ids:
    val_mask[_id] = 1
for _id in test_ids:
    test_mask[_id] = 1
    
        
edge_attr = np.array(edge_attr)
np.save("edge_attr", edge_attr)
edge_index = np.array(edge_index)
np.save("edge_index", edge_index)

np.save("train_mask", train_mask)
np.save("val_mask", val_mask)
np.save("test_mask", test_mask)

labels = np.zeros(len(node2index))
print("Total number of nodes: ", len(node2index))
for vidx in node2label:
    labels[vidx] = node2label[vidx]
labels = np.array(labels)
np.save("node_label", labels)


proc_nodes = []
dia_nodes = []
for node in node2index:
    if "proc:" in node:
        proc_nodes.append(node)
    elif "dia:" in node:
        dia_nodes.append(node)
print(len(proc_nodes))
print(len(dia_nodes))

import pickle
with open("node2feat.pickle", "wb") as output_file:
    pickle.dump(node2feat, output_file)
