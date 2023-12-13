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
from sklearn import preprocessing
import os

NUMLABS = 1044
edge_index = {
    "process": [[], []], # patient -> treatment
    "diagnosis": [[], []] # patient -> diagnosis
}
node2index = {
    "patient": {},
    "icd": {},
    "ndc": {}
}

patient2label = []
patient2feat = [] # age + normalize

parser = argparse.ArgumentParser()
parser.add_argument('--label', type=int, default=0)
args = parser.parse_args()

LABELLIST = ['EXPIRE_FLAG', 'cohort', 'Obesity', 'Non.Adherence',
       'Developmental.Delay.Retardation', 'Advanced.Heart.Disease',
       'Advanced.Lung.Disease',
       'Schizophrenia.and.other.Psychiatric.Disorders', 'Alcohol.Abuse',
       'Other.Substance.Abuse', 'Chronic.Pain.Fibromyalgia',
       'Chronic.Neurological.Dystrophies', 'Advanced.Cancer', 'Depression',
       'Dementia', 'Unsure']
def process_patient(infile, label, min_length_of_stay=0):
    
    labelname = LABELLIST[label]
    
    inff = open(infile, 'r') # "/root/IDL_Project/MIMIC3/patient.csv"
    count = 0
    for line in csv.DictReader(inff):
        if count % 10000 == 0:
            sys.stdout.write('%d\r' % count)
            sys.stdout.flush()
        patient_id = str(int(float(line['SUBJECT_ID'])))
        encounter_id = str(int(float(line['HADM_ID'])))
        age = int(line["AGE"])
        if line["GENDER"].lower() == "m":
            gender = 0
        else:
            gender = 1
        
        patient_node = patient_id + ":" + encounter_id
        label = line[labelname] == "1"
#         expired = line['EXPIRE_FLAG'] == "1"

        if patient_node not in node2index["patient"]:
            node2index["patient"][patient_node] = len(node2index["patient"])
            nodeindex = node2index["patient"][patient_node]
            patient2label.append(label)
            patient2feat.append([0 for _ in range(NUMLABS)] + [age, gender])
            
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
        vi = node2index["patient"][patient_node]
        
        dx_id = "dia:" + line['NDC'].lower()
        if dx_id not in node2index["ndc"]:
            node2index["ndc"][dx_id] = len(node2index["ndc"])
        vj = node2index["ndc"][dx_id]
        
        edge_index["diagnosis"][0].append(vi)
        edge_index["diagnosis"][1].append(vj)
        
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
        if patient_node not in node2index["patient"]:
            continue
        vi = node2index["patient"][patient_node]
        
        treatment_id = "proc:" + line['ICD9_CODE'].lower()
        if treatment_id not in node2index["icd"]:
            node2index["icd"][treatment_id] = len(node2index["icd"])
        vj = node2index["icd"][treatment_id]
        
        edge_index["process"][0].append(vi)
        edge_index["process"][1].append(vj)
        
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
        if patient_node not in node2index["patient"]:
            continue
        patient_id = node2index["patient"][patient_node]
        
        lab_id = line['CHART_ITEMID'].lower()
        if lab_id not in lab2index:
            lab2index[lab_id] = len(lab2index)
        lab_index = lab2index[lab_id]
        
        patient2feat[patient_id][lab_index] = float(line['CHART_VALUENUM'])

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
lab_file = input_path + '/lab_final.csv'

process_patient(admission_dx_file, args.label)
process_diagnosis(diagnosis_file)
process_treatment(treatment_file)
process_lab(lab_file)

input_path = '../rawdata/'
train_ids, val_ids, test_ids = [], [], []
with open(input_path + "train_ids.txt", "r") as f:
    for line in f.readlines():
        _id = line.rstrip()
        idx = node2index["patient"][_id]
        train_ids.append(idx)
with open(input_path + "val_ids.txt", "r") as f:
    for line in f.readlines():
        _id = line.rstrip()
        idx = node2index["patient"][_id]
        val_ids.append(idx)
with open(input_path + "test_ids.txt", "r") as f:
    for line in f.readlines():
        _id = line.rstrip()
        idx = node2index["patient"][_id]
        test_ids.append(idx)
train_mask = [] #np.zeros(len(train_ids) + len(val_ids) + len(test_ids))
val_mask = [] # np.zeros(len(train_ids) + len(val_ids) + len(test_ids))
test_mask = [] # np.zeros(len(train_ids) + len(val_ids) + len(test_ids))
for _id in train_ids:
    train_mask.append(_id)
for _id in val_ids:
    val_mask.append(_id)
for _id in test_ids:
    test_mask.append(_id)

outpath = "./" + str(args.label) + "/"
if os.path.isdir(outpath) is False:
    os.makedirs(outpath)

np.save(outpath + "train_mask", np.array(train_mask))
np.save(outpath + "val_mask", np.array(val_mask))
np.save(outpath + "test_mask", np.array(test_mask))

for keyname in edge_index.keys():
    ei = np.array(edge_index[keyname])
    np.save(outpath + "edge_index_" + keyname, ei)
    
labels = np.array(patient2label)
np.save(outpath + "patient_label", labels)

feats = np.array(patient2feat)
np.save(outpath + "patient_feat", feats)
# Normalize?
# scaler = preprocessing.StandardScaler().fit(feats)
# feat_scaled = scaler.transform(feats)
# np.save("patient_feat", feat_scaled)


ndc_feats = np.zeros(len(node2index["ndc"]))
ndc_feats = np.expand_dims(ndc_feats, axis=1)
np.save(outpath + "ndc_feat", ndc_feats)
icd_feats = np.zeros(len(node2index["icd"]))
icd_feats = np.expand_dims(icd_feats, axis=1)
np.save(outpath + "icd_feat", icd_feats)
