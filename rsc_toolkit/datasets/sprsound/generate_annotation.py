import pandas as pd
from glob import glob
import os
import torchaudio
import json
from torchaudio import transforms as T
import math

def merge_annotation(train, valid):
    final_meta_data = pd.concat([train, valid], axis=0)
    
    final_meta_data['start'] = pd.to_numeric(final_meta_data['start']) * 0.001 #1079 -> 1.079 seconds
    final_meta_data['end'] = pd.to_numeric(final_meta_data['end']) * 0.001
    
    return final_meta_data
    

def get_annotation(dataset, mode):
    annotation_list = list()

    for filename in dataset:
        with open (filename, "r") as f:
            data = json.load(f)
    
        filename_tokens = os.path.basename(filename).split('_')
        
    
        data_dict = {
            'Filename': os.path.basename(filename.split('.json')[0]),
            'Patient_number': filename_tokens[0], 
            'Age': filename_tokens[1], 
            'Gender': filename_tokens[2], 
            'Recording_location': filename_tokens[3],
            'Recording_number': filename_tokens[4].split('.')[0]
        }
    
        data_dict.update(data)
        annotation_list.append(data_dict)
    
    flattened_rows = []
    for record in annotation_list:
        base_info = {k: v for k, v in record.items() if k != 'event_annotation'}
        for event in record['event_annotation']:
            combined = {**base_info, **event}
            flattened_rows.append(combined)
    final_train_meta = pd.DataFrame(flattened_rows)
    final_train_meta['Split'] = 'train' if mode == 'train' else 'valid'
    
    return final_train_meta
    
    



train_data_all = sorted(glob('./Classification/train_classification_json/*.json'))
valid_data_all = sorted(glob('./Classification/valid_classification_json/2022/inter_test_json/*.json'))

print(len(train_data_all), len(valid_data_all))

train_meta = get_annotation(train_data_all, mode='train')
valid_meta = get_annotation(valid_data_all, mode='valid')

final_meta = merge_annotation(train_meta, valid_meta)


final_meta = final_meta.rename(columns={
    "record_annotation": "Record_annotation",
    "start": "Start",
    "end": "End",
    "type": "Label"
})

# Crackle+Wheeze → Both
final_meta["Label"] = final_meta["Label"].replace("Wheeze+Crackle", "Both")

final_meta.to_csv("metadata.csv", index=False)