'''
HF_Lung_V1 dataset metadata annotation for lung sound classification (segment-level preprocessing).
'''

import os

metacsv = open('metadata.csv', 'w', encoding='utf-8')
metacsv.write('filename,split,device,location\n')

def get_device_and_location(filename):
    """extract device and location from filename"""
    device, location = "unknown", "unknown"

    if filename.startswith("steth_"):
        device = "Litt3200"
    elif filename.startswith("trunc_"):
        device = "HF_Type-1"

        # e.g.: trunc_2019-05-07-14-29-34-L2_1.txt → location = L2
        parts = filename.split('-')
        for part in parts:
            if part.startswith("L") and "_" in part:
                location = part.split('_')[0]  # L2_1 → L2
                break

    return device, location

# ---- Train ----
path = './train'
folders = os.listdir(path)

for file in folders:
    # extract device and location information
    if file.endswith('.txt'):
        device, location = get_device_and_location(file)
        metacsv.write(f"{file.split('.')[0][:-6]},train,{device},{location}\n")

# ---- Test ----
path = './test'
folders = os.listdir(path)

for file in folders:
    # extract device and location information
    if file.endswith('.txt'):
        device, location = get_device_and_location(file)
        metacsv.write(f"{file.split('.')[0][:-6]},test,{device},{location}\n")

metacsv.close()
