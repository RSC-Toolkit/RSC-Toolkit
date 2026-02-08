# -*- coding: cp949 -*-
import os
import pandas as pd

def time_to_sec(tstr):
    """Convert HH:MM:SS.xxx → seconds (float)"""
    h, m, s = tstr.split(":")
    return float(h) * 3600 + float(m) * 60 + float(s)

def load_annotation(txt_path):
    """
    Read HF_Lung_V1 annotation txt file and convert to DataFrame
    columns = ["Start","End","Label"]
    """
    rows = []
    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:   # e.g. I 00:00:03.215 00:00:03.818
                label, start, end = parts
            elif len(parts) == 4: # e.g. Rhonchi 00:00:04.443 00:00:05.124
                label, start, end = parts[0], parts[1], parts[2]
            else:
                continue
            rows.append([time_to_sec(start), time_to_sec(end), label])
    return pd.DataFrame(rows, columns=["Start", "End", "Label"])


def split_segments(annotation_df, mode):
    """
    Split overlapping annotations into finer segments and assign final labels:
    Normal / Crackle / Continuous / Both
    """
    # 1. Collect all boundary points
    boundaries = sorted(set(annotation_df["Start"].tolist() + annotation_df["End"].tolist()))
    segments = []

    # 2. Check active labels for each segment
    for i in range(len(boundaries)-1):
        s, e = boundaries[i], boundaries[i+1]
        if s == e:
            continue

        active_labels = []
        for _, row in annotation_df.iterrows():
            if row["Start"] < e and row["End"] > s:  # overlapping interval
                active_labels.append(row["Label"])

        if mode == 'unified':
            # 3. Label decision
            has_continuous = any(l in ["Wheeze","Rhonchi","Stridor"] for l in active_labels)
            has_crackle = "D" in active_labels
            has_phase = any(l in ["I","E"] for l in active_labels)
    
            # Priority: Both > Continuous > Crackle > Normal
            if has_continuous and has_crackle:
                final_label = "Both"
            elif has_continuous:
                final_label = "Continuous"
            elif has_crackle:
                final_label = "Crackle"
            elif has_phase:
                final_label = "Normal"
            else:
                continue
        elif mode == 'intra':
            # 3. Label decision
            if any(l in ["Wheeze","Rhonchi","Stridor"] for l in active_labels):
                final_label = "Continuous"
            elif "D" in active_labels:
                final_label = "Crackle"
            elif any(l in ["I","E"] for l in active_labels):
                final_label = "Normal"
            else:
                continue

        segments.append({
            "start": s,
            "end": e,
            "labels": active_labels,
            "final": final_label
        })

    return pd.DataFrame(segments)

# Process multiple files in a folder
def process_folder(folder_path, mode, out_csv="test_both_segmented_labels.csv"):
    all_segments = []
    for file in os.listdir(folder_path):
        if file.endswith(".txt"):
            ann_df = load_annotation(os.path.join(folder_path, file))
            seg_df = split_segments(ann_df, mode)
            seg_df["filename"] = file.replace(".txt","")
            all_segments.append(seg_df)
    all_segments = pd.concat(all_segments, ignore_index=True)
    all_segments.to_csv(out_csv, index=False)
    return all_segments

# -----------------------
# Example usage
# -----------------------

# Single file test
'''
txt_path = "./test/trunc_2019-08-21-11-39-26-L2_6_label.txt"  # txt file containing I/E/D/Wheeze/Rhonchi/Stridor annotations
ann_df = load_annotation(txt_path)
seg_df = split_segments(ann_df, mode='unified')
print(seg_df)
'''

results = process_folder(folder_path='./train/', mode='intra', out_csv='train_intra_segmented_labels.csv')
results = process_folder(folder_path='./test/', mode='intra', out_csv='test_intra_segmented_labels.csv')    
results = process_folder(folder_path='./train/', mode='unified', out_csv='train_unified_segmented_labels.csv')
results = process_folder(folder_path='./test/', mode='unified', out_csv='test_unified_segmented_labels.csv')