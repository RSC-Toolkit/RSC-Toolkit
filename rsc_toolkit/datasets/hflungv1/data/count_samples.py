import pandas as pd

df = pd.read_csv("test_unified_segmented_labels.csv")

counts = df["final"].value_counts()

print(counts)

counts_dict = counts.to_dict()
print(counts_dict)