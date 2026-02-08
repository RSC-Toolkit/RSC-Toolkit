# -*- coding: cp949 -*-
import gdown

url = "https://drive.google.com/file/d/1rgq1ev7yk-vPTnlfJW3K_eWj-cy3ywL8/view?usp=sharing"
output = "classification.zip"

print(f"Downloading via gdown: {url} → {output}")
gdown.download(url=url, output=output, fuzzy=True)
print("Download complete.")