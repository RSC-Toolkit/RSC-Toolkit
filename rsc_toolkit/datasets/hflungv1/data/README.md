# HF_Lung_V1
This dataset can be downloaded from https://gitlab.com/techsupportHF/HF_Lung_V1

This repository contains an open access lung sound database used for developing automated inhalation, exhalation, and adventitious sounds detection algorithms. 

The lung sound recordings come from two sources. The first source was a database used in a datathon in Taiwan Smart Emergency and Critical Care (TSECC), 2020, under the license of Creative Commons Attribution 4.0 (CC BY 4.0), provided by the Taiwan Society of Emergency and Critical Care Medicine (TSECCM). Lung sound recordings in the TSECC database were acquired from 261 patients. 

The second source was sound recordings acquired from 18 residents of a respiratory care ward (RCW) or a respiratory care center (RCC) in Northern Taiwan between August 2018 and October 2019. The recordings were approved by the Research Ethics Review Committee of Far Eastern Memorial Hospital (case number: 107052-F). Written informed consent was obtained from the 18 patients.

We have used the database to train and benchmark eight recurrent neural network models for automated respiratory sound analysis. The detailed information about how we used the database to train AI models and the benchmarking results are published in PLOS ONE, which can be downloaded at https://doi.org/10.1371/journal.pone.0254134.

Unzip the files to get two folders, train and test, which are the training dataset and test dataset used in a study.
Inside the folders you can find 15-second-long lung sound recordings and the corresponding labels, including 9765 15-second audio files, 34,095 inhalation labels, 18,349 exhalation labels, 13,883 continuous adventitious sound labels (including 8,457 wheeze labels, 686 stridor labels and 4,740 rhonchi labels) and 15,606 discontinuous adventitious sound (all are crackles) labels in total.

The filename of the truncated 15-second long audio files recorded by the digital stethoscope Littmann 3200 is prefixed by “steth_”. The filename of 15-second-long audio files recorded by a multichannel acoustic recording device, HF_Type-1, is prefixed by “trunc_”. The information of the date and time of the recording follows the prefix, e.g., 
“steth_yyyymmdd_HH_MM_ss” or “trunc_yyyy-mm-dd-HH-MM-ss”. Note that, the date on the filename is randomly shifted for the purpose of de-identification.

For the files from the HF-Type-1 device, the additional information of auscultation locations and truncated order trail on the filename, e.g., “trunc_yyyy-mm-dd-HH-MM-ss-LX_N”, where LX stands for the auscultation location and the number N stands for the Nth 15 seconds truncation from the original sound recording, where L1 represents the 2nd intercostal space (ICS) on the right midclavicular line (MCL), L2 represents the 5th ICS on the rigt MCL, L3 repsresents the 4th ICS on the right midaxillary line (MAL), L4 represents the 10th ICS on the right MAL, L5 represents the 2nd ICS on the left MCL, L6 represents the 5th ICS on the left MCL, L7 represents the 4th ICS on the left MAL and L8 represents the 10th ICS on the left MCL.

Not all eight files recorded in each round with the Littmann 3200 device (prefixed with “steth_”)
are included in the established database; therefore, the auscultation location should not be inferred
when using these files.

The files with the same date are very likely collected from the same subject with only short time intervals between each other; hence we suggest that the files with the same date should be assigned to only one of the following datasets, training, validation or testing.

The labels are saved in Text (.txt) format. The filename of the labels has the same beginning as the filename of corresponding audio file but is suffixed by “_label”. 

It should be noted that the labels in a single label file is made only by one annotator. Even the annotators were kept having good agreement on labeling criteria by attending regular meetings, the labeling is still not perfect. 

The establishment of ground-truth labels annotated by multiple labelers is under process. After the ground-truth labels are created, the database will be updated and declared.

The database HF_Lung_V1 by Heroic-Faith Medical Science Co. Ltd. is licensed under a Creative Commons Attribution 4.0 (CC BY 4.0) International License.


Updated on 2022/01/18:
1. Some audio files with a prefix of “steth_” collected by 3M Littmann 3200 were longer than 15 seconds. We have correctly truncated them into 15 seconds.
2. During the deidentification process, 5 audio files and 5 label files had incorrect filenames showing a date string with “20190931”. Even though the filenames did not influence the AI training and testing, we have made a correction.
3. One audio file “steth_20190507_14_34_58.wav” did not have full 15-second signal because the recording was not completed correctly. We have padded 0 to its tail and made it a 15-second audio file. 
4. The training and test data sets are repacked and reuploaded without compression. Therefore, we have more zip files.

