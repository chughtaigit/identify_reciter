# Udacity Capstone Project: Quran Reciter Identification (Speech Recognition) using Machine Learning
## Machine Learning Nanodegree 2020
This directory contain all code that was used for the Udacity Machine Learning Engineer Nanodegree Program's Capstone project.

### Notebooks
The following Notebooks were used for this project in this order:
- 1_Data_Download: Summarizes the downloaded data & explores data characteristics & visulizations
- 2_Data_Preparation: This notebook prepares the data for training.
- 3_Training_Testing: This notebook loads the train/val/test data, trains the NN, and tests it on the test data.

### Project write-up
- The project proposal is in proposal.pdf file. 
- The final project write-up is in Capstone_Report_Identify_Reciter_Speech_Recognition.pdf file.

### Data set used
A snapshot of the processed train/val/test data can be downloaded from here:
https://1drv.ms/u/s!Arg4zfuZ8qTzihJYZJizybMsDQJ3?e=gO7QRw

It can also be created from scratch using the Notebooks and raw data set below.
The raw dataset for this project can be downloaded from here: https://everyayah.com/data/status.php

Here are the reciters whose data was used and direct links to the ZIP'd MP3 files for each Aya for each reciter:
- Ahmed Ibn Ali Al Ajamy 128kbps https://everyayah.com/data/ahmed_ibn_ali_al_ajamy_128kbps/000_versebyverse.zip
- Abdurrahmaan As-Sudais 64kbps https://everyayah.com/data/Abdurrahmaan_As-Sudais_64kbps/000_versebyverse.zip
- Alafasy 64kbps https://everyayah.com/data/Alafasy_64kbps/000_versebyverse.zip
- Fares Abbad 64kbps https://everyayah.com/data/Fares_Abbad_64kbps/000_versebyverse.zip
- Ghamadi 40kbps https://everyayah.com/data/Ghamadi_40kbps/000_versebyverse.zip

### Software used:
- Windows 10 OS 
- Python 3.7.6
- conda 4.8.3

WINDOWS> "C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe"

        +-----------------------------------------------------------------------------+
        | NVIDIA-SMI 445.87       Driver Version: 445.87       CUDA Version: 11.0     |
        |-------------------------------+----------------------+----------------------+
        | GPU  Name            TCC/WDDM | Bus-Id        Disp.A | Volatile Uncorr. ECC |
        | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
        |===============================+======================+======================|
        |   0  GeForce GTX 980M   WDDM  | 00000000:01:00.0 Off |                  N/A |
        | N/A   48C    P8    14W /  N/A |     47MiB /  8192MiB |      0%      Default |
        +-------------------------------+----------------------+----------------------+

        +-----------------------------------------------------------------------------+
        | Processes:                                                                  |
        |  GPU                  PID   Type   Process name                  GPU Memory |
        |                                                                  Usage      |
        |=============================================================================|
        |  No running processes found                                                 |
        +-----------------------------------------------------------------------------+

WINDOWS> nvcc -V

        nvcc: NVIDIA (R) Cuda compiler driver
        Copyright (c) 2005-2019 NVIDIA Corporation
        Built on Sun_Jul_28_19:12:52_Pacific_Daylight_Time_2019
        Cuda compilation tools, release 10.1, V10.1.243

### Python Package versions:
- librosa==0.7.2
- audioread==2.1.8
- pydub==0.24.0
- Keras==2.3.1
- Keras-Applications==1.0.8
- Keras-Preprocessing==1.1.2
- tensorflow==2.2.0
- tensorflow-estimator==2.2.0
- tensorflow-gpu==2.2.0
- tensorflow-gpu-estimator==2.2.0
- scikit-learn==0.23.0
