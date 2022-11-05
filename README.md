# _Federated Few-shot learning for cough classification_
![F2LCough workflow](https://github.com/ngandh/F2LCough/blob/main/figures/F2LCough_workflow.png?raw=true "F2LCough workflow")
Link to video on Youtube: https://www.youtube.com/watch?v=ri77TvXXKMY
### Table of Content
1. [Introduction](https://github.com/ngandh/F2LCough#Introduction)
2. [Dataset](https://github.com/ngandh/F2LCough#Data)
3. [Getting started](https://github.com/ngandh/F2LCough#Getting-started)
    - [Requirements](https://github.com/ngandh/F2LCough#Requirements)
    - [Installation](https://github.com/ngandh/F2LCough#Installation)
    - [How to run](https://github.com/ngandh/F2LCough#How-to-run)

## Introduction
Detecting cough sounds is an important problem in supporting the diagnosis and treatment of respiratory diseases.  However, collecting a huge amount of labeled cough dataset is challenging mainly due to high laborious expenses, data scarcity, and privacy concerns. In this work, we develop a framework called F2LCough that can perform cough classification effectively even in situations. F2LCough not only learns effectively in scarce data situations but also ensures the privacy of patients supplying audio by combining both few-shot and federated learning technique.

## Dataset

[**COVID-19 Thermal Face \& Cough Dataset**](https://zenodo.org/record/4739682#.Y1LK13ZBxPY) includes the thermal face dataset and the cough dataset. We utilize the cough dataset for our experiments. Each audio file lasts 1 second at the sample rate of 44,100 Hz. The cough dataset consists of  53,471 seconds of “not cough” samples consisting of background noise, office, music, airport, coffee shop, harbor, nightclub, simulated turbulence sounds, and 1,557 seconds of cough sounds. Additionally, 40,856 seconds of cough sounds are augmented with random background noise at a random volume ratio. In our experiments, we only use eight types of cough: _Barking cough, Chesty and wet cough, Coughing up crap again, Dry afternoon cough, Gaggy wet cough, Spring allergy coughing, Heavy cold, and sore throat coughing, Night wet cough_. A data file is a cough record lasting 1 second and the file format is a “.wav” file. Files are named with the same syntax: label_ordinalNumber. 
- Training data: Barking cough, Chesty and wet cough, Coughing up crap again, Dry afternoon cough, Gaggy wet cough, Spring allergy coughing (6 types)
- Test data: Heavy cold, and sore throat coughing, Night wet cough (2 types)

## Getting started
### Requirements
Python 3.7
GPU
### Installation
Other libraries are listed in requirement.txt
```
!pip install -r requirements.txt
```
### How to run
- Clone this repository
- Pick a notebook called F2LCough_training to train F2LCough
- Get a trained model in folder named results 
- Use a notebook in evaluation folder to test
