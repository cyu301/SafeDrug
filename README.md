# Data and Code for Reproducing the SafeDrug paper

### Citation To the Original paper
```bibtex
@inproceedings{yang2021safedrug,
    title = {SafeDrug: Dual Molecular Graph Encoders for Safe Drug Recommendations},
    author = {Yang, Chaoqi and Xiao, Cao and Ma, Fenglong and Glass, Lucas and Sun, Jimeng},
    booktitle = {Proceedings of the Thirtieth International Joint Conference on
               Artificial Intelligence, {IJCAI} 2021},
    year = {2021}
}
```

### Link to the original repo

The repo of the original code by the authors are available from this [link](https://github.com/ycq091044/SafeDrug). This README.md is highly relying on the original repo's README.md file.

### Step 1: Package Dependency

- first, install the rdkit conda environment
```python
conda create -c conda-forge -n SafeDrug  rdkit
conda activate SafeDrug
```

- then, in SafeDrug environment, install the following package
```python
pip install scikit-learn, dill, dnc
```
Note that torch setup may vary according to GPU hardware. Generally, run the following
```python
pip install torch
```
If you are using RTX 3090, then plase use the following, which is the right way to make torch work.
```python
python3 -m pip install --user torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
```

- Finally, install other packages if necessary
```python
pip install [xxx] # any required package if necessary, maybe do not specify the version, the packages should be compatible with rdkit
```

Here is a list of reference versions for all package

```shell
pandas: 1.3.0
dill: 0.3.4
torch: 1.8.0+cu111
rdkit: 2021.03.4
scikit-learn: 0.24.2
numpy: 1.21.1
```

### Step 2: Data Processing

- Go to https://physionet.org/content/mimiciii/1.4/ to download the MIMIC-III dataset (You may need to get the certificate)

  ```python
  cd ./data
  wget -r -N -c -np --user [account] --ask-password https://physionet.org/files/mimiciii/1.4/
  ```

- go into the folder and unzip three main files

  ```python
  cd ./physionet.org/files/mimiciii/1.4
  gzip -d PROCEDURES_ICD.csv.gz # procedure information
  gzip -d PRESCRIPTIONS.csv.gz  # prescription information
  gzip -d DIAGNOSES_ICD.csv.gz  # diagnosis information
  ```

- download the DDI file and move it to the data folder
  download https://drive.google.com/file/d/1mnPc0O0ztz0fkv3HF-dpmBb8PLWsEoDz/view?usp=sharing
  ```python
  mv drug-DDI.csv ./data
  ```

- processing the data to get a complete records_final.pkl

  ```python
  cd ./data  
  python processing.py
  ```


### Step 3: run the code

```python
python SafeDrug.py
```

here is the argument:

    usage: SafeDrug.py [-h] [--lr LR] [--target_ddi TARGET_DDI] 
                       [--kp KP] [--dim DIM] [--round ROUND]
                       [--method METHOD]
    
    optional arguments:
      -h, --help            show this help message and exit
      --lr LR               learning rate
      --target_ddi TARGET_DDI
                            target ddi
      --kp KP               coefficient of P signal
      --dim DIM             dimension
      --round ROUND         training round
      --method METHOD       segmentation method

### Pretrained model

There is a pretrained model available under the path ```./src/saved/SafeDrug/```. You can load it without long training waiting time.

### Main Reproducibility results
  | Model | DDI | Jaccard | F1-score | PRAUC | Avg. \# of Drugs |
  |-------|-----|---------|----------|-------|------------------|   
  | LR | 0.0790 $\pm$ 0.0009 | 0.4945 $\pm$ 0.0027 | 0.6513 $\pm$ 0.0025 | 0.7591 $\pm$ 0.0027 | 16.5061 $\pm$ 0.1375 |
  | ECC | 0.0784 $\pm$ 0.0010 | 0.4830 $\pm$ 0.0028 | 0.6392 $\pm$ 0.0027 | 0.7578 $\pm$ 0.0025 | 15.8447 $\pm$ 0.1466 |
  | RETAIN | 0.0888 $\pm$ 0.0018 | 0.4778 $\pm$ 0.0039 | 0.6376 $\pm$ 0.0038 | 0.7459 $\pm$ 0.0043 | 18.6059 $\pm$ 0.2360 |
  | LEAP | 0.0731 $\pm$ 0.0009 | 0.4440 $\pm$ 0.0031 | 0.6055 $\pm$ 0.0031 | 0.6499 $\pm$ 0.0040 | 18.5961 $\pm$ 0.0758 |
  | GAMENet | 0.0881 $\pm$ 0.0009 | 0.5081 $\pm$ 0.0030 | 0.6636 $\pm$ 0.0028 | 0.7622 $\pm$ 0.0030 | 24.6271 $\pm$ 0.1616 |
  | SafeDrug | 0.0622 $\pm$ 0.0004 | 0.5053 $\pm$ 0.0035 | 0.6614 $\pm$ 0.0033 | 0.7570 $\pm$ 0.0035 | 18.7787 $\pm$ 0.1282 |

### Folder Specification
- ```data/```
    - **procesing.py** our data preprocessing file.
    - ```Input/``` (extracted from external resources)
        - **PRESCRIPTIONS.csv**: the prescription file from MIMIC-III raw dataset
        - **DIAGNOSES_ICD.csv**: the diagnosis file from MIMIC-III raw dataset
        - **PROCEDURES_ICD.csv**: the procedure file from MIMIC-III raw dataset
        - **RXCUI2atc4.csv**: this is a NDC-RXCUI-ATC4 mapping file, and we only need the RXCUI to ATC4 mapping. This file is obtained from https://github.com/sjy1203/GAMENet, where the name is called ndc2atc_level4.csv.
        - **drug-atc.csv**: this is a CID-ATC file, which gives the mapping from CID code to detailed ATC code (we will use the prefix of the ATC code latter for aggregation). This file is obtained from https://github.com/sjy1203/GAMENet.
        - **rxnorm2RXCUI.txt**: rxnorm to RXCUI mapping file. This file is obtained from https://github.com/sjy1203/GAMENet, where the name is called ndc2rxnorm_mapping.csv.
        - **drugbank_drugs_info.csv**: drug information table downloaded from drugbank here https://www.dropbox.com/s/angoirabxurjljh/drugbank_drugs_info.csv?dl=0, which is used to map drug name to drug SMILES string.
        - **drug-DDI.csv**: this a large file, containing the drug DDI information, coded by CID. The file could be downloaded from https://drive.google.com/file/d/1mnPc0O0ztz0fkv3HF-dpmBb8PLWsEoDz/view?usp=sharing
    - ```Output/```
        - **atc3toSMILES.pkl**: drug ID (we use ATC-3 level code to represent drug ID) to drug SMILES string dict
        - **ddi_A_final.pkl**: ddi adjacency matrix
        - **ddi_matrix_H.pkl**: H mask structure generated by using BRICS segmentation (This file is created by **ddi_mask_H.py**)
        - **ddi_matrix_H_recap.pkl**: H mask structure generated by using RECAP segmentation (This file is created by **ddi_mask_H.py**) 
        - **ehr_adj_final.pkl****: used in GAMENet baseline (if two drugs appear in one set, then they are connected)
        - **records_final.pkl**: The final diagnosis-procedure-medication EHR records of each patient, used for train/val/test split.
        - **voc_final.pkl**: diag/prod/med index to code dictionary
- ```src/```
    - **SafeDrug.py**: The SafeDrug model
    - baselines:
        - **GAMENet.py**
        - **Leap.py**
        - **Retain.py**
        - **ECC.py**
        - **LR.py**
    - setting file
        - **model.py**
        - **util.py**
        - **layer.py**
