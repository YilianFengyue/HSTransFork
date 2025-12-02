# HSTrans
Here, we introduce HSTrans, an end-to-end homogeneous substructures transformer network designed for
predicting drug-side effect frequencies. 

# Environment
* python == 3.8.5
* pytorch == 1.6
* Numpy == 1.21.0
* scikit-learn == 0.23.2
* scipy == 1.5.0
* rdkit == 2020.03.6
* matplotlib == 3.3.1
* networkx == 2.5

# Files:
1.original_data

This folder contains our original side effects and drugs data.

* **Supplementary Data 1.txt:**     
  The standardised drug side effect frequency classes used in our study. 


* **Supplementary Data 2.txt:**  
The postmarketing drug side effect associations from SIDER and OFFSIDES.
  

* **Supplementary Data 3.txt:**  
Main or High‐Level Term (HLT) Medical Dictionary for Regulatory Activities (MedDRA) terminology classification for each side effect term.
  

* **Supplementary Data 4.txt:**  
High‐Level Group Term Medical Dictionary for Regulatory Activities (MedDRA) terminology classification for each side effect term.
  


2.data
* **drug_codes_chembl_freq_1500.txt**   
Corpus of drug substructures used for learning BPE encoder.


* **drug_side.pkl**     
Frequency matrix of side effects of 750 drugs.


* **drug_SMILES_750.csv**   
SMILES of 750 drugs.


* **drug_SMILES_759.csv**   
SMILES files for 750 initial drugs and 9 independent test sets, and the position of 9 drugs is before 750 drugs.


* **raw_frequency_750.mat**  
The original frequency matrix of side effects of 750 drugs, including frequency matrix 'R', drug name 'drugs' and side effect name' sideeffect '.


* **frequency_750+9.mat**      
The original frequency matrix of side effects of drugs in 750 drugs and 9 independent test sets, and the position of 9 drugs is before 750 drugs.
  

* **data/SE_sub_index_50.npy**  
Effective substructures extracted for each side effect.


* **SE_sub_mask_50.npy**  
Substructure mask matrix for side effects.


* **side_effect_label_750.mat**  
Label vector for 994 side effects.

* **subword_units_map_chembl_freq_1500.csv**  
The substructures learned by the BPE encoder and their corresponding indices.


# Code 

main.py: Test of 750 drugs.

Net.py: It defines the model used by the code.

Encoder.py: It defines transformer encoder.

smiles2vector.py: It defines a method to calculate the smiles of drugs as vertices and edges of a graph.

utils.py: It defines performance indicators.


# Run

epoch: Define the number of epochs.

lr: Define the learning rate.

lamb: Define weights for unknown associations.


Example:
```bash
python main.py --save_model --epoch 300 --lr 0.0001
```

# Contact
If you have any questions or suggestions with the code, please let us know. Contact Kaiyi Xu at xuky@cug.edu.cn