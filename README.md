# ds_301_mid_term_project
DS 301: Mid-Term Project

# 1.Overview of Mid-term project 

## Classification of poisonous mushrooms using High-Performance Support Vector Machines

### Project member

Shigeo,Hina,Emilio

### Dataset
Mushroom dataset included 8124 rows × 23 columns categorical data
It consisted of 22 independent features and 1 target feature which mentioned poisonous mushroom or not

### Purpose of the project

Confirmimng the usefulness and scalability of the High-Performance Support Vector Machine (HPSVM) algorithm proposed by He et al. by applying it to various datasets, including the mushroom dataset, and by comparing its accuracy with that of other algorithms.

### Project Processing 

- Starting project : 2025/3/10
- Preprocessing : 2025/3/11 (Hina, Emilio)
- Reproduce the paper : 2025/3/11 (Shigeo)
- Improvements : 2025/3/12
- ・Testing methodology to different datasets : Shigeo
- ・Experimenting with different parameters : Emilio
- ・Try different classification model : Hina
- Merge to the main : 2025/3/13

### Directory components


```
.
├── LICENSE
├── README.md
├── data
│   ├── hate_speech_and_offensive_language_dataset.csv
│   ├── mushroom_dataset.csv
│   ├── preprocessing_mushroom.csv
│   └── wholesale_customers_data.csv
└── models
    ├── HPSVM_model.ipynb
    ├── SVMs.ipynb
    ├── hpsvm.py
    ├── hpsvm_application.ipynb
    ├── mushroom-ML.ipynb
    └── mushroom.ipynb
```


### Environments

#### Necesarry tool and version
- Python 3.11.7
- pip 23.3.1
- Jupyter Lab 4.0.11

### Create and Activate a Virtual Environment (Recommended)

--bash
##### Create a virtual environment ("myenv" can be any name you prefer)
python -m venv myenv

##### Activate the virtual environment (Mac/Linux)
source myenv/bin/activate

##### Activate the virtual environment (Windows)
myenv\Scripts\activate

### How to Set up Jupyer lab
##### Install Jupyter Lab and the kernel package
pip install jupyterlab ipykernel

##### Register the virtual environment in Jupyter Lab
python -m ipykernel install --user --name=myenv --display-name "Python (myenv)"

### Neccesary libraries
pip install ucimlrepo
!pip install --upgrade xgboost



