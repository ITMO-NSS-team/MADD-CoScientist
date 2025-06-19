# CoScientist

## ChemCoScientist Query Examples

### 🧪 Start communication with
```python
"What can you do?"
```
### 🧪Dataset preparation
```python
"Download data from ChemBL on the MEK1 protein with IC_50 calculations. Be sure to prepare them for training - remove junk data"
"Prepare data for training from the file ./data_dir_for_coder/ChEMBL_data.xlsx - delete all values ​​where docking_score > -6."
"Download data from BindingDB on MEK1 protein with Ki calculations. Remove junk data."
```
### 🧪AutoML/DL
```python
"Run training of the generative model on data from ./data_dir_for_coder/processed_MEK1_IC50_data.xlsx , specify the IC50 target, name the case MEK1."
"Check the status of the training for the MEK1 case"
"Start generating molecules for the MEK1 case."
"Predict the properties of COc1ccc(-c2cc3ncn(C)c(=O)c3c(NC3CC3)n2)cc1OC using the MEK1 ml model."
"Find out for which cases there are generative models ready for inference?"
```
## Agent system diagram
![Multi-Agent System](https://github.com/ITMO-NSS-team/CoScientist/tree/main/ChemCoScientist/diagram.png)







