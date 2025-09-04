# MADD

## Query Examples

### 🧪 Start communication with
```python
"What can you do?"
```
### 🧪Dataset preparation
```python
"Download data from ChemBL on KRAS protein with IC50 values."
"Download data from BindingDB on KRAS protein with IC50."
```
### 🧪AutoML/DL
```python
"Run training of the generative model to predict IC50 on my attached data."
"Check the status of the training for the 'IC50_predictor'."
"Generate molecules to inhibit the KRAS G12C protein mutation, focusing on selective binding and ensuring that HRAS and NRAS are not affected."
"Create compounds with electron-donating groups that can stabilize radical intermediates, reducing oxidative damage in neurons."
"Predict IC50 for CN(C)CCOC(C1=CC=CC=C1)C1=CC=CC=C1 by 'IC50_predictor' model."
```
## Agent system diagram
![Multi-Agent System](./diagram.png)

### Environment Setup for Development
```commandline
pip install .
```

### Start MADD with GUI
Run in CLI:
```commandline
streamlit run MADD/streamlit_app.py
```

You should look something like thise:
```commandline
You can now view your Streamlit app in your browser.

Local URL: http://localhost:8501
Network URL: http://192.168.0.12:8501
```
Click on the URL, the link will open.










