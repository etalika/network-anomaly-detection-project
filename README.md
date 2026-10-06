# Network Anomaly Detection Project

## How to Run

### Project Structure
```text
network-anomaly-project/
│
├── data/
│   ├── README.md
│   └── raw/
│       └── synthetic_network_flows.csv
│
├── results/
│   ├── figures/
│   ├── predictions/
│   └── tables/
│
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── anomaly_detector.py
│   ├── evaluation.py
│   ├── anomaly_type_analysis.py
│   ├── false_positive_analysis.py
│   ├── generate_traceable_results.py
│   └── visualization.py
│
├── requirements.txt
└── README.md
‍‍‍
``` 

---

### 1. Get the code from the repository

Download or clone the project repository, then open the terminal in the project root directory.

The project root must contain the following folders and files:

text
data/
src/
requirements.txt

---

### 2. Create a virtual environment

Use this command:

powershell
python -m venv .venv

Activate the virtual environment:

powershell
.\.venv\Scripts\Activate.ps1

If PowerShell blocks the activation, use:

powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1

---

### 3. Install dependencies

Use this command:

powershell
python -m pip install -r .\requirements.txt

---

### 4. Check the raw data

Make sure the raw dataset exists at:

text
.\data\raw\synthetic_network_flows.csv

---

### 5. Preprocess the data

Run:

powershell
python .\src\preprocessing.py

---

### 6. Run the model

Run the Isolation Forest model:

powershell
python .\src\anomaly_detector.py

The prediction file will be saved in:

text
.\results\predictions\isolation_forest_predictions.csv

---

### 7. Evaluate the model results

Run:

powershell
python .\src\evaluation.py

The evaluation results will be saved in:

text
.\results\tables\

---

### 8. Analyze anomaly types

Run:

powershell
python .\src\anomaly_type_analysis.py

---

### 9. Analyze false positives

Run:

powershell
python .\src\false_positive_analysis.py

---

### 10. Generate traceable predictions

Run:

powershell
python .\src\generate_traceable_results.py

The traceable predictions will be saved in:

text
.\results\predictions\traceable_predictions.csv

---

### 11. Generate visualizations

Run:

powershell
python .\src\visualization.py

The generated figures will be saved in:

text
.\results\figures\

---

## Run All Commands

After activating the virtual environment, run the following commands in order:

powershell
python .\src\preprocessing.py
python .\src\anomaly_detector.py
python .\src\evaluation.py
python .\src\anomaly_type_analysis.py
python .\src\false_positive_analysis.py
python .\src\generate_traceable_results.py
python .\src\visualization.py
