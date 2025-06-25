
# TriageTech: AI-Based Medical Triage System

TriageTech is a smart patient triaging tool designed to prioritize patient treatment using real-time sensor data and AI-driven classification. This project uses a machine learning model to categorize patients into emergency classes based on vitals like temperature, SpO₂, blood pressure, heart rate, and age.

##  Project Overview

The system includes:
- A serial interface to collect real-time vitals from medical sensors.
- An MLP-based neural network model to predict triage priority.
- A web-based medical form for user input.
- An automated Excel-based priority list generator.

##  File Structure

| File | Description |
|------|-------------|
| `ANN.py` | Main AI pipeline: trains an MLP classifier on vitals data, predicts patient class, and saves a prioritized list. |
| `pyserial_final.py` | Collects sensor data via serial port and appends it to `sensor_data.xlsx`. |
| `TriageForm2.html` | Front-end form to collect patient metadata and symptoms. |
| `sensor_data_training.xlsx` | Training dataset for model training. |
| `sensor_data.xlsx` | Live input from sensors during testing. |
| `patient_priority_list.xlsx` | Output file with predicted triage priorities. |

##  Model Summary

- **Type**: Multi-layer Perceptron (MLP)
- **Framework**: TensorFlow / Keras
- **Input Features**: Temperature, SpO₂, BP, HR, Age
- **Classes**:
  - Class A: Emergency (0–5 min)
  - Class B: Urgent (10–20 min)
  - Class C: Non-Urgent (30+ min)
- **Output**: Severity class, triage priority, and estimated wait time

##  How to Run

### 1. Prerequisites

```bash
pip install pandas numpy openpyxl matplotlib scikit-learn keras tensorflow pyserial
```

### 2. Collect Sensor Data

Run this script to log real-time data from a serial device (e.g., Arduino with sensors):

```bash
python pyserial_final.py
```

Ensure the device outputs data in the format:
```
Temperature=36.5
SpO2=98
Heart Rate=72
```

### 3. Train and Predict Using ANN

Train the neural network and generate the triage priority list:

```bash
python ANN.py
```

This will output `patient_priority_list.xlsx` with predictions.

### 4. Web Form Usage (Optional)

Open `TriageForm2.html` in a browser to input patient details manually. Submissions will be sent to a connected backend or script (Google Apps Script in this case).

##  Output Example

| Temp | SpO₂ | BP | HR | Age | Severity | Priority | Wait Time |
|------|------|----|----|-----|----------|----------|-----------|
| 39.0 | 85   | 190| 105| 72  | Emergency (A) | 1 | 0-5 min |
| 37.0 | 92   | 160| 90 | 55  | Urgent (B)    | 2 | 10-20 min |
| 36.5 | 98   | 120| 72 | 24  | Non-Urgent (C)| 3 | 30+ min |

##  Highlights

- Real-time triage support using AI
- Dynamic input via serial port
- GUI-ready HTML form
- Excel integration for real-world hospital use

##  Tech Stack

- Python
- TensorFlow/Keras
- Pandas / NumPy
- pySerial
- HTML + Bulma CSS
- Google Apps Script (form backend)

