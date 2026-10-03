# AI-Based Industrial Machine Behaviour Discovery and Early Anomaly Detection Using Unsupervised Machine Learning

## 📌 Project Overview

This project presents an AI-based system for discovering industrial machine behaviour patterns and detecting unusual machine conditions using unsupervised machine learning.

The system analyses multi-sensor time-series data from industrial machines, learns recurring behaviour states, identifies anomalous observations, studies behaviour transitions over time, and analyses the relationship between unusual behaviour and later-life machine conditions.

The project is designed as a **behaviour discovery and anomaly detection system**, rather than a direct machine failure prediction system.

---

## 🎯 Objectives

* Discover different machine behaviour patterns automatically.
* Detect unusual observations using unsupervised anomaly detection.
* Analyse machine behaviour changes over time.
* Identify behaviour states associated with later-life machine observations.
* Understand which sensor-derived features distinguish behaviour states.
* Provide an interactive dashboard for analysing machine behaviour and anomalies.

---

## 🔍 Problem Statement

Industrial machines generate large amounts of sensor data during operation. Manually analysing multiple sensor signals over time can make it difficult to identify unusual behaviour patterns.

Traditional monitoring approaches may focus mainly on predefined thresholds or known failure conditions.

This project explores an unsupervised machine learning approach that can learn patterns directly from historical sensor data and identify observations that differ from commonly observed machine behaviour.

---

## 🧠 Machine Learning Workflow

```text
Industrial Sensor Data
        ↓
Data Preprocessing
        ↓
Sensor Quality Analysis
        ↓
Feature Engineering
        ↓
Anomaly Detection
        ↓
Behaviour Clustering
        ↓
Temporal Behaviour Discovery
        ↓
Failure/Late-Life Association
        ↓
Model Validation
        ↓
AI Behaviour Explanation
        ↓
Interactive Dashboard
```

---

## 📊 Dataset

The project uses the **NASA C-MAPSS FD001** turbofan engine simulation dataset.

The FD001 training dataset contains:

* **100 simulated engines**
* **20,631 observations**
* Multiple operating settings
* Multiple sensor measurements
* Time-series observations represented using engine cycles

The dataset is used to study machine behaviour throughout its operating life.

> Note: The sensor channels in C-MAPSS are treated as anonymous sensor variables. No physical interpretation is assigned to individual sensor numbers without supporting documentation.

---

## ⚙️ Data Preprocessing

The preprocessing stage includes:

* Loading the C-MAPSS sensor data.
* Assigning meaningful column names.
* Checking missing values.
* Checking duplicate observations.
* Identifying constant features.
* Removing non-informative constant sensor columns.
* Preparing the cleaned dataset for machine learning.

### Data Quality

* Missing values: **0**
* Duplicate rows: **0**
* Constant features were identified and removed before modelling.

---

## 🛠️ Feature Engineering

To capture both sensor behaviour and temporal changes, multiple features were generated from the sensor signals.

The engineered features include:

* Original sensor values
* Sensor differences between consecutive cycles
* Rolling mean features
* Rolling standard deviation features

These features help the models capture:

* Current sensor behaviour
* Short-term changes
* Local trends
* Sensor variability

A total of **60 machine-learning features** were used for the final clustering analysis.

---

## 🚨 Anomaly Detection

An **Isolation Forest** model was used for unsupervised anomaly detection.

Configuration:

```text
Algorithm: Isolation Forest
Estimators: 100
Contamination: 2%
Random State: 42
```

### Results

* Total observations: **20,631**
* Normal observations: **20,218**
* Detected anomalies: **413**
* Anomaly rate: **2%**

The detected anomalies represent observations that are unusual compared with the learned data distribution. They are **not automatically treated as machine failures**.

---

## 🧩 Behaviour Discovery

Machine behaviour was discovered using **K-Means clustering** on the engineered sensor features.

The project uses **4 behaviour states** for detailed downstream behavioural analysis.

### Behaviour Distribution

| Behaviour State | Observations | Percentage |
| --------------- | -----------: | ---------: |
| State 0         |        6,935 |     33.61% |
| State 1         |        1,486 |      7.20% |
| State 2         |        9,415 |     45.64% |
| State 3         |        2,795 |     13.55% |

The cluster numbers are machine-generated labels and do not have an inherent physical meaning.

---

## 📈 K-Means Validation

Different values of `k` were evaluated using the silhouette score.

| Number of Clusters | Silhouette Score |
| -----------------: | ---------------: |
|                  2 |           0.2029 |
|                  3 |           0.1200 |
|                  4 |           0.1174 |
|                  5 |           0.1226 |
|                  6 |           0.1139 |
|                  7 |           0.0814 |
|                  8 |           0.0739 |

The highest silhouette score among the tested values was obtained with **k = 2**.

However, **k = 4** was retained for this project to provide finer-grained behavioural segmentation for temporal analysis, anomaly analysis, and late-life association.

The relatively low silhouette scores also indicate that the behavioural groups are not strongly separated into simple spherical clusters.

---

## ⏱️ Temporal Behaviour Discovery

The project analyses how machine behaviour states change from one cycle to another.

A total of:

**1,294 behaviour transitions**

were identified in the analysed dataset.

Examples include transitions such as:

```text
State 0 → State 2
State 2 → State 0
State 2 → State 3
State 3 → State 2
State 2 → State 1
```

Only **3 anomalies** occurred exactly during behaviour-state transitions.

Therefore, a behaviour transition by itself is not treated as an anomaly indicator.

---

## 📉 Late-Life Association

The machine operating life was divided into:

* Early Life
* Middle Life
* Late Life

The detected anomaly rates were:

| Life Stage | Anomaly Rate |
| ---------- | -----------: |
| Early      |        0.83% |
| Middle     |       0.015% |
| Late       |        5.04% |

The higher anomaly concentration in late-life observations indicates a **retrospective association between unusual behaviour and later stages of machine operation**.

This analysis does not claim that the system independently predicts an exact future failure time.

---

## 🧪 Model Validation

A retrospective near-failure analysis was performed by defining the final 20% of each engine's operating life as the near-failure window.

Results:

| Metric    |  Value |
| --------- | -----: |
| Precision |  0.862 |
| Recall    |  0.085 |
| F1 Score  | 0.1548 |

The results show that the detected anomalies have a strong association with the defined near-failure window, but the low recall means that many near-failure observations were not detected as anomalies.

Therefore, the model is positioned primarily as an **unsupervised behaviour discovery and anomaly detection system**, not as a standalone failure prediction model.

---

## 🔎 AI Behaviour Explanation

To understand the discovered behaviour states, cluster centroids and feature differences were analysed.

Some of the strongest distinguishing engineered features included:

* `sensor_14_rolling_mean`
* `sensor_14`
* `sensor_9_rolling_mean`
* `sensor_9`
* `sensor_8_rolling_mean`
* `sensor_13_rolling_mean`

These features helped identify how the discovered behaviour states differ in the learned feature space.

---

## 🖥️ Interactive Dashboard

The project includes a **Streamlit-based interactive dashboard** for analysing the discovered machine behaviour.

### Dashboard Modules

* Dashboard
* Live Monitoring
* Behaviour Discovery
* Anomaly Analysis
* Historical Analysis
* AI Insights
* Alerts

The dashboard provides:

* Machine-level monitoring
* Sensor trend visualisation
* Behaviour state information
* Anomaly information
* Behaviour timelines
* Historical analysis
* AI-based feature insights
* Machine alerts

> The "Live Monitoring" interface is a visualization interface built using the historical C-MAPSS dataset; it is not connected to live industrial hardware.

---

## 🧰 Technology Stack

### Programming

* Python

### Data Processing

* Pandas
* NumPy
* SciPy

### Machine Learning

* Scikit-learn
* Isolation Forest
* K-Means Clustering
* PCA
* StandardScaler

### Visualization

* Matplotlib
* Plotly

### Dashboard

* Streamlit

### Development Tools

* Visual Studio Code
* Jupyter Notebook
* Git
* GitHub

---

## 📁 Project Structure

```text
Industrial_Machine_AI/
│
├── data/
│   └── CMAPSSData/
│
├── dashboard/
│   └── app.py
│
├── notebooks/
│   ├── 01_data_loading.py
│   ├── 02_data_cleaning.py
│   ├── 03_data_quality.py
│   ├── 04_sensor_analysis.py
│   ├── 05_correlation_analysis.py
│   ├── 06_feature_engineering.py
│   ├── 07_anomaly_detection.py
│   ├── 08_anomaly_analysis.py
│   ├── 09_behavior_clustering.py
│   ├── 10_cluster_analysis.py
│   ├── 11_behavior_map.py
│   ├── 12_temporal_behavior.py
│   ├── 13_failure_association.py
│   ├── 14_model_validation.py
│   ├── 15_ai_behavior_explanation.py
│   ├── 16_kmeans_validation.py
│   └── 17_compare_k2_k4.py
│
├── results/
│   ├── ai_behavior_explanation.csv
│   ├── failure_association_results.csv
│   ├── k2_k4_comparison.csv
│   ├── kmeans_validation_results.csv
│   ├── model_validation_results.csv
│   └── temporal_behavior_results.csv
│
├── .gitignore
└── README.md
```

---

## ▶️ How to Run

### 1. Clone the repository

```bash
git clone https://github.com/ushanandhini16/industrial-machine-ai.git
```

### 2. Open the project

```bash
cd industrial-machine-ai
```

### 3. Install required packages

```bash
pip install pandas numpy scipy scikit-learn matplotlib plotly streamlit
```

### 4. Run the dashboard

```bash
streamlit run dashboard/app.py
```

The Streamlit dashboard will open in the browser.

---

## ⚠️ Limitations

* The analysis uses historical simulated C-MAPSS data.
* The anomaly detector is unsupervised and does not directly predict exact failure dates.
* The near-failure evaluation is retrospective and derived from the engine's recorded endpoint.
* K-Means cluster labels do not represent predefined physical machine states.
* Silhouette scores indicate relatively weak cluster separation.
* The current dashboard is not connected to real-time industrial sensors.
* Some early-cycle engineered features may be affected by limited historical window information.

---

## 🚀 Future Enhancements

Possible future improvements include:

* Real-time industrial sensor integration.
* Streaming anomaly detection.
* Online model updating.
* Remaining Useful Life estimation.
* Deep learning-based temporal modelling.
* Autoencoder-based anomaly detection.
* Advanced explainable AI techniques.
* Edge/IoT deployment for industrial monitoring.
* Cross-dataset validation using additional C-MAPSS subsets.
* Comparison with other clustering and anomaly detection algorithms.

---

## 👩‍💻 Author

**Usha Nandhini M**

B.Tech Artificial Intelligence and Data Science
Mahendra Engineering College

GitHub: https://github.com/ushanandhini16

---

## 📌 Project Summary

This project demonstrates how unsupervised machine learning can be used to analyse industrial machine sensor data, discover behavioural patterns, detect unusual observations, study temporal behaviour transitions, and investigate their association with later stages of machine operation.

The system combines **machine learning, time-series feature engineering, anomaly detection, behaviour clustering, explainable analysis, and interactive visualization** into a single industrial behaviour discovery framework.
