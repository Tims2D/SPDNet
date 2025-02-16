# SPDNet: Seasonal-Periodic Decomposition Network for Advanced Residential Demand Forecasting

**SPDNet** is a deep learning framework designed for **individual residential electricity demand forecasting**. The model effectively captures intricate temporal variations, including multiple seasonalities, periodicities, and abrupt fluctuations, using **Seasonal-Trend Decomposition Module (STDM)** and **Periodical Decomposition Module (PDM).**

---

## 📁 Repository Structure

### **1. Data Provider**
This folder contains scripts for **data preprocessing** and **loading datasets** for training and inference.

#### 🔹 `data_loader.py`
- Handles dataset preprocessing for training and evaluation.
- Reads the dataset, scales the values, handles missing data, and prepares sequences for forecasting.
- Implements **time encoding** to include time-based features in the dataset.

#### 🔹 `data_factory.py`
- Selects and loads the appropriate dataset class (`Dataset_Custom` or `ML_DataLoader`) based on the user’s configuration.
- Creates dataset instances and prepares **PyTorch DataLoader** for efficient batch processing.

---

### **2. Experiment (exp)**
The **`exp`** folder is responsible for **forecasting electricity demand and selecting models** for evaluation. It contains two key scripts:

#### 🔹 `exp_basic.py`
- Defines the **base experimental setup** for different models.
- Supports multiple deep learning architectures such as **LSTM, GRU, Transformer, Informer, TimesNet, PatchTST**, and SPDNet.
- Handles **device allocation (CPU/GPU)** and model initialization.

#### 🔹 `exp_forecasting.py`
- Implements **training, validation, and testing** for different forecasting models.
- Integrates **optimizer selection, loss functions (MSELoss), and learning rate scheduling**.
- Monitors **RAM and GPU memory usage** during training for efficiency analysis.
- Supports **early stopping** and **automatic model checkpointing** to prevent overfitting.
- Generates and saves **forecasting results**, including predictions and evaluation metrics (MSE, MAE, RMSE).

---

## 📜 Getting Started

### **Installation**
To use SPDNet, clone the repository and install the required dependencies:
```bash
git clone https://github.com/your_username/SPDNet.git
cd SPDNet
pip install -r requirements.txt
