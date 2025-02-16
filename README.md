# SPDNet: Seasonal-Periodic Decomposition Network for Advanced Residential Demand Forecasting

**SPDNet** is a deep learning framework designed for **individual residential electricity demand forecasting**.  


# SPDNet: Seasonal-Periodic Decomposition Network for Advanced Residential Demand Forecasting

**SPDNet** is a deep learning framework designed for **individual residential electricity demand forecasting**. The model effectively captures intricate temporal variations, including multiple seasonalities, periodicities, and abrupt fluctuations, using **Seasonal-Trend Decomposition Module (STDM)** and **Periodical Decomposition Module (PDM).**

---

## 📁 Repository Structure

### **1. Data Provider**
This folder contains scripts for **data preprocessing** and **loading datasets** for training and inference.

#### 🔹 `data_loader.py`
- Handles dataset preprocessing for training and evaluation.
- Reads the dataset, scales the values, handles missing data, and prepares sequences for forecasting.
- Supports different feature modes:  
  - `'S'` (single target variable)
  - `'M'` (multiple input features)
  - `'MS'` (multi-variable with target included).
- Implements **time encoding** to include time-based features in the dataset.

#### 🔹 `data_factory.py`
- Selects and loads the appropriate dataset class (`Dataset_Custom` or `ML_DataLoader`) based on the user’s configuration.
- Creates dataset instances and prepares **PyTorch DataLoader** for efficient batch processing.
- Supports different datasets by mapping dataset names to their corresponding dataset classes.

---
