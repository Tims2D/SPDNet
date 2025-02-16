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
---

## 📂 Models
The **`models`** folder contains various deep learning architectures used in this study. These models are responsible for learning patterns from electricity demand data and generating predictions. 

The **output** of each model is passed to `exp/exp_forecasting.py` for **training, validation, and evaluation**.

---

### **Implemented Models**
SPDNet is benchmarked against various **traditional, advanced, and state-of-the-art** forecasting models:

#### 🔹 **Recurrent Neural Networks (RNNs)**
- `LSTM.py`: Long Short-Term Memory network, designed to capture **long-term dependencies**.
- `GRU.py`: Gated Recurrent Unit, optimized for **faster training and reduced complexity**.
- `BiLSTM.py`: Bidirectional LSTM, processes sequences **forward and backward** to enhance feature extraction.
- `ConvLSTM.py`: A hybrid model combining **CNNs with LSTMs** to capture **spatial-temporal dependencies**.

#### 🔹 **Convolutional-Based Models**
- `CNN.py`: Captures **local temporal features** in time series data.
- `ResLSTM.py`: Integrates **residual connections** into LSTM for enhanced gradient flow.
- `Conv2DLSTM.py`: Utilizes **2D convolutions** along with LSTM layers to improve pattern recognition.

#### 🔹 **Transformer-Based Models**
- `Transformer.py`: Implements the **vanilla transformer** for time series forecasting.
- `Informer.py`: An efficient transformer variant designed for **long-sequence forecasting**.
- `Crossformer.py`: Captures **cross-dimensional dependencies** to improve prediction accuracy.

#### 🔹 **State-of-the-Art Forecasting Models**
- `PatchTST.py`: Employs **patch-based input encoding** for better handling of sequential patterns.
- `TimesNet.py`: A **period-aware** model optimized for **long-term electricity forecasting**.
- `DLinear.py`: A **linear decomposition-based** time series model with fast training.

#### 🔹 **Our Proposed Model: SPDNet**
- `SPDNet.py`: 
  - **Captures seasonality and periodicity** through the **Seasonal-Trend Decomposition Module (STDM)**.
  - **Extracts dominant periods** using the **Periodical Decomposition Module (PDM)**.
  - **Combines Conv1D, Conv2D, and Transformer Encoder** to capture short-term, intra-period, and global dependencies.

---

## 🛠 How Models Work
1. **Each model processes the input time series** and outputs forecasted electricity demand.
2. **The outputs are then passed to** `exp/exp_forecasting.py` for training, validation, and evaluation.
3. **The best-performing model is selected based on MSE and MAE metrics**.

---

🔜 **Next: Training and Evaluation Scripts**
We now move to the `utils` folder, which provides essential tools for **data processing, metrics calculation, and visualization**.



## 📜 Getting Started

### **Installation**
To use SPDNet, clone the repository and install the required dependencies:
```bash
git clone https://github.com/your_username/SPDNet.git
cd SPDNet
pip install -r requirements.txt
