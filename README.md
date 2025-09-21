# Air Quality Forecasting Using RNNs and LSTM Models

## Project Overview

This project focuses on predicting PM2.5 concentrations using Recurrent Neural Networks (RNNs) and Long Short-Term Memory (LSTM) models. It was developed as part of the **Kaggle Beijing PM2.5 Forecasting Challenge** to predict hourly PM2.5 levels using historical air quality and meteorological data.

### Key Objectives
- Predict hourly PM2.5 concentrations using time-series forecasting
- Implement and compare different LSTM architectures
- Handle missing data and engineer relevant features
- Achieve competitive performance on Kaggle leaderboard

## Dataset Description

The dataset contains **30,676 entries** with **12 columns** comprising:

### Features:
- **Meteorological Data**: `DEWP`, `TEMP`, `PRES` (dew point, temperature, pressure)
- **Wind & Precipitation**: `Iws`, `Is`, `Ir` (wind speed, snow/rain indicators)
- **Wind Direction**: `cbwd_NW`, `cbwd_SE`, `cbwd_cv` (one-hot encoded)
- **Time Information**: `datetime`, `No` (timestamp and index)
- **Target Variable**: `pm2.5` (PM2.5 concentration - 28,755 non-null values)

## Quick Start with Google Colab

### Option 1: Direct Colab Access (Recommended)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1YEbHwOol7bmFSkxorbGYcDWYEnCMpNvx?usp=sharing)
1. **Click the "Open in Colab" badge** above to access the notebook directly
2. **Make a copy** of the notebook to your Google Drive
3. **Update the data loading section** (Cell 2) with the provided dataset links
4. **Run all cells** to execute the complete pipeline

### Option 2: Manual Upload
1. Download the notebook from this repository
2. Upload it to your Google Colab
3. Follow the same steps as Option 1

### Required Datasets
Before running the notebook, ensure you have access to these datasets:

- **Training Data**: [Download Train Dataset](https://drive.google.com/file/d/1IvU-QCa12KiAajBWZ-J0ViBPwHzqDsC7/view?usp=sharing)
- **Test Data**: [Download Test Dataset](https://drive.google.com/file/d/1f73ZBSb2TRbtwCfk4uOIPM65Lf3MWnLt/view?usp=sharing)

### Setup Instructions for Google Colab

1. **Open the Colab notebook** using the badge above
2. **In Cell 2**, replace the data loading paths with:
   ```python
   # Update these paths with the provided Google Drive links
   train_url = "https://drive.google.com/file/d/1IvU-QCa12KiAajBWZ-J0ViBPwHzqDsC7/view?usp=sharing"
   test_url = "https://drive.google.com/file/d/1f73ZBSb2TRbtwCfk4uOIPM65Lf3MWnLt/view?usp=sharing"
   ```
3. **Run all cells** sequentially to execute the complete workflow

## Data Preprocessing Pipeline

Our preprocessing approach includes:

- **Missing Data Handling**: Interpolation using forward fill (`ffill`) and backward fill (`bfill`)
- **Feature Scaling**: Normalization using `StandardScaler`
- **Feature Engineering**: Time-based features (hour, day, month, weekday/weekend)
- **Sequence Creation**: 24-hour rolling windows for LSTM input
- **Data Splitting**: Train/validation/test splits for model evaluation

## Model Architecture

### Final Selected Model: LSTM with L2 Regularization

```
Input: 24 timesteps × N features
├── LSTM(64, return_sequences=True, L2=0.0001)
├── LSTM(32, L2=0.0001)
├── Dense(16, ReLU, L2=0.0001)
├── Dense(8, ReLU)
└── Dense(1, linear)

Optimizer: Adagrad (lr=0.001)
Loss: Mean Squared Error (MSE)
Training: 64 batch size, 70 epochs
```

## Experimental Results

| Experiment | Model Type | Validation RMSE | Kaggle Score | Notes |
|------------|------------|-----------------|--------------|-------|
| 1 | BiLSTM + Dropout | 3034.24 | - | Good but overfitted |
| **2** | **LSTM + L2 Reg** | **61.82** | **~4460.55** | **Best performance** |
| 3 | Deep BiLSTM | 3894.23 | - | Overfitting issues |

## Key Findings

- ✅ **L2 regularization** significantly improved model stability and generalization
- ✅ **Simpler architectures** outperformed deeper models by avoiding overfitting
- ✅ **Preprocessing quality** was crucial for stable and reliable results
- ✅ **24-hour sequences** effectively captured temporal dependencies

## Performance Metrics

- **Final Kaggle RMSE**: ~4460.55
- **Model Selection**: Based on Kaggle leaderboard performance
- **Validation Strategy**: Time-series cross-validation approach

## Future Improvements

- [ ] **External Data Integration**: Traffic flow, satellite pollution indices
- [ ] **Advanced Architectures**: Attention mechanisms, Transformers
- [ ] **Ensemble Methods**: Combine LSTMs with XGBoost/LightGBM
- [ ] **Real-time Deployment**: Build API for live predictions

## Dependencies

The notebook automatically installs required packages:
```python
# Main libraries used
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

- **GitHub**: [NiyonshutiDavid](https://github.com/NiyonshutiDavid)
- **Project Link**: [Air Quality Forecasting Repository](https://github.com/NiyonshutiDavid/Kaggle_competition_ML-air_quality_forcasting.git)

---
