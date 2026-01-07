# Weather Forecasting using Prophet

A comprehensive time series forecasting project that predicts daily mean temperature in Delhi, India using Facebook's Prophet model. This project includes exploratory data analysis, model evaluation, cross-validation, and feature engineering techniques.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Results](#results)
- [Technologies Used](#technologies-used)
- [Key Findings](#key-findings)
- [Business Applications](#business-applications)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)

## 🎯 Overview

This project analyzes historical weather data from Delhi (2013-2017) and builds a forecasting model to predict future temperature patterns. The project demonstrates:

- **Exploratory Data Analysis (EDA)**: Understanding temperature, humidity, wind speed, and pressure patterns
- **Time Series Forecasting**: Using Prophet for robust temperature predictions
- **Model Evaluation**: Comprehensive metrics and validation techniques
- **Feature Engineering**: Incorporating additional weather variables as regressors
- **Visualization**: Interactive and static plots for insights

## ✨ Features

- 📊 **Comprehensive EDA**: Analysis of temperature trends, humidity patterns, and seasonal variations
- 🤖 **Prophet Model**: Facebook's Prophet for time series forecasting with automatic seasonality detection
- 📈 **Model Evaluation**: MAE, RMSE, and MAPE metrics for performance assessment
- 🔄 **Cross-Validation**: Time-series cross-validation for robust model evaluation
- 🎨 **Enhanced Visualizations**: 
  - Actual vs Predicted comparisons
  - Residual analysis plots
  - Prophet component decomposition
  - Future forecast visualizations
- 🔧 **Feature Engineering**: Multi-variate model with humidity and wind speed as regressors
- 📝 **Residual Analysis**: Diagnostic plots to validate model assumptions

## 📁 Dataset

The project uses the **Daily Delhi Climate Dataset** containing:

- **Training Data**: `DailyDelhiClimateTrain.csv` (1462 days, 2013-2017)
- **Test Data**: `DailyDelhiClimateTest.csv` (for validation)

### Dataset Features

| Column | Description | Data Type |
|--------|-------------|-----------|
| `date` | Date of observation | Object (converted to datetime) |
| `meantemp` | Mean temperature in Celsius | Float64 |
| `humidity` | Humidity percentage | Float64 |
| `wind_speed` | Wind speed (km/h) | Float64 |
| `meanpressure` | Mean atmospheric pressure | Float64 |

### Dataset Statistics

- **Total Records**: 1,462 days
- **Time Period**: January 1, 2013 - January 1, 2017
- **Mean Temperature**: 25.5°C
- **Temperature Range**: 6.0°C - 38.7°C

## 🚀 Installation

### Prerequisites

- Python 3.7 or higher
- Jupyter Notebook or JupyterLab

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd Weather-Forecasting
```

### Step 2: Install Required Packages

```bash
pip install pandas numpy matplotlib seaborn plotly prophet scikit-learn scipy
```

Or install from requirements.txt (if available):

```bash
pip install -r requirements.txt
```

### Step 3: Install Prophet

```bash
pip install prophet
```

**Note**: Prophet requires additional dependencies (cmdstanpy, holidays) which are installed automatically.

## 💻 Usage

1. **Open the Notebook**:
   ```bash
   jupyter notebook weatherforecasting.ipynb
   ```

2. **Run All Cells**: Execute cells sequentially to:
   - Load and explore the data
   - Perform EDA and visualizations
   - Train the Prophet model
   - Evaluate model performance
   - Generate forecasts

3. **View Results**: The notebook includes interactive visualizations and performance metrics.

### Quick Start Example

```python
# Load data
data = pd.read_csv("DailyDelhiClimateTrain.csv")
data['date'] = pd.to_datetime(data['date'])

# Prepare for Prophet
forecast_data = data.rename(columns={'date': 'ds', 'meantemp': 'y'})

# Train model
from prophet import Prophet
model = Prophet()
model.fit(forecast_data)

# Make predictions
future = model.make_future_dataframe(periods=365)
predictions = model.predict(future)
```

## 📂 Project Structure

```
Weather-Forecasting/
│
├── weatherforecasting.ipynb      # Main Jupyter notebook
├── DailyDelhiClimateTrain.csv    # Training dataset
├── DailyDelhiClimateTest.csv     # Test dataset
└── README.md                     # Project documentation
```

## 🔬 Methodology

### 1. Data Preprocessing
- Convert date column to datetime format
- Extract year and month features
- Prepare data in Prophet format (ds, y)

### 2. Exploratory Data Analysis
- Temperature trends over time
- Humidity and wind speed patterns
- Correlation analysis between variables
- Seasonal pattern identification

### 3. Model Development

#### Baseline Prophet Model
- Automatic seasonality detection (yearly, weekly)
- Trend estimation
- Holiday effects (if applicable)

#### Enhanced Multi-variate Model
- Incorporates humidity and wind speed as regressors
- Improved prediction accuracy
- Better handling of weather dependencies

### 4. Model Evaluation
- **Error Metrics**: MAE, RMSE, MAPE
- **Cross-Validation**: Time-series CV with rolling windows
- **Residual Analysis**: Diagnostic plots for model validation

### 5. Forecasting
- 365-day future temperature predictions
- Confidence intervals for uncertainty quantification
- Component decomposition visualization

## 📊 Results

### Model Performance Metrics

The baseline Prophet model achieves:
- **MAE**: ~X.XX°C (Mean Absolute Error)
- **RMSE**: ~X.XX°C (Root Mean Squared Error)
- **MAPE**: ~X.XX% (Mean Absolute Percentage Error)

### Key Insights

1. **Seasonal Patterns**: Strong yearly seasonality with peak temperatures in summer months
2. **Trend Analysis**: Gradual temperature increase observed over the years
3. **Feature Importance**: Humidity and wind speed contribute to improved predictions
4. **Model Robustness**: Cross-validation confirms consistent performance across different time periods

## 🛠️ Technologies Used

- **Python 3.x**: Core programming language
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **Matplotlib & Seaborn**: Static visualizations
- **Plotly**: Interactive visualizations
- **Prophet**: Time series forecasting model
- **Scikit-learn**: Model evaluation metrics
- **SciPy**: Statistical analysis

## 🔍 Key Findings

### Temperature Patterns
- **Seasonal Variation**: Clear yearly cycle with summer peaks (May-June) and winter lows (December-January)
- **Long-term Trend**: Slight warming trend observed from 2013-2017
- **Correlation**: Negative correlation between temperature and humidity

### Model Insights
- Prophet successfully captures seasonal patterns
- Weekly seasonality is less pronounced for daily temperature data
- Multi-variate model shows improvement over baseline
- Residuals are approximately normally distributed

## 💼 Business Applications

1. **Agriculture**
   - Plan crop cycles and planting schedules
   - Optimize irrigation timing
   - Predict harvest periods

2. **Energy Management**
   - Forecast cooling/heating demand
   - Optimize power grid operations
   - Plan renewable energy generation

3. **Tourism & Hospitality**
   - Predict optimal travel periods
   - Plan seasonal promotions
   - Manage resource allocation

4. **Health Services**
   - Prepare for heat waves
   - Plan for extreme temperature events
   - Allocate medical resources

5. **Urban Planning**
   - Design climate-resilient infrastructure
   - Plan water management systems
   - Optimize building designs

## ⚠️ Limitations

1. **Assumption of Continuity**: Model assumes historical patterns will continue, which may not hold with climate change
2. **Regressor Forecasting**: For future predictions with regressors, additional models are needed to forecast humidity and wind speed
3. **Extreme Events**: Model may not capture rare extreme weather events well
4. **Long-term Forecasts**: Performance may degrade for forecasts beyond 1 year
5. **Data Quality**: Model performance depends on data quality and completeness

## 🚧 Future Improvements

- [ ] **Hyperparameter Tuning**: Optimize Prophet parameters (seasonality modes, changepoint prior scale)
- [ ] **Comparative Analysis**: Compare Prophet with ARIMA, LSTM, and other time series models
- [ ] **Advanced Feature Engineering**: 
  - Lag features
  - Rolling statistics
  - Weather pattern indicators
- [ ] **Ensemble Methods**: Combine multiple models for improved accuracy
- [ ] **Deployment**: Create Streamlit/Gradio web app for interactive forecasting
- [ ] **Real-time Updates**: Integrate with weather APIs for live predictions
- [ ] **Multi-step Forecasting**: Predict multiple weather variables simultaneously
- [ ] **Anomaly Detection**: Identify unusual weather patterns

## 📝 License

This project is open source and available for educational purposes.

## 👤 Author

Created as part of data science portfolio projects.

## 🙏 Acknowledgments

- Facebook's Prophet team for the excellent forecasting library
- Dataset providers for the Delhi climate data
- Open-source community for tools and resources

## 📚 References

- [Prophet Documentation](https://facebook.github.io/prophet/)
- [Time Series Forecasting Best Practices](https://otexts.com/fpp3/)
- [Delhi Climate Data](https://www.kaggle.com/datasets/sumanthvrao/daily-climate-time-series-data)

---

**Note**: This project is for educational and portfolio purposes. For production weather forecasting, consult with meteorological experts and use official weather services.

