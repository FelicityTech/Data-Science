# Supermarket Sales Analysis & Gross Income Prediction

## 📊 Project Overview

This project analyzes supermarket sales data to predict gross income using machine learning techniques. The dataset contains 1000 transactions with features including branch location, customer type, product line, payment method, and sales metrics. The project implements a Random Forest Regressor model with comprehensive hyperparameter tuning using both GridSearchCV and RandomizedSearchCV.

## 🎯 Objectives

- Perform exploratory data analysis (EDA) on sales data
- Build a predictive model for gross income using machine learning
- Optimize model performance through hyperparameter tuning
- Evaluate model performance using multiple metrics (RMSE, MAE, R² Score)
- Save and prepare models for deployment

## 📁 Dataset

The dataset (`SuperMarket Analysis.csv`) contains **1000 transactions** with the following features:

| Feature | Description | Type |
|---------|-------------|------|
| Invoice ID | Unique invoice identifier | Object |
| Branch | Store branch (A, B, C) | Object |
| City | City location | Object |
| Customer type | Member or Normal | Object |
| Gender | Customer gender | Object |
| Product line | Product category | Object |
| Unit price | Price per unit | Float |
| Quantity | Number of items purchased | Integer |
| Tax 5% | 5% tax on transaction | Float |
| Sales | Total sales amount | Float |
| Date | Transaction date | Object |
| Time | Transaction time | Object |
| Payment | Payment method | Object |
| cogs | Cost of goods sold | Float |
| gross margin percentage | Gross margin % | Float |
| gross income | **Target variable** | Float |
| Rating | Customer rating | Float |

**Key Statistics:**
- No missing values
- 17 columns total (16 features + 1 target)
- All transactions are complete

## 🔧 Requirements

### Python Libraries

```python
pandas
numpy
matplotlib
seaborn
scikit-learn
scipy
joblib
```

### Installation

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy joblib
```

## 📂 Project Structure

```
Super-market-sale-predicting/
│
├── SuperMarket Analysis.csv          # Dataset
├── superMarketAnalysis.ipynb         # Main analysis notebook
├── random_forest_gross_income_model.pkl      # Baseline model
├── random_forest_tuned_model.pkl             # Optimized model (best)
├── grid_search_cv.pkl                       # GridSearchCV results
├── random_search_cv.pkl                     # RandomizedSearchCV results
└── README.md                                # This file
```

## 🚀 Usage

### Running the Analysis

1. **Clone or download the repository**
   ```bash
   cd Super-market-sale-predicting
   ```

2. **Ensure the dataset is in the same directory**
   - File: `SuperMarket Analysis.csv`

3. **Open and run the Jupyter Notebook**
   ```bash
   jupyter notebook superMarketAnalysis.ipynb
   ```

4. **Execute cells sequentially** to:
   - Load and explore the data
   - Perform EDA
   - Train baseline model
   - Perform hyperparameter tuning
   - Evaluate and compare models
   - Save trained models

### Loading a Saved Model

```python
import joblib

# Load the optimized model
model = joblib.load('random_forest_tuned_model.pkl')

# Make predictions
predictions = model.predict(X_new)
```

## 📈 Methodology

### 1. Data Preprocessing
- **Feature Selection**: Unit price, Quantity, Branch, Payment, Customer type
- **Categorical Encoding**: One-hot encoding with `drop_first=True`
- **Train-Test Split**: 80/20 split with `random_state=42`

### 2. Model Selection
- **Algorithm**: Random Forest Regressor
- **Rationale**: 
  - Handles non-linear relationships effectively
  - Provides feature importance insights
  - Robust to outliers
  - Excellent performance on tabular data

### 3. Baseline Model
- **Hyperparameters**: 
  - `n_estimators`: 100
  - `random_state`: 42
  - Default values for other parameters

### 4. Hyperparameter Tuning

#### GridSearchCV
- **Method**: Exhaustive search through all parameter combinations
- **Parameter Grid**:
  - `n_estimators`: [50, 100, 200, 300]
  - `max_depth`: [5, 10, 20, None]
  - `min_samples_split`: [2, 5, 10]
  - `min_samples_leaf`: [1, 2, 4]
  - `max_features`: ['sqrt', 'log2', None]
- **Total Combinations**: 432
- **Cross-Validation**: 5-fold CV
- **Scoring**: Negative Mean Squared Error

#### RandomizedSearchCV
- **Method**: Random sampling of parameter space
- **Iterations**: 50 (much faster than GridSearchCV)
- **Same parameter grid** as GridSearchCV
- **Cross-Validation**: 5-fold CV

### 5. Model Evaluation
- **Metrics Used**:
  - Root Mean Squared Error (RMSE)
  - Mean Absolute Error (MAE)
  - R² Score (Coefficient of Determination)
  - Cross-validated RMSE

## 📊 Results

### Model Performance Comparison

| Model | RMSE | MAE | R² Score |
|-------|------|-----|----------|
| **Baseline** | 0.3429 | 0.2219 | 0.9992 |
| **GridSearchCV** | 0.3375 | 0.2181 | 0.9992 |
| **RandomizedSearchCV** | 0.3375 | 0.2181 | 0.9992 |

### Key Findings

✅ **Excellent Model Performance**: 
- R² Score of **0.9992** indicates the model explains **99.92%** of variance in gross income
- Very low error rates relative to the target variable scale

✅ **Hyperparameter Tuning Results**:
- **1.57% RMSE reduction** achieved through tuning
- Best parameters found:
  - `n_estimators`: 300
  - `max_depth`: 20
  - `max_features`: None
  - `min_samples_split`: 2
  - `min_samples_leaf`: 1

✅ **Stable Performance**: 
- Cross-validated RMSE: 0.3542 ± 0.0303 (low standard deviation indicates consistency)

✅ **Statistical Insights**:
- T-test on sales by customer type: p-value = 0.0611 (borderline significant at α=0.05)

### Exploratory Data Analysis Highlights

- Sales distribution varies by branch and product line
- Customer rating distribution shows normal-like pattern
- Member vs Normal customers show slight differences in sales (statistically borderline)

## 📁 Files Description

- **`superMarketAnalysis.ipynb`**: Complete analysis notebook with EDA, modeling, and hyperparameter tuning
- **`SuperMarket Analysis.csv`**: Original dataset with 1000 transactions
- **`random_forest_gross_income_model.pkl`**: Baseline model (n_estimators=100)
- **`random_forest_tuned_model.pkl`**: Optimized model (best hyperparameters)
- **`grid_search_cv.pkl`**: Complete GridSearchCV object with all results
- **`random_search_cv.pkl`**: Complete RandomizedSearchCV object with all results

## 🔮 Future Improvements

### 1. Feature Engineering
- **Temporal Features**: Extract day of week, month, hour from Date/Time columns
- **Interaction Features**: Create features like `Unit price × Quantity`
- **Additional Features**: Include Product line, Gender, Rating, City
- **Feature Scaling**: Consider standardization if using other algorithms

### 2. Alternative Models
- **Gradient Boosting**: XGBoost, LightGBM, or CatBoost for potentially better performance
- **Neural Networks**: Deep learning models if more data becomes available
- **Ensemble Methods**: Combine multiple models (voting/stacking)

### 3. Model Interpretability
- **SHAP Values**: Use SHAP for better feature importance explanation
- **Partial Dependence Plots**: Understand feature effects on predictions
- **Model Explanation Tools**: LIME for local interpretability

### 4. Data Improvements
- **More Data**: Collect additional samples to improve generalization
- **Feature Collection**: Gather more relevant business features (promotions, seasonality, etc.)
- **Data Quality**: Check for and handle outliers more systematically

### 5. Deployment Considerations
- **API Development**: Create REST API for model serving
- **Monitoring**: Set up model performance monitoring in production
- **A/B Testing**: Compare model versions in production
- **Retraining Pipeline**: Automate model retraining with new data

### 6. Advanced Analysis
- **Customer Segmentation**: Cluster analysis on customer behavior
- **Product Analysis**: Analyze which products drive gross income
- **Seasonal Patterns**: Time series analysis for seasonal trends
- **Advanced Evaluation**: Time series validation, business-specific KPIs, error analysis

## 📝 Notes

- The model shows exceptional performance with an R² score of 0.9992, which suggests the relationship between features and gross income is highly predictable
- Hyperparameter tuning provides marginal but consistent improvement over baseline
- Both GridSearchCV and RandomizedSearchCV found similar optimal parameters
- The model is ready for deployment and can be integrated into production systems

## 👤 Author

Solomon Adegoke (FelicityTech) - Data Science Projects

## 📄 License

This project is open source and available for educational and research purposes.

---

**Last Updated**: 2026

For questions or contributions, please open an issue or submit a pull request.

