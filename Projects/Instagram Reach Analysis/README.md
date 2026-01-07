# 📊 Instagram Reach Analysis

A comprehensive data science project analyzing Instagram post performance to identify key factors influencing reach and engagement. This project combines exploratory data analysis, statistical correlation studies, and advanced machine learning ensemble methods to predict post reach with 96.78% accuracy.

![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-0.24+-green.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-Latest-lightgreen.svg)

---

## 🎯 Project Overview

This project analyzes Instagram post data to understand:
- Which engagement metrics drive the most reach
- How different traffic sources contribute to impressions
- What content strategies are most effective
- How to predict post reach before publishing

**Key Achievement**: Built an XGBoost model achieving **96.78% accuracy** in predicting Instagram post reach.

---

## 📁 Project Structure

```
Instagram Reach Analysis/
│
├── Instagram-data.csv          # Dataset (119 Instagram posts)
├── instagramAnalysis.ipynb     # Main analysis notebook
└── README.md                   # Project documentation
```

---

## 📊 Dataset Information

- **Sample Size**: 119 Instagram posts
- **Features**: 13 columns
  - **Engagement Metrics**: Likes, Comments, Shares, Saves
  - **Traffic Sources**: From Home, From Hashtags, From Explore, From Other
  - **User Actions**: Profile Visits, Follows
  - **Content**: Captions, Hashtags
  - **Target Variable**: Total Impressions
- **Data Quality**: Complete dataset with no missing values

---

## 🚀 Getting Started

### Prerequisites

- Python 3.7 or higher
- Jupyter Notebook or JupyterLab
- pip (Python package installer)

### Installation

1. **Clone the repository** (or download the project files)
   ```bash
   git clone <repository-url>
   cd "Instagram Reach Analysis"
   ```

2. **Install required packages**
   ```bash
   pip install pandas numpy matplotlib seaborn plotly wordcloud scikit-learn xgboost
   ```

   Or install from requirements file (if available):
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook
   ```

4. **Open the analysis notebook**
   - Open `instagramAnalysis.ipynb` in Jupyter

---

## 📖 Usage

### Running the Analysis

1. **Open the notebook**: `instagramAnalysis.ipynb`
2. **Run all cells**: Go to `Cell` → `Run All` (or run cells sequentially)
3. **View results**: All visualizations and model outputs will be displayed inline

### Key Sections in the Notebook

1. **Data Loading & Exploration**
   - Load and inspect the dataset
   - Check for missing values
   - Understand data types

2. **Reach Source Analysis**
   - Distribution of impressions from different sources
   - Percentage breakdown of traffic sources

3. **Content Analysis**
   - Word cloud visualizations for captions and hashtags
   - Content theme identification

4. **Relationship Analysis**
   - Correlation analysis between metrics and impressions
   - Scatter plots with trend lines

5. **Conversion Rate Analysis**
   - Calculate conversion rate (Profile Visits → Follows)
   - Visualize relationship

6. **Machine Learning Models**
   - Baseline model (PassiveAggressiveRegressor)
   - Ensemble methods (Random Forest, Gradient Boosting, AdaBoost, XGBoost)
   - Advanced ensemble methods (Voting, Stacking)
   - Model comparison and evaluation

7. **Feature Importance Analysis**
   - Identify most important features for prediction
   - Visualize feature importance

---

## 🤖 Machine Learning Models

### Model Performance Comparison

| Rank | Model | R² Score | RMSE | MAE |
|------|-------|----------|------|-----|
| 🥇 | **XGBoost** | **0.9678** | **1118.18** | **714.69** |
| 🥈 | Voting Regressor | 0.9162 | 1803.65 | 991.71 |
| 🥉 | Gradient Boosting | 0.8974 | 1996.50 | 1002.59 |
| 4th | Random Forest | 0.8810 | 2149.90 | 1130.15 |
| 5th | Passive Aggressive | 0.8239 | 2615.50 | 1224.50 |
| 6th | AdaBoost | 0.7123 | 3342.76 | 1562.74 |

### Feature Importance (XGBoost - Best Model)

1. **Follows** → 62.52% importance
2. **Likes** → 30.46% importance
3. Comments → 2.96%
4. Shares → 2.88%
5. Profile Visits → 0.66%
6. Saves → 0.51%

**Key Finding**: Follows and Likes together account for **93% of predictive power**.

---

## 💡 Key Insights

### 1. Algorithmic Distribution is Critical
- Explore section impressions show the strongest correlation (0.894) despite being only 19.2% of total impressions
- **Recommendation**: Create content optimized for Instagram's Explore algorithm

### 2. Engagement Priority Hierarchy
- **Primary Focus**: Follows (new follower acquisition) and Likes
- **Secondary**: Saves and Profile Visits
- **Lower Priority**: Shares and Comments (minimal impact on reach)

### 3. Traffic Source Breakdown
- **44.1%** from Home feed (existing followers)
- **33.6%** from Hashtags (discovery mechanism)
- **19.2%** from Explore section (algorithmic recommendations)
- **3.05%** from Other sources

### 4. Exceptional Conversion Performance
- **41% conversion rate** (Profile Visits → Follows)
- Indicates high-quality content that effectively converts visitors to followers

### 5. Strong Correlations with Impressions
- From Explore: **0.894** (strongest)
- Follows: **0.889**
- Likes: **0.850**
- From Home: **0.845**

---

## 🛠️ Technologies Used

### Data Processing
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations

### Visualization
- **Matplotlib**: Static visualizations
- **Seaborn**: Statistical visualizations
- **Plotly**: Interactive visualizations

### Text Analysis
- **WordCloud**: Text visualization

### Machine Learning
- **Scikit-learn**: 
  - PassiveAggressiveRegressor
  - RandomForestRegressor
  - GradientBoostingRegressor
  - AdaBoostRegressor
  - VotingRegressor
  - StackingRegressor
  - Evaluation metrics (R² Score, RMSE, MAE)
- **XGBoost**: XGBRegressor for gradient boosting

---

## 📈 Results & Findings

### Model Performance
- **Best Model**: XGBoost with **96.78% R² Score**
- **Improvement**: 17.5% better than baseline model
- **Error Metrics**: RMSE = 1118.18, MAE = 714.69

### Strategic Recommendations
1. **Focus on Follows and Likes**: These are the primary drivers of reach
2. **Optimize for Explore**: Content that performs well in Explore section has highest correlation with total reach
3. **Maintain Quality**: 41% conversion rate validates current content strategy
4. **Use Predictive Model**: Leverage the model to forecast reach before posting

---

## 🔮 Future Enhancements

1. **Model Optimization**
   - Hyperparameter tuning for further performance gains
   - Cross-validation strategies for better generalization

2. **Feature Engineering**
   - Temporal features (posting time, day of week, seasonality)
   - Caption analysis (sentiment, length, readability)
   - Hashtag metrics (count, diversity, trending status)

3. **Advanced Analytics**
   - Time series analysis for trend prediction
   - A/B testing framework for content optimization
   - Real-time monitoring dashboard

4. **Deployment**
   - Web application for interactive predictions
   - API development for integration with content management systems
   - Automated reporting and insights generation

---

## 📝 Usage Example

### Making Predictions

```python
import numpy as np
import xgboost as xgb

# Load trained model (after training in notebook)
# model = xgb.XGBRegressor(...)
# model.fit(xtrain, ytrain)

# Features: [Likes, Saves, Comments, Shares, Profile Visits, Follows]
features = np.array([[282.0, 233.0, 4.0, 9.0, 165.0, 54.0]])

# Predict impressions
prediction = model.predict(features)
print(f"Predicted Impressions: {prediction[0]:.2f}")
```

---

## 📊 Visualizations

The notebook includes various visualizations:
- Distribution plots for different traffic sources
- Word clouds for captions and hashtags
- Scatter plots with trend lines for relationships
- Pie charts for traffic source breakdown
- Model performance comparison charts
- Feature importance visualizations
- Predicted vs. Actual scatter plots
- Residual plots

---

## 🤝 Contributing

Contributions are welcome! If you'd like to contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

## 👤 Author

**Your Name**
- GitHub: [@FelicityTech](https://github.com/FelicityTech)
- LinkedIn: [Solomon Adegoke](https://www.linkedin.com/in/solomon-eniola-adegoke/)

---

## 🙏 Acknowledgments

- Instagram for providing the data structure
- The open-source community for excellent ML libraries
- Contributors and reviewers of this project

---

## 📞 Contact

For questions, suggestions, or collaborations, please open an issue or contact the author.

---

## ⭐ Show Your Support

If you find this project helpful, please give it a ⭐ on GitHub!

---

**Last Updated**: Jan, 2026

**Status**: ✅ Active Development

