# Human Activity Recognition using Smartphone Data

## Overview
This project demonstrates human activity recognition using machine learning on smartphone sensor data. The goal is to classify activities such as walking, sitting, standing, laying, walking upstairs, and walking downstairs based on accelerometer and gyroscope measurements.

## Dataset
- **Source**: UCI Human Activity Recognition Using Smartphones Dataset
- **Description**: The dataset contains 561 features derived from smartphone sensors (accelerometer and gyroscope) collected from 30 subjects performing 6 different activities.
- **Activities**: WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING
- **Files**: 
  - `train-1.csv`: Training data
  - `test.csv`: Test data

## Installation
1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Jupyter notebook: `human-activity-recognition.ipynb`

## Project Structure
```
human-activity-recognition/
├── data/
│   ├── train-1.csv
│   └── test.csv
├── notebooks/
│   └── human-activity-recognition.ipynb
├── models/  # Saved trained models
├── README.md
└── requirements.txt
```

## Methodology
1. **Data Exploration**: Understanding the dataset, distributions, and correlations
2. **Preprocessing**: Feature scaling and dimensionality reduction using PCA
3. **Model Training**: Comparing multiple algorithms (KNN, Random Forest, SVM, etc.)
4. **Evaluation**: Cross-validation, accuracy, precision, recall, F1-score, confusion matrix

## Key Findings
- Best performing model: [To be filled after analysis]
- Accuracy achieved: [To be filled]
- Important features: [Analysis results]

## Technologies Used
- Python 3.x
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- Jupyter Notebook

## Future Improvements
- Implement deep learning models (CNNs for time-series data)
- Deploy as a web application
- Real-time activity recognition

## License
This project is for educational purposes. Dataset license follows UCI terms.