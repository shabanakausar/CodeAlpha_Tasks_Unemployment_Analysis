# CodeAlpha_Tasks_Unemployment_Analysis

📊 Unemployment Rate Analysis & Prediction
Unemployment Dashboard

🔍 Project Overview
This project analyzes and predicts unemployment rates using:

Exploratory Data Analysis to uncover trends and COVID-19 impacts

Machine Learning (Random Forest, Lasso, Ridge) for accurate predictions

Automated Reporting with PDF generation

🚀 Key Features
✅ Comprehensive EDA with interactive visualizations

✅ COVID-19 impact analysis across regions

✅ Multiple ML models comparison

✅ Best model achieves R² score of 0.65

✅ Automated PDF report generation

📂 Dataset
Unemployee.csv containing:

Target: Estimated Unemployment Rate (%)

Features: Region, Area, Employment stats, Date

Time Period: Pre and post COVID-19 data

🛠️ Requirements
bash
pip install pandas numpy matplotlib seaborn scikit-learn reportlab
💻 Usage
Run analysis:

python
python unemployment_analysis.py
The script will:

Perform EDA and generate visualizations

Train and compare ML models

Save best model as best_model.pkl

Create Data_Science_Summary_Report.pdf

📈 Results
Model	R² Score	MAE	RMSE
Random Forest	0.6523	1.42	2.15
Ridge	0.078	2.87	3.92
Lasso	0.076	2.89	3.94
🔑 Key Findings
COVID-19 increased unemployment by 5-15% across regions

Rural areas showed greater volatility than urban

Random Forest outperformed linear models significantly
