# ARIMA_Gold
Forecasting Gold Price using Gradient Boosting Machines (XGBOOST), Decision Trees and Random Forests and ARIMA

### **Steps for Forecasting Gold Price**

#### **Step 1: Data Analysis**
1. **Data Collection:** Gather historical gold prices and relevant external factors.
2. **Preprocessing:** Handle missing values, convert dates, and engineer features (e.g., lags, rolling statistics).
3. **EDA:** Visualize trends, seasonality, and correlations. Test for stationarity.
4. **Data Split:** Divide into train-test sets, ensuring temporal order.

#### **Step 2: Decision Trees & Random Forests**
1. **Decision Trees:**
   - Train a Decision Tree Regressor.
   - Tune parameters (e.g., depth).
   - Evaluate using MAE, RMSE, feature importance.

2. **Random Forests:**
   - Train a Random Forest Regressor.
   - Optimize parameters (e.g., number of trees).
   - Assess performance and feature importance.

#### **Step 3: Gradient Boosting Machines (XGBoost)**
1. **Feature Engineering:** Enhance features with interaction terms, advanced rolling stats.
2. **Model Training:**
   - Train XGBoost with hyperparameter tuning (e.g., learning rate, depth).
   - Use cross-validation.
   - Evaluate using MAE, RMSE, and SHAP values for feature importance.

#### **Step 4: ARIMA**
1. **Preprocessing:** Ensure stationarity, possibly by differencing.
2. **Model Training:**
   - Fit ARIMA using identified lags (p, d, q).
   - Diagnose residuals for model fit.
   - Evaluate with MAE, RMSE, AIC/BIC.

#### **Step 5: Rating Models**
1. **Compare Models:** Rank based on MAE, RMSE, and complexity.
2. **Ensemble Option:** Consider combining models for improved accuracy.
3. **Final Selection:** Choose the best model for deployment, based on performance and practicality.
4. **Reporting:** Summarize results with visualizations and key findings.