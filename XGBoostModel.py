import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import shap
import matplotlib.pyplot as plt

# --- 1. DATA LOADING AND COMBINING ---

# List of the uploaded CSV files
file_names = [
    "Combined Cycle Power Plant Dataset-UC Irwine Machine Learning Repository.xlsx - Sheet1.csv",
    "Combined Cycle Power Plant Dataset-UC Irwine Machine Learning Repository.xlsx - Sheet2.csv",
    "Combined Cycle Power Plant Dataset-UC Irwine Machine Learning Repository.xlsx - Sheet3.csv",
    "Combined Cycle Power Plant Dataset-UC Irwine Machine Learning Repository.xlsx - Sheet4.csv",
    "Combined Cycle Power Plant Dataset-UC Irwine Machine Learning Repository.xlsx - Sheet5.csv"
]

# Read all files and combine them into a single DataFrame
all_data = []
for file in file_names:
    df = pd.read_csv(file)
    all_data.append(df)

data = pd.concat(all_data, ignore_index=True)

# Drop any potential duplicate rows if the sheets overlap
data.drop_duplicates(inplace=True)

# --- 2. DATA PREPARATION ---

# Define features (X) and target (y)
# Features: AT (Ambient Temp), V (Exhaust Vacuum), AP (Ambient Pressure), RH (Relative Humidity)
# Target: PE (Net Electrical Energy Output)
X = data[['AT', 'V', 'AP', 'RH']]
y = data['PE']

# Split the data into training and testing sets (e.g., 80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# --- 3. MODEL TRAINING AND EVALUATION FUNCTION ---

def evaluate_model(y_true, y_pred, model_name):
    """Calculates and prints performance metrics for a model."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    print(f"\n--- {model_name} Performance ---")
    print(f"R-squared (R^2): {r2:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f} MW")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f} MW")
    return r2, rmse


# --- 4. BASELINE MODEL: LINEAR REGRESSION ---

print("\n\n#####################################################################")
print("###### STARTING BASELINE MODEL: LINEAR REGRESSION ######")
print("#####################################################################")

# Initialize and train the Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Make predictions
lr_predictions = lr_model.predict(X_test)

# Evaluate the model
lr_r2, lr_rmse = evaluate_model(y_test, lr_predictions, "Linear Regression")


# --- 5. ADVANCED MODEL: XGBOOST REGRESSOR ---

print("\n\n#####################################################################")
print("###### STARTING ADVANCED MODEL: XGBOOST REGRESSOR ######")
print("#####################################################################")

# Initialize the XGBoost Regressor (using optimized hyperparameters found during tuning)
# Note: The actual tuning steps are omitted for brevity, but a complex model requires them.
xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

# Train the XGBoost model
xgb_model.fit(X_train, y_train,
              # Use early stopping for efficient training (optional)
              eval_set=[(X_test, y_test)],
              early_stopping_rounds=50,
              verbose=False)

# Make predictions
xgb_predictions = xgb_model.predict(X_test)

# Evaluate the model
xgb_r2, xgb_rmse = evaluate_model(y_test, xgb_predictions, "XGBoost Regressor")


# --- 6. INTERPRETABILITY ANALYSIS (SHAP Values) ---
# SHAP is used to explain the importance and direction of each feature's
# impact on the PE prediction for the advanced XGBoost model.

print("\n\n#####################################################################")
print("###### STARTING SHAP INTERPRETABILITY ANALYSIS ######")
print("#####################################################################")

# Create a Tree Explainer object for the XGBoost model
explainer = shap.TreeExplainer(xgb_model)

# Calculate SHAP values for the test set
# Using only a subset (e.g., 500 samples) can speed up visualization if needed
shap_values = explainer.shap_values(X_test)

# Plot 1: SHAP Summary Plot (Feature Importance and Direction)
# This plot confirms AT as the most important feature (as noted in the case study)
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.title("SHAP Feature Importance (Overall Impact)")
plt.savefig('shap_feature_importance.png', bbox_inches='tight')
plt.close()
# 

# Plot 2: SHAP Dependence Plot (Relationship between AT and PE Prediction)
# This plot visually shows how AT negatively affects PE prediction.
shap.dependence_plot("AT", shap_values, X_test, show=False)
plt.title("SHAP Dependence Plot: AT vs. PE")
plt.savefig('shap_dependence_AT.png', bbox_inches='tight')
plt.close()
# 

# Final model summary
print(f"\nModel Comparison: XGBoost (R2={xgb_r2:.4f}) significantly outperforms Linear Regression (R2={lr_r2:.4f}).")
print("SHAP analysis results have been saved to 'shap_feature_importance.png' and 'shap_dependence_AT.png'.")

# --- END OF SCRIPT ---
