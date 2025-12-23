#          - שמירה של המודל XGOOST 1א
#         כדי שנוכל להריץ אותו בפעם הבאה ללא חיזוי מחדש

#          - מודל 1א
#================================
#       XGBOOST
#================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 1. load the dataset
# וודא שהקובץ אכן נמצא בנתיב הזה ב-Colab
file_path = "/content/Model_1ABC.xlsx"
df = pd.read_excel(file_path)

# 2. טיוב נתונים
cols_to_drop = [
    df.columns[0], 'Farm:Plot', 'Planting Year',
    'Coverage_Upper_Fruits Bunch-1', 'Coverage_Center_Fruits Bunch-1',
    'Coverage_Lower_Fruits Bunch-1', 'Coverage_Bunches',
    'Coverage_Fruits Tree-1', 'Coverage_Date',
    'T_Growth', 'T_June_Drop', 'T_Ripening', 'T_Harvest',
    'H_Growth', 'H_June_Drop', 'H_Ripening', 'H_Harvest',
    'E_Growth', 'E_June_Drop', 'E_Ripening', 'E_Harvest',
    'Thinning_Date', 'Number of trees'
]

df_model = df.drop(columns=cols_to_drop, errors='ignore')
df_model = df_model.fillna(df_model.mean())

# 3. הגדרת X ו-y
X = df_model.drop(columns=['Yield per tree'])
y = df_model['Yield per tree']

# 4. train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. אימון מודל XGBoost
xgb_model = XGBRegressor(
    n_estimators=2000,      # שקול ל-iterations
    learning_rate=0.01,
    max_depth=5,
    subsample=0.8,          # רגולריזציה (לא חובה, אבל מומלץ)
    colsample_bytree=0.8,
    objective='reg:squarederror',
    random_state=42,
    n_jobs=-1
)

xgb_model.fit(X_train, y_train)

# 6. חיזוי
y_pred_xgb = xgb_model.predict(X_test)

# 7. הערכת ביצועים
r2_xgb = r2_score(y_test, y_pred_xgb)
rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)

# 8. חשיבות פיצ'רים
feature_importances_xgb = pd.DataFrame({
    'Feature': X.columns,
    'Importance': xgb_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print(f"--- Model 1A Performance (XGBoost Regressor) ---")
print(f"R² Score: {r2_xgb:.4f}")
print(f"RMSE: {rmse_xgb:.2f} kg")
print(f"MAE: {mae_xgb:.2f} kg")

print("\n--- Top 5 Important Features (XGBoost) ---")
print(feature_importances_xgb.head(5))

# ==========================================
# 9. שמירת המודל והורדה (החלק החדש)
# ==========================================

# שמירת המודל לקובץ בסביבת הריצה של Colab
model_filename = "xgboost_yield_model_1a.json"
xgb_model.save_model(model_filename)
print(f"\nModel saved successfully as: {model_filename}")

# ניסיון להוריד את הקובץ אוטומטית למחשב המקומי (עובד רק ב-Google Colab)
try:
    from google.colab import files
    files.download(model_filename)
    print("Downloading model file to your local computer...")
except ImportError:
    print("Not running in Google Colab? File is saved locally in the script directory.")