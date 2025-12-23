import pandas as pd
import numpy as np
import math


class DataProcessor:
    def __init__(self):
        # קבועים פיזיקליים לחישוב אידוי (Penman-Monteith)
        self.RHO = 1.2  # צפיפות אוויר (kg/m³)
        self.CP = 1013  # קיבול חום סגולי (J/kg°C)
        self.LAMBDA_V = 2.45 * 10 ** 6  # חום כמוס של אידוי
        self.GAMMA = 0.065  # קבוע פסיכומטרי

    def calculate_saturation_vapor_pressure(self, T):
        """חישוב לחץ אדים רווי"""
        return 0.6108 * np.exp((17.27 * T) / (T + 237.3))

    def calculate_delta_slope(self, T):
        """חישוב שיפוע עקומת לחץ האדים"""
        es = self.calculate_saturation_vapor_pressure(T)
        return (4098 * es) / ((T + 237.3) ** 2)

    def calculate_penman_monteith(self, row):
        """
        חישוב אידוי לפי פנמן-מונטית' עבור רשומה בודדת (10 דקות)
        מבוסס על הלוגיקה מתוך Main_YS.ipynb
        """
        try:
            Rn = row['Global Radiation']  # קרינה גלובלית בוואט למ"ר
            T = row['Temperature']  # טמפרטורה בצלזיוס
            RH = row['Relative Humidity']  # לחות יחסית באחוזים

            if pd.isna(Rn) or pd.isna(T) or pd.isna(RH) or Rn < 0:
                return 0

            # המרת קרינה מ-W/m2 ל-MJ/m2 ל-10 דקות
            Rn_MJ = (Rn / 1_000_000) * 600

            es = self.calculate_saturation_vapor_pressure(T)
            ea = (RH / 100) * es
            delta = self.calculate_delta_slope(T)

            numerator = delta * Rn_MJ + self.RHO * self.CP * (es - ea) / self.LAMBDA_V
            denominator = delta + self.GAMMA

            return max(0, numerator / denominator)  # אידוי במ"מ ל-10 דקות
        except Exception:
            return 0

    def calculate_degree_hours(self, temp, threshold=18):
        """חישוב שעות חום (מעל 18 מעלות) ביחידות של 10 דקות"""
        if temp > threshold:
            # כל יחידה היא 10 דקות, כלומר שישית השעה
            return (temp - threshold) * (10 / 60)
        return 0

    def process_weather_data(self, raw_data_list, current_year):
        """
        מקבל רשימת נתונים גולמיים מה-API ומחזיר את הפיצ'רים למודל
        """
        # המרה ל-DataFrame
        df = pd.DataFrame(raw_data_list)

        # התאמת שמות עמודות בהתאם ל-API של השירות המטאורולוגי
        # שמות העמודות כאן הם דוגמה, נצטרך להתאים בדיוק למה שה-API מחזיר
        column_map = {
            'TD': 'Temperature',
            'RH': 'Relative Humidity',
            'Grad': 'Global Radiation',
            'Time': 'Time',  # פורמט hhmm
            'Date': 'Date'  # נניח שיש תאריך
        }
        # הערה: לרוב ה-API מחזיר JSON עם שדות באנגלית, נצטרך לוודא מיפוי ב-Main App
        df.rename(columns=column_map, inplace=True)

        # המרת תאריכים
        df['Datetime'] = pd.to_datetime(df['datetime'])  # נניח שה-API מחזיר שדה datetime מלא
        df['Date'] = df['Datetime'].dt.date

        # --- חישובים לכל שורה ---
        # 1. אידוי (Evaporation)
        df['E_10min'] = df.apply(self.calculate_penman_monteith, axis=1)

        # 2. שעות חום (Degree Hours)
        df['Heat_Units'] = df['Temperature'].apply(lambda x: self.calculate_degree_hours(x, 18))

        # --- הגדרת תקופות (לפי Main_YS.ipynb) ---
        # התמיינות: 1 בנובמבר (שנה קודמת) - 10 בפברואר (שנה נוכחית)
        # פריחה: 11 בפברואר - 31 במרץ
        # דילול: 1 באפריל - 15 במאי

        prev_year = current_year - 1

        periods = {
            'Inf_differentiation': (pd.Timestamp(f'{prev_year}-11-01'), pd.Timestamp(f'{current_year}-02-10')),
            'Flowering': (pd.Timestamp(f'{current_year}-02-11'), pd.Timestamp(f'{current_year}-03-31')),
            'Thinning': (pd.Timestamp(f'{current_year}-04-01'), pd.Timestamp(f'{current_year}-05-15'))
        }

        features = {}

        for period_name, (start_date, end_date) in periods.items():
            mask = (df['Datetime'] >= start_date) & (df['Datetime'] <= end_date)
            period_data = df.loc[mask]

            if period_data.empty:
                # אם אין נתונים (למשל אנחנו בתחילת העונה), נשים 0 או ממוצע
                features[f'T_{period_name}'] = 0
                features[f'H_{period_name}'] = 0
                features[f'E_{period_name}'] = 0
            else:
                # T_... = סכום שעות חום
                features[f'T_{period_name}'] = period_data['Heat_Units'].sum()
                # H_... = ממוצע לחות
                features[f'H_{period_name}'] = period_data['Relative Humidity'].mean()
                # E_... = סכום אידוי
                features[f'E_{period_name}'] = period_data['E_10min'].sum()

        return features

    def prepare_input_vector(self, user_inputs, weather_features, tree_age):
        """
        יוצר את השורה הסופית להכנסה למודל (X)
        """
        # בניית המילון הסופי בהתאם לסדר שה-XGBoost מצפה לו
        # שמות המפתחות חייבים להיות *זהים* למה שהמודל אומן עליו
        input_data = {
            # נתוני חקלאי
            'Thinning_Clusters_Tree-1': user_inputs['clusters'],  # מספר אשכולות
            'Thinning_Branches_Bunch-1': user_inputs['branches'],  # סנסנים לאשכול
            'Thinning_Fruitlets_Branch-1': user_inputs['fronds'],  # חנטים לסנסן (Fronds/Fruitlets)
            'Age': tree_age,

            # נתוני מזג אוויר מחושבים
            'T_Inf_differentiation': weather_features.get('T_Inf_differentiation', 0),
            'H_Inf_differentiation': weather_features.get('H_Inf_differentiation', 0),
            'E_Inf_differentiation': weather_features.get('E_Inf_differentiation', 0),

            'T_Flowering': weather_features.get('T_Flowering', 0),
            'H_Flowering': weather_features.get('H_Flowering', 0),
            'E_Flowering': weather_features.get('E_Flowering', 0),

            'T_Thinning': weather_features.get('T_Thinning', 0),
            'H_Thinning': weather_features.get('H_Thinning', 0),
            'E_Thinning': weather_features.get('E_Thinning', 0),
        }

        # המרה ל-DataFrame של שורה אחת
        return pd.DataFrame([input_data])