import pandas as pd
import numpy as np


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
        שלב 2: עיבוד נתונים מטאורולוגיים גולמיים
        מקבל רשימת נתונים גולמיים מה-API ומחזיר את הפיצ'רים למודל

        מבנה הנתונים מה-API של השירות המטאורולוגי:
        - datetime: תאריך ושעה
        - TD: טמפרטורה יבשה (°C)
        - RH: לחות יחסית (%)
        - Grad: קרינה גלובלית (W/m²)
        """
        # שלב 2.1: המרה ל-DataFrame
        # בדיקה אם הנתונים במבנה מקונן (IMS API מחזיר נתונים בתוך 'channels')
        if raw_data_list and isinstance(raw_data_list[0], dict):
            # בדיקה אם יש שדה 'channels' - מבנה מקונן
            if 'channels' in raw_data_list[0]:
                # המרה ממבנה מקונן למבנה שטוח
                flattened_data = []
                for record in raw_data_list:
                    flat_record = {'datetime': record.get('datetime')}
                    for channel in record.get('channels', []):
                        channel_name = channel.get('name', '')
                        channel_value = channel.get('value')
                        if channel_name and channel_value is not None:
                            flat_record[channel_name] = channel_value
                    flattened_data.append(flat_record)
                df = pd.DataFrame(flattened_data)
                print(f"DEBUG: Flattened {len(flattened_data)} records from nested format")
            else:
                df = pd.DataFrame(raw_data_list)
        else:
            df = pd.DataFrame(raw_data_list)

        print(f"DEBUG: DataFrame columns: {list(df.columns)}")
        print(f"DEBUG: DataFrame shape: {df.shape}")
        if len(df) > 0:
            print(f"DEBUG: First row sample: {df.iloc[0].to_dict()}")

        # שלב 2.2: התאמת שמות עמודות בהתאם ל-API של השירות המטאורולוגי
        # ה-API מחזיר נתונים בפורמט שונה - צריך להתאים
        column_map = {
            'TD': 'Temperature',
            'TDmax': 'Temperature',  # גיבוי אם אין TD
            'RH': 'Relative Humidity',
            'Grad': 'Global Radiation',
        }

        # ניסיון למפות עמודות קיימות
        for old_name, new_name in column_map.items():
            if old_name in df.columns and new_name not in df.columns:
                df.rename(columns={old_name: new_name}, inplace=True)

        # שלב 2.3: טיפול בשדה datetime - ה-API של IMS מחזיר שדה 'datetime'
        # שימוש ב-utc=True כדי לטפל ב-timezones מעורבים
        datetime_col = None
        if 'datetime' in df.columns:
            datetime_col = 'datetime'
        elif 'date' in df.columns:
            datetime_col = 'date'
        else:
            # ניסיון למצוא כל עמודה שמכילה תאריך
            for col in df.columns:
                if 'date' in col.lower() or 'time' in col.lower():
                    datetime_col = col
                    break

        if datetime_col is not None:
            # המרה לתאריכים עם UTC
            df['Datetime'] = pd.to_datetime(df[datetime_col], utc=True, errors='coerce')

            # הסרה של timezone info כדי לעבוד עם timestamps פשוטים
            if df['Datetime'].dt.tz is not None:
                df['Datetime'] = df['Datetime'].dt.tz_convert(None)
        else:
            print("Warning: No datetime column found in data")
            return self._get_empty_features()

        # הסרת שורות ללא תאריך תקין
        df = df.dropna(subset=['Datetime'])

        if df.empty:
            print("Warning: No valid datetime entries found in weather data")
            return self._get_empty_features()

        df['Date'] = df['Datetime'].dt.date

        # שלב 2.4: וידוא קיום עמודות נדרשות - יצירת ברירות מחדל אם חסרות
        if 'Temperature' not in df.columns:
            print("Warning: Temperature column not found, using default")
            df['Temperature'] = 25.0  # ברירת מחדל

        if 'Relative Humidity' not in df.columns:
            print("Warning: Relative Humidity column not found, using default")
            df['Relative Humidity'] = 50.0  # ברירת מחדל

        if 'Global Radiation' not in df.columns:
            print("Warning: Global Radiation column not found, using default")
            df['Global Radiation'] = 500.0  # ברירת מחדל

        # שלב 2.5: חישובים לכל שורה
        # 1. אידוי (Evaporation)
        df['E_10min'] = df.apply(self.calculate_penman_monteith, axis=1)

        # 2. שעות חום (Degree Hours)
        df['Heat_Units'] = df['Temperature'].apply(lambda x: self.calculate_degree_hours(x, 18) if pd.notna(x) else 0)

        # שלב 2.6: הגדרת תקו��ות פיזיולוגיות (לפי Main_YS.ipynb)
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

        # שלב 2.7: חישוב פיצ'רים לכל תקופה
        for period_name, (start_date, end_date) in periods.items():
            mask = (df['Datetime'] >= start_date) & (df['Datetime'] <= end_date)
            period_data = df.loc[mask]

            if period_data.empty:
                # אם אין נתונים (למשל אנחנו בתחילת העונה), נשים 0 או ממוצע
                features[f'T_{period_name}'] = 0
                features[f'H_{period_name}'] = 50.0  # ממוצע סביר
                features[f'E_{period_name}'] = 0
            else:
                # T_... = סכום שעות חום
                features[f'T_{period_name}'] = period_data['Heat_Units'].sum()
                # H_... = ממוצע לחות
                features[f'H_{period_name}'] = period_data['Relative Humidity'].mean()
                # E_... = סכום אידוי
                features[f'E_{period_name}'] = period_data['E_10min'].sum()

        return features

    def _get_empty_features(self):
        """שלב 2.8: החזרת פיצ'רים ריקים במקרה של שגיאה"""
        periods = ['Inf_differentiation', 'Flowering', 'Thinning']
        features = {}
        for period in periods:
            features[f'T_{period}'] = 0
            features[f'H_{period}'] = 50.0
            features[f'E_{period}'] = 0
        return features

    def prepare_input_vector(self, user_inputs, weather_features, tree_age, current_year=None):
        """
        יוצר את השורה הסופית להכנסה למודל (X)

        שמות הפיצ'רים חייבים להתאים בדיוק למה שהמודל אומן עליו:
        - Tree age (לא Age)
        - year
        - Thinning_Upper_Fruits Bunch-1
        - Thinning_Center_Fruits Bunch-1
        - Thinning_Lower_Fruits Bunch-1
        - Thinning_Bunches
        - Thinning_Fruits Tree-1
        - T/H/E לתקופות פיזיולוגיות
        """
        from datetime import datetime

        if current_year is None:
            current_year = datetime.now().year

        # חישוב חנטים לאשכול לפי הדורות
        # אם יש פרוטוקול כללי - משתמשים באותם ערכים לכל הדורות
        upper_fruits = user_inputs.get('upper_fronds', user_inputs.get('fronds', 120))
        center_fruits = user_inputs.get('middle_fronds', user_inputs.get('fronds', 120))
        lower_fruits = user_inputs.get('lower_fronds', user_inputs.get('fronds', 120))

        # מספר אשכולות
        bunches = user_inputs.get('clusters', 8)

        # חישוב סה"כ חנטים לעץ
        # Thinning_Fruits Tree-1 = סה"כ חנטים = אשכולות * סנסנים * חנטים לסנסן
        branches = user_inputs.get('branches', 25)
        avg_fronds = (upper_fruits + center_fruits + lower_fruits) / 3
        fruits_per_tree = bunches * branches * avg_fronds

        # בניית המילון הסופי בהתאם לסדר שה-XGBoost מצפה לו
        input_data = {
            # נתוני חקלאי - בשמות שהמודל מצפה להם
            'Tree age': tree_age,
            'year': current_year,
            'Thinning_Upper_Fruits Bunch-1': upper_fruits,
            'Thinning_Center_Fruits Bunch-1': center_fruits,
            'Thinning_Lower_Fruits Bunch-1': lower_fruits,
            'Thinning_Bunches': bunches,
            'Thinning_Fruits Tree-1': fruits_per_tree,

            # נתוני מזג אוויר מחושבים - בסדר שהמודל מצפה
            'T_Inf_differentiation': weather_features.get('T_Inf_differentiation', 0),
            'T_Flowering': weather_features.get('T_Flowering', 0),
            'T_Thinning': weather_features.get('T_Thinning', 0),

            'H_Inf_differentiation': weather_features.get('H_Inf_differentiation', 50),
            'H_Flowering': weather_features.get('H_Flowering', 50),
            'H_Thinning': weather_features.get('H_Thinning', 50),

            'E_Inf_differentiation': weather_features.get('E_Inf_differentiation', 0),
            'E_Flowering': weather_features.get('E_Flowering', 0),
            'E_Thinning': weather_features.get('E_Thinning', 0),
        }

        # המרה ל-DataFrame של שורה אחת
        return pd.DataFrame([input_data])