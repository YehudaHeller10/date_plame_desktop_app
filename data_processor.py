# data_processor.py

import numpy as np
import pandas as pd
from datetime import datetime

# =========================
# קבועים פיזיקליים
# =========================
RHO = 1.2  # צפיפות אוויר (kg/m³)
CP = 1013  # קיבול חום סגולי של אוויר (J/kg°C)
LAMBDA_V = 2.45e6  # חום כמוס של התאדות (J/kg)
GAMMA = 0.065  # קבוע פסיכרומטרי (kPa/°C)


# =========================
# פונקציות חישוב אקלימיות
# =========================

def saturation_vapor_pressure(temp_celsius):
    """חישוב לחץ אדים רווי לפי טמפרטורה (kPa)"""
    if pd.isna(temp_celsius):
        return np.nan
    return 0.6108 * np.exp((17.27 * temp_celsius) / (temp_celsius + 237.3))


def actual_vapor_pressure(rh_percent, temp_celsius):
    """חישוב לחץ אדים בפועל לפי לחות יחסית וטמפרטורה (kPa)"""
    if pd.isna(rh_percent) or pd.isna(temp_celsius):
        return np.nan
    es = saturation_vapor_pressure(temp_celsius)
    return (rh_percent / 100) * es


def delta_slope(temp_celsius):
    """שיפוע עקומת לחץ האדים הרווי (kPa/°C)"""
    if pd.isna(temp_celsius):
        return np.nan
    es = saturation_vapor_pressure(temp_celsius)
    denominator = (temp_celsius + 237.3) ** 2
    return (4098 * es) / denominator if denominator != 0 else np.nan


def penman_monteith_evaporation(radiation_wm2, temp_celsius, rh_percent):
    """
    חישוב התאדות לפי נוסחת פנמן-מונטית (mm/10min)
    """
    if pd.isna(radiation_wm2) or pd.isna(temp_celsius) or pd.isna(rh_percent):
        return np.nan
    # ודא שהערכים אינם שליליים לפני החישוב
    if radiation_wm2 < 0 or rh_percent < 0:
        return np.nan

    radiation_mj_per_10min = (radiation_wm2 / 1e6) * 600  # מווט למגה-ג'ול ל-10 דקות
    G = 0  # שטף חום קרקע מוזנח

    es = saturation_vapor_pressure(temp_celsius)
    ea = actual_vapor_pressure(rh_percent, temp_celsius)
    delta = delta_slope(temp_celsius)

    # ודא שהחישובים עצמם אינם מחזירים NaN
    if pd.isna(es) or pd.isna(ea) or pd.isna(delta):
        return np.nan

    numerator = delta * (radiation_mj_per_10min - G) + RHO * CP * (es - ea) / LAMBDA_V
    denominator = delta + GAMMA
    return numerator / denominator if denominator != 0 else np.nan


# =========================
# עיבוד נתונים ראשי
# =========================

def process_weather_data_for_model(raw_data: list, year: int) -> dict:
    """
    פונקציית עיבוד נתוני אקלים לשימוש במודל חיזוי.

    שלבים עיקריים:
    1. ניקוי וסטנדרטיזציה של שמות ערוצים
    2. בניית DataFrame ראשי מהמדידות
    3. חישוב שעות חום (degree hours) והתאדות
    4. פילוח לפי תקופות פנולוגיות
    5. אגירת מאפיינים לכל תקופה (טמפ', התאדות, לחות)

    Args:
        raw_data (list): רשימת תצפיות גולמיות מה-API
        year (int): שנת היעד של היבול

    Returns:
        dict: מאפייני מודל לפי תקופות פנולוגיות
    """
    if not raw_data:
        raise ValueError("לא התקבלו נתונים לעיבוד.")

    # מיפוי שמות ערוצים לפורמט אחיד ותקני
    channel_map = {
        'TD': 'Temperature (°C)', 'Td': 'Temperature (°C)', 'TDmax': 'Temperature (°C)',
        'TDmin': 'Temperature (°C)', 'TG': 'Temperature (°C)', 'RH': 'Relative humidity (%)',
        'RH ': 'Relative humidity (%)', 'Grad': 'Radiation (W/m2)', 'Grad ': 'Radiation (W/m2)',
        'Rad': 'Radiation (W/m2)', 'DiffR': 'Radiation (W/m2)', 'NIP': 'Radiation (W/m2)',
    }

    # בניית רשומות מעובדות מהנתונים הגולמיים
    records = []
    for entry in raw_data:
        # המרת זמן לזמן מקומי והסרת מידע על אזור זמן
        row_date = pd.to_datetime(entry.get('datetime'), utc=True).tz_convert('Asia/Jerusalem').tz_localize(None)
        row = {'Date': row_date}

        # אתחול עמודות עם NaN למקרה שלא כל המדידות קיימות בכל נקודת זמן
        for standard_name in set(channel_map.values()):
            row[standard_name] = np.nan

        for ch in entry.get('channels', []):
            name = ch.get('name', '').strip()

            # הערה לעצמי כדי לא לחזור על הטעות הארורה שוב!!! : ה-API מציין ש-status=1 הוא ערוץ תקין, ו-status=2 לא תקין.
            # יש לקבל רק מדידות עם סטטוס 1.
            if name in channel_map and ch.get('status') == 1 and ch.get('valid', True):
                standard_name = channel_map[name]
                row[standard_name] = pd.to_numeric(ch.get('value'), errors='coerce')

        records.append(row)

    if not records:
        raise ValueError("לא נמצאו רשומות תקינות לאחר סינון ראשוני.")

    # יצירת DataFrame, מיון לפי תאריך ומילוי חורים קטנים
    df = pd.DataFrame(records).set_index('Date').sort_index()
    # אסטרטגיית איחוד - אם יש כפילויות תאריכים, נשתמש בממוצע
    df = df.groupby(df.index).mean()

    # חישוב שעות חום (מעל 18 מעלות), מומר ליחידות של שעה
    df['degree_hours_10_min'] = np.where(df['Temperature (°C)'] > 18,
                                         (df['Temperature (°C)'] - 18) * (10 / 60),
                                         0)

    # ניקוי ערכים שליליים שעשויים להיות טעויות מדידה
    for col in ['Radiation (W/m2)', 'Relative humidity (%)']:
        if col in df.columns:
            df[col] = df[col].where(df[col] >= 0, np.nan)

    # חישוב התאדות לפי פנמן-מונטית עבור כל שורה
    df['Evaporation (mm/10min)'] = df.apply(
        lambda row: penman_monteith_evaporation(
            row.get('Radiation (W/m2)'),
            row.get('Temperature (°C)'),
            row.get('Relative humidity (%)')
        ), axis=1
    )

    # הגדרת תקופות עונתיות לפי לוח שנה פנולוגי
    periods = {
        "Inf_differentiation": (f"{year - 1}-11-01", f"{year}-02-10"),
        "Flowering": (f"{year}-02-11", f"{year}-03-31"),
        "Thinning": (f"{year}-04-01", f"{year}-05-15"),
        "Growth": (f"{year}-05-16", f"{year}-07-31"),
        "June_Drop": (f"{year}-06-01", f"{year}-06-30"),
        "Ripening": (f"{year}-08-01", f"{year}-08-31"),
        "Harvest": (f"{year}-09-01", f"{year}-10-31"),
    }

    # יצירת מילון המאפיינים הסופי עבור המודל
    features = {}
    for name, (start, end) in periods.items():
        sub_df = df.loc[start:end]
        if not sub_df.empty:
            features[f"T_{name}"] = round(sub_df['degree_hours_10_min'].sum(), 2)
            features[f"E_{name}"] = round(sub_df['Evaporation (mm/10min)'].sum(), 2)
            features[f"H_{name}"] = round(sub_df['Relative humidity (%)'].mean(), 2)
        else:
            # אם אין נתונים בתקופה, נכניס ערכי None כדי למנוע שגיאות בהמשך
            features[f"T_{name}"] = None
            features[f"E_{name}"] = None
            features[f"H_{name}"] = None

    # בדיקה סופית לוודא שלא כל הערכים הם None
    if all(v is None for v in features.values()):
        raise ValueError("העיבוד הסתיים אך לא נוצרו מאפיינים. ייתכן שהנתונים חסרים בטווחי התאריכים הנדרשים.")

    return features