# Date Palm Yield Prediction - Model 1A
## מערכת תמיכת החלטה לחקלאים - חיזוי יבול תמרים

---

## מודל 1א - ניבוי יבול בצומת ההחלטה (אפריל-מאי)

מודל XGBoost שמנבא יבול (ק"ג לעץ) עבור החלטת המגדל בתקופת אפריל-מאי,
תוך שימוש בנתוני מזג אוויר מהעונה הקודמת ועד מועד ההחלטה.

---

## Screenshots

### Splash screen
![Splash screen](1.png)

### Main application window  
![Main application](2.png)

---

## שלבי הפעולה במערכת

### שלב 1: טעינת נתונים מטאורולוגיים
- **WeatherAPIClient** - לקוח API לשירות המטאורולוגי הישראלי (IMS)
- טוען נתונים היסטוריים לשלוש תקופות פיזיולוגיות:
  - התמיינות: 1 בנובמבר (שנה קודמת) - 10 בפברואר
  - פריחה: 11 בפברואר - 31 במרץ  
  - דילול: 1 באפריל - 15 במאי

### שלב 2: עיבוד נתונים מטאורולוגיים (DataProcessor)
- חישוב שעות חום (Degree Hours) מעל 18°C
- חישוב לחות ממוצעת (%)
- חישוב אידוי לפי Penman-Monteith (מ"מ)

### שלב 3: קלט מהמשתמש (חקלאי)
- גיל העץ (שנים)
- מספר סנסנים לאשכול
- מספר חנטים לסנסן
- מספר אשכולות

### שלב 4: הרצת מודל XGBoost
- טעינת המודל מ-`xgboost_yield_model_1a.json`
- בניית וקטור פיצ'רים (13 פיצ'רים)
- חיזוי יבול בק"ג לעץ

### שלב 5: הצגת תוצאות
- גרף פעמון (Bell Curve) עם רווחי ביטחון
- תחזית יבול מספרית

---

## פיצ'רים למודל

| קטגוריה | פיצ'ר | תיאור |
|---------|-------|--------|
| חקלאי | Thinning_Clusters_Tree-1 | מספר אשכולות |
| חקלאי | Thinning_Branches_Bunch-1 | סנסנים לאשכול |
| חקלאי | Thinning_Fruitlets_Branch-1 | חנטים לסנסן |
| חקלאי | Age | גיל העץ |
| אקלים | T_Inf_differentiation | שעות חום - התמיינות |
| אקלים | H_Inf_differentiation | לחות ממוצעת - התמיינות |
| אקלים | E_Inf_differentiation | אידוי - התמיינות |
| אקלים | T_Flowering | שעות חום - פריחה |
| אקלים | H_Flowering | לחות ממוצעת - פריחה |
| אקלים | E_Flowering | אידוי - פריחה |
| אקלים | T_Thinning | שעות חום - דילול |
| אקלים | H_Thinning | לחות ממוצעת - דילול |
| אקלים | E_Thinning | אידוי - דילול |

---

## קבצים

| קובץ | תיאור |
|------|--------|
| `main_application.py` | האפליקציה הראשית (PyQt6) |
| `data_processor.py` | עיבוד נתונים מטאורולוגיים |
| `splash_screen.py` | מסך פתיחה מונפש |
| `xgboost_yield_model_1a.json` | מודל XGBoost מאומן |
| `volcani_logo.png` | לוגו מכון וולקני |

---

## הרצה

```bash
python main_application.py
```

## דרישות

```
PyQt6
xgboost
pandas
numpy
scipy
matplotlib
requests
```

---

## Technical Implementation Details

### data_processor.py
- Data parsing and sanitization for raw weather API responses
- Physical constants and meteorological helpers (saturation vapor pressure, vapor pressure deficit)
- Penman–Monteith based evaporation estimate at 10-minute resolution
- Degree-hours computation above 18°C
- Feature aggregation across phenological periods

### splash_screen.py
- Animated QSplashScreen using PyQt6
- Neural-network visual with pulsing logo
- Rotating circular loading indicator
- ~60 FPS animation with cycling messages

### main_application.py
- Modern PyQt6 desktop UI with RTL support
- XGBoost model integration for yield prediction
- Historical weather data loading with progress indicator
- Bell curve visualization with confidence intervals
- Button freezing during processing to prevent double-clicks

---

© מכון וולקני - ARO
