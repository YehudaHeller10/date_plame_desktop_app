# main_application.py
import sys
import requests
import numpy as np
import xgboost as xgb
from datetime import datetime
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from scipy.stats import norm

# =====================================================================================
#  ייבוא מסך הפתיחה ומעבד הנתונים
# =====================================================================================
from splash_screen import AnimatedSplashScreen
from data_processor import DataProcessor

# =====================================================================================
# 1. הגדרות עיצוב וסגנון גלובליות
# =====================================================================================
# פלטת צבעים מודרנית ובהירה
COLORS = {
    'primary': '#3B82F6',  # Blue 500
    'primary_dark': '#2563EB',  # Blue 600
    'primary_light': '#60A5FA',  # Blue 400
    'secondary': '#10B981',  # Emerald 500
    'accent': '#F59E0B',  # Amber 500
    'surface': '#FFFFFF',
    'background': '#F8FAFC',  # Slate 50
    'background_darker': '#F1F5F9',  # Slate 100
    'card': '#FFFFFF',
    'text': '#1F2937',  # Slate 800
    'text_secondary': '#6B7280',  # Slate 500
    'border': '#E5E7EB',  # Slate 200
    'sidebar_bg': '#1F2937',  # Slate 800
    'sidebar_bg_darker': '#111827',  # Slate 900
    'sidebar_selected': '#374151',  # Slate 700
    'sidebar_hover': '#4B5563',  # Slate 600
    'sidebar_text': '#E5E7EB'  # Slate 200
}

# גיליון סגנונות מרכזי (QSS)
STYLES = f"""
    QMainWindow, QDialog {{
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 {COLORS['background']}, stop:1 {COLORS['background_darker']});
    }}

    /* --- סרגל צד --- */
    QListWidget {{
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {COLORS['sidebar_bg_darker']}, stop:1 {COLORS['sidebar_bg']});
        border: none;
        padding-top: 20px;
        font-size: 15px;
        font-weight: 600;
    }}
    QListWidget::item {{
        color: {COLORS['sidebar_text']};
        padding: 18px 24px;
        border-radius: 8px;
        margin: 4px 12px;
    }}
    QListWidget::item:hover {{
        background-color: {COLORS['sidebar_hover']};
    }}
    QListWidget::item:selected {{
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {COLORS['primary_light']}, stop:1 {COLORS['primary']});
        color: white;
    }}
    /* --- הגדרות עבור QSplitter --- */
    QSplitter::handle {{
        background-color: {COLORS['border']};
    }}
    QSplitter::handle:horizontal {{
        width: 1px;
    }}
    QSplitter::handle:vertical {{
        height: 1px;
    }}

    /* --- לשוניות (Tabs) --- */
    QTabWidget::pane {{
        border: 1px solid {COLORS['border']};
        border-top: none;
        border-radius: 0 0 8px 8px;
        background: {COLORS['background']};
        padding: 15px;
    }}
    QTabBar::tab {{
        background: {COLORS['background_darker']};
        color: {COLORS['text_secondary']};
        border: 1px solid {COLORS['border']};
        border-bottom: none;
        padding: 10px 20px;
        font-weight: 600;
        border-top-left-radius: 8px;
        border-top-right-radius: 8px;
    }}
    QTabBar::tab:selected {{
        background: {COLORS['background']};
        color: {COLORS['primary']};
        border-bottom: 1px solid {COLORS['background']};
    }}
    QTabBar::tab:hover {{
        background: {COLORS['border']};
        color: {COLORS['text']};
    }}

    /* --- כפתורים --- */
    QPushButton {{
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 {COLORS['primary']}, stop:1 {COLORS['primary_dark']});
        color: white;
        border: none;
        padding: 12px 24px;
        font-size: 15px;
        font-weight: 600;
        border-radius: 8px;
        min-height: 38px;
    }}
    QPushButton:hover {{
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 {COLORS['primary_light']}, stop:1 {COLORS['primary']});
    }}
    QPushButton:pressed {{
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 {COLORS['primary_dark']}, stop:1 {COLORS['primary_dark']});
    }}
    QPushButton:disabled {{
        background: #9CA3AF;
        color: #E5E7EB;
    }}

    /* --- עיצוב משופר לכפתור הניתוח --- */
    QPushButton#AnalyzeButton {{
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {COLORS['secondary']}, stop:1 #059669);
        font-size: 16px;
        padding: 14px 30px;
        margin: 20px 0 10px 0; /* מרווח עליון גדול יותר להפרדה מהטאבים */
        min-width: 200px;
    }}
    QPushButton#AnalyzeButton:hover {{
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #34D399, stop:1 {COLORS['secondary']});
    }}
    QPushButton#AnalyzeButton:disabled {{
        background: #9CA3AF;
        color: #E5E7EB;
    }}

    /* --- כרטיסים וקונטיינרים --- */
    QFrame#Card, QGroupBox {{
        background: {COLORS['card']};
        border-radius: 16px;
        border: 1px solid {COLORS['border']};
    }}
    QGroupBox {{
        padding: 20px 10px 10px 10px;
        margin-top: 10px;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        subcontrol-position: top center;
        padding: 0 10px;
        color: {COLORS['text']};
        font-weight: 700;
    }}

    /* --- תוויות --- */
    QLabel#PageTitle {{
        color: {COLORS['text']};
        font-size: 28px;
        font-weight: 800;
        padding-bottom: 5px;
    }}
    QLabel#PageSubtitle {{
        color: {COLORS['text_secondary']};
        font-size: 16px;
        padding-bottom: 20px;
    }}
    QLabel#CardTitle {{
        color: {COLORS['text']};
        font-size: 18px;
        font-weight: 700;
        padding-bottom: 10px;
        border-bottom: 1px solid {COLORS['border']};
        margin-bottom: 10px;
    }}

    /* --- שדות קלט --- */
    QLineEdit, QComboBox {{
        border: 1px solid {COLORS['border']};
        border-radius: 8px;
        padding: 10px 14px;
        font-size: 14px;
        background-color: white;
        min-height: 24px;
        min-width: 150px;
    }}
    QLineEdit:focus, QComboBox:focus {{
        border: 2px solid {COLORS['primary']};
    }}
    QComboBox::drop-down {{
        border: none;
    }}

    /* --- רכיבים נוספים --- */
    QTextEdit {{
        border: 1px solid {COLORS['border']};
        border-radius: 8px;
        background-color: white;
        font-size: 13px;
        padding: 8px;
        min-height: 80px;
    }}

    /* --- עיצוב עבור אזור גלילה --- */
    QScrollArea {{
        border: none;
        background-color: transparent;
    }}
    QScrollArea QWidget {{
        background-color: transparent;
    }}
    QScrollArea QScrollBar:vertical {{
        background-color: {COLORS['background_darker']};
        width: 8px;
        border-radius: 4px;
        margin: 0px;
    }}
    QScrollArea QScrollBar::handle:vertical {{
        background-color: {COLORS['primary_light']};
        border-radius: 4px;
        min-height: 20px;
    }}
    QScrollArea QScrollBar::handle:vertical:hover {{
        background-color: {COLORS['primary']};
    }}
    QScrollArea QScrollBar::add-line:vertical, QScrollArea QScrollBar::sub-line:vertical {{
        height: 0px;
        border: none;
        background: none;
    }}
    QScrollArea QScrollBar::up-arrow:vertical, QScrollArea QScrollBar::down-arrow:vertical {{
        background: none;
    }}
"""


# =====================================================================================
# 2. מחלקות עזר ורכיבים
# =====================================================================================
def apply_shadow(widget, blur_radius=25, x_offset=0, y_offset=4, color=QColor(100, 100, 100, 30)):
    """פונקציית עזר להחלת אפקט צל על ווידג'טים"""
    shadow = QGraphicsDropShadowEffect()
    shadow.setBlurRadius(blur_radius)
    shadow.setXOffset(x_offset)
    shadow.setYOffset(y_offset)
    shadow.setColor(color)
    widget.setGraphicsEffect(shadow)


class WeatherAPIClient:
    """
    שלב 1: לקוח API לשירות המטאורולוגי הישראלי (IMS)
    מאפשר טעינת רשימת תחנות ונתונים היסטוריים לפי טווח תאריכים
    """
    def __init__(self, api_token: str):
        self.api_token = api_token
        self.base_url = "https://api.ims.gov.il/v1/envista"
        self.headers = {"Authorization": f"ApiToken {api_token}"}

    def get_stations(self):
        """שלב 1.1: קבלת רשימת כל התחנות המטאורולוגיות"""
        try:
            response = requests.get(f"{self.base_url}/stations", headers=self.headers, timeout=15)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"שגיאה בקריאת נתוני תחנות: {e}")

    def get_station_data(self, station_id: int):
        """שלב 1.2: קבלת נתונים אחרונים מתחנה (לתצוגה מהירה)"""
        try:
            url = f"{self.base_url}/stations/{station_id}/data/latest"
            response = requests.get(url, headers=self.headers, timeout=15)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"שגיאה בקריאת נתונים מטאורולוגיים: {e}")

    def get_historical_data(self, station_id: int, start_date: str, end_date: str):
        """
        שלב 1.3: קבלת נתונים היסטוריים מתחנה לפי טווח תאריכים
        פורמט תאריכים: YYYY/MM/DD
        """
        try:
            url = f"{self.base_url}/stations/{station_id}/data"
            params = {"from": start_date, "to": end_date}
            response = requests.get(url, headers=self.headers, params=params, timeout=60)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"שגיאה בקריאת נתונים היסטוריים: {e}")


class APIWorker(QThread):
    """שלב 2.1: Worker לטעינת רשימת תחנות או נתונים אחרונים (לא חוסם UI)"""
    data_ready = pyqtSignal(object)
    error_occurred = pyqtSignal(str)

    def __init__(self, api_client, station_id=None):
        super().__init__()
        self.api_client = api_client
        self.station_id = station_id
        self.operation = 'stations' if station_id is None else 'data'

    def run(self):
        try:
            data = self.api_client.get_stations() if self.operation == 'stations' else self.api_client.get_station_data(
                self.station_id)
            self.data_ready.emit(data)
        except Exception as e:
            self.error_occurred.emit(str(e))


class HistoricalWeatherWorker(QThread):
    """
    שלב 2.2: Worker לטעינת נתונים מטאורו��וגיים היסטוריים (ברקע)
    טוען נתונים עבור כל התקופות הפיזיולוגיות הנדרשות למודל 1א:
    - התמיינות: 1 בנובמבר (שנה קודמת) - 10 בפברואר
    - פריחה: 11 בפברואר - 31 במרץ
    - דילול: 1 באפריל - 15 במאי
    """
    data_ready = pyqtSignal(object)  # dict עם weather_features
    error_occurred = pyqtSignal(str)
    progress_update = pyqtSignal(str)  # עדכון סטטוס למשתמש

    def __init__(self, api_client, station_id: int, current_year: int):
        super().__init__()
        self.api_client = api_client
        self.station_id = station_id
        self.current_year = current_year
        self.data_processor = DataProcessor()

    def run(self):
        """
        שלב 2.3: תהליך טעינה ועיבוד נתונים מטאורולוגיים
        """
        try:
            prev_year = self.current_year - 1

            # שלב 2.3.1: הגדרת טווח התאריכים הכולל
            start_date = f"{prev_year}/11/01"
            end_date = f"{self.current_year}/05/15"

            self.progress_update.emit(
                f"Loading meteorological data ({start_date.replace('/', '-')} to {end_date.replace('/', '-')})..."
            )

            # שלב 2.3.2: קריאה ל-API לקבלת כל הנתונים ההיסטוריים
            raw_response = self.api_client.get_historical_data(
                self.station_id, start_date, end_date
            )

            # שלב 2.3.3: חילוץ הנתונים מתוך התשובה
            if isinstance(raw_response, dict) and 'data' in raw_response:
                raw_data_list = raw_response['data']
            elif isinstance(raw_response, list):
                raw_data_list = raw_response
            else:
                raise Exception("פורמט נתונים לא צפוי מה-API")

            if not raw_data_list:
                raise Exception("לא התקבלו נתונים מהתחנה לתקופה המבוקשת")

            self.progress_update.emit("Processing weather features...")

            # שלב 2.3.4: עיבוד הנתונים וחישוב הפיצ'רים למודל
            weather_features = self.data_processor.process_weather_data(
                raw_data_list, self.current_year
            )

            self.data_ready.emit(weather_features)

        except Exception as e:
            self.error_occurred.emit(str(e))


class YieldDistributionChart(FigureCanvas):
    """
    שלב 3: גרף התפלגות התחזית (Bell Curve)
    מציג את החיזוי של המודל עם רווחי ביטחון
    טקסט באנגלית למניעת בעיות עם RTL
    """
    def __init__(self):
        self.fig = Figure(figsize=(8, 4.5), facecolor=COLORS['card'])
        super().__init__(self.fig)
        self.setMinimumHeight(350)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor(COLORS['background'])
        self.fig.patch.set_alpha(0)

    def plot(self, mean, std_dev):
        """
        שלב 3.1: ציור גרף הפעמון עם תחזית היבול
        mean: ממוצע התחזית (ק"ג לעץ)
        std_dev: סטיית תקן של התחזית
        """
        self.ax.clear()

        # שלב 3.2: יצירת עקומת ההתפלגות הנורמלית
        x = np.linspace(mean - 4 * std_dev, mean + 4 * std_dev, 400)
        y = norm.pdf(x, mean, std_dev)

        # שלב 3.3: ציור העקומה ורווחי הביטחון
        self.ax.plot(x, y, color=COLORS['primary'], linewidth=2.5, label='Yield Distribution')
        self.ax.fill_between(x, y, where=(x >= mean - std_dev) & (x <= mean + std_dev),
                             color=COLORS['primary_light'], alpha=0.6, label='68% Confidence')
        self.ax.fill_between(x, y, where=(x >= mean - 2 * std_dev) & (x <= mean + 2 * std_dev),
                             color=COLORS['primary_light'], alpha=0.3, label='95% Confidence')

        # שלב 3.4: קו אנכי לציון הממוצע
        self.ax.axvline(mean, color=COLORS['accent'], linestyle='--', linewidth=2,
                        label=f'Predicted Yield: {mean:.1f} kg/tree')

        # שלב 3.5: הגדרות תצוגה (באנגלית)
        self.ax.set_xlabel('Yield Prediction (kg/tree)', fontsize=12, color=COLORS['text_secondary'])
        self.ax.set_ylabel('Probability Density', fontsize=12, color=COLORS['text_secondary'])
        self.ax.set_title(f'XGBoost Model 1A - Yield Prediction', fontsize=14,
                         color=COLORS['text'], fontweight='bold', pad=10)

        self.ax.tick_params(colors=COLORS['text_secondary'])
        for spine in self.ax.spines.values():
            spine.set_visible(False)
        self.ax.grid(True, axis='y', alpha=0.3, linestyle='--')
        self.ax.get_yaxis().set_ticks([])
        self.ax.legend(loc='upper right', frameon=False, fontsize=10)
        self.fig.tight_layout(pad=2.0)
        self.draw()


# =====================================================================================
# 3. דפי המערכת (Widgets)
# =====================================================================================
class HomePage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(40, 30, 40, 30)
        layout.setSpacing(15)
        logo_label = QLabel()
        logo_pixmap = QPixmap('volcani_logo.png')
        if not logo_pixmap.isNull():
            logo_label.setPixmap(logo_pixmap.scaled(120, 120, Qt.AspectRatioMode.KeepAspectRatio,
                                                    Qt.TransformationMode.SmoothTransformation))
            logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(logo_label)
            layout.addSpacing(5)
        title = QLabel("🏠 ברוכים הבאים למערכת תומכת החלטה")
        title.setObjectName("PageTitle")
        apply_shadow(title, blur_radius=5, x_offset=1, y_offset=2, color=QColor(0, 0, 0, 20))
        subtitle = QLabel("כלי בינה מלאכותית מתקדם לקבלת החלטות מושכלות בגיזום ודילול תמרים.")
        subtitle.setObjectName("PageSubtitle")
        layout.addWidget(title)
        layout.addWidget(subtitle)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content_widget = QWidget()
        scroll.setWidget(content_widget)
        content_layout = QVBoxLayout(content_widget)
        steps_card = self.create_step_card()
        content_layout.addWidget(steps_card)
        layout.addWidget(scroll)

    def create_step_card(self):
        card = QFrame()
        card.setObjectName("Card")
        apply_shadow(card)
        card_layout = QVBoxLayout(card)
        card_layout.setSpacing(20)
        card_title = QLabel("איך משתמשים במערכת?")
        card_title.setObjectName("CardTitle")
        card_layout.addWidget(card_title)
        steps_text = [
            ("<b>שלב 1: הזנת נתונים</b>",
             "עברו לדף 'הזנת נתונים'. שם תוכלו לבחור את שיטת הזנת הגיל, למלא פרוטוקול דילול (כללי או לפי דור) ולבחור תחנה מטאורולוגית."),
            ("<b>שלב 2: ניתוח חכם</b>",
             "לאחר מילוי כל השדות, לחצו על כפתור 'נתח וצור תחזית'. המערכת תעבד את הנתונים שהזנתם יחד עם מודלים חקלאיים מתקדמים."),
            ("<b>שלב 3: קבלת תוצאות</b>",
             "המערכת תציג לכם אוטומטית את דף התוצאות, הכולל גרף התפלגות של תחזית היבול והמלצות מותאמות אישית להמשך טיפול.")
        ]
        for title_text, desc in steps_text:
            step_label = QLabel(f"📝 {title_text}")
            step_label.setStyleSheet("font-size: 16px; color: #1E40AF; font-weight: 600;")
            desc_label = QLabel(desc)
            desc_label.setWordWrap(True)
            desc_label.setStyleSheet("font-size: 14px; color: #374151; line-height: 1.5;")
            card_layout.addWidget(step_label)
            card_layout.addWidget(desc_label)
            card_layout.addSpacing(10)
        return card


class DataEntryPage(QWidget):
    """
    שלב 4: דף הזנת נתונים
    כולל: בחירת תחנה, פרמטרים מהמשתמש, וטעינת נתונים מטאורולוגיים
    """
    analysis_requested = pyqtSignal(dict)

    def __init__(self, api_client, parent=None):
        super().__init__(parent)
        self.api_client = api_client
        self.stations_data = []
        self.weather_features = None  # שלב 4.1: שמירת הפיצ'רים המטאורולוגיים
        self.is_loading = False  # שלב 4.2: מצב טעינה

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(40, 30, 40, 30)
        main_layout.setSpacing(20)

        title = QLabel("📝 הזנת נתונים ופרוטוקול דילול")
        title.setObjectName("PageTitle")
        apply_shadow(title, blur_radius=5, x_offset=1, y_offset=2, color=QColor(0, 0, 0, 20))
        subtitle = QLabel("מלאו את הנתונים הבאים כדי לקבל תחזית יבול (מודל 1א - צומת החלטה אפריל-מאי).")
        subtitle.setObjectName("PageSubtitle")
        main_layout.addWidget(title)
        main_layout.addWidget(subtitle)

        content_layout = QHBoxLayout()
        content_layout.setSpacing(25)

        # --- כרטיס נתוני אקלים (צד שמאל) ---
        weather_card = QFrame()
        weather_card.setObjectName("Card")
        apply_shadow(weather_card)
        weather_layout = QVBoxLayout(weather_card)
        weather_title = QLabel("נתוני אקלים - תחנה מטאורולוגית")
        weather_title.setObjectName("CardTitle")
        weather_layout.addWidget(weather_title)

        self.station_combo = QComboBox()
        self.station_combo.setPlaceholderText("טוען תחנות...")
        self.station_combo.setEnabled(False)
        weather_layout.addWidget(self.station_combo)

        self.load_data_btn = QPushButton("📡 טען נתוני תחנה")
        self.load_data_btn.setEnabled(False)
        self.load_data_btn.clicked.connect(self.load_weather_data)
        weather_layout.addWidget(self.load_data_btn)

        # שלב 4.3: אזור סטטוס טעינה עם Spinner
        self.status_container = QWidget()
        status_layout = QHBoxLayout(self.status_container)
        status_layout.setContentsMargins(0, 10, 0, 10)

        # Spinner (אנימציית טעינה)
        self.spinner_label = QLabel("⏳")
        self.spinner_label.setStyleSheet("font-size: 20px;")
        self.spinner_label.setVisible(False)

        # טקסט סטטוס
        self.status_label = QLabel("")
        self.status_label.setStyleSheet(f"color: {COLORS['primary']}; font-size: 13px; font-weight: 600;")
        self.status_label.setWordWrap(True)

        status_layout.addWidget(self.spinner_label)
        status_layout.addWidget(self.status_label, 1)
        weather_layout.addWidget(self.status_container)

        # טיימר לאנימציית הספינר
        self.spinner_timer = QTimer(self)
        self.spinner_timer.timeout.connect(self._animate_spinner)
        self.spinner_frames = ["⏳", "⌛", "🔄", "⏳"]
        self.spinner_index = 0

        self.data_display = QTextEdit()
        self.data_display.setPlainText("בחר תחנה ולחץ על 'טען נתוני תחנה' כדי לטעון נתונים מטאורולוגיים היסטוריים...")
        self.data_display.setReadOnly(True)
        weather_layout.addWidget(self.data_display)

        # --- כרטיס פרוטוקול (צד ימין) ---
        protocol_card = QFrame()
        protocol_card.setObjectName("Card")
        apply_shadow(protocol_card)
        protocol_layout = QVBoxLayout(protocol_card)
        protocol_layout.setSpacing(15)
        protocol_title = QLabel("פרטי העץ ופרוטוקול הדילול")
        protocol_title.setObjectName("CardTitle")
        protocol_layout.addWidget(protocol_title)
        self.create_age_input_group(protocol_layout)

        tab_widget = self.create_thinning_tabs()
        protocol_layout.addWidget(tab_widget)

        # --- מיקום חדש ומשופר לכפתור הניתוח ---
        self.analyze_btn = QPushButton("⚠️ טען נתונים תחילה")
        self.analyze_btn.setObjectName("AnalyzeButton")
        self.analyze_btn.setEnabled(False)  # מושבת עד שנטענים נתונים מטאורולוגיים
        self.analyze_btn.clicked.connect(self.request_analysis)

        button_inside_card_layout = QHBoxLayout()
        button_inside_card_layout.addStretch()
        button_inside_card_layout.addWidget(self.analyze_btn)
        button_inside_card_layout.addStretch()

        protocol_layout.addLayout(button_inside_card_layout)
        protocol_layout.addStretch()

        # הוספת הכרטיסים לפריסה הראשית
        content_layout.addWidget(weather_card, 1)
        content_layout.addWidget(protocol_card, 2)
        main_layout.addLayout(content_layout)

        self.load_stations()

    def _animate_spinner(self):
        """שלב 4.4: אנימציית הספינר"""
        self.spinner_index = (self.spinner_index + 1) % len(self.spinner_frames)
        self.spinner_label.setText(self.spinner_frames[self.spinner_index])

    def _set_loading_state(self, is_loading: bool, status_text: str = ""):
        """שלב 4.5: הקפאת/שחרור הממשק בזמן טעינה"""
        self.is_loading = is_loading

        # עדכון מצב הכפתורים
        self.load_data_btn.setEnabled(not is_loading and len(self.stations_data) > 0)
        self.station_combo.setEnabled(not is_loading and len(self.stations_data) > 0)

        # עדכון כפתור הניתוח - מושבת עד שנטענים נתונים מטאורולוגיים
        can_analyze = not is_loading and self.weather_features is not None
        self.analyze_btn.setEnabled(can_analyze)

        # שינוי טקסט הכפתור בהתאם למצב
        if is_loading:
            self.analyze_btn.setText("⏳ בתהליך - המתן לסיום...")
            self.load_data_btn.setText("⏳ טוען...")
        else:
            self.analyze_btn.setText("🔬 נתח וצור תחזית")
            self.load_data_btn.setText("📡 טען נתוני תחנה")

            # אם אין נתונים מטאורולוגיים - הודעה מתאימה
            if self.weather_features is None:
                self.analyze_btn.setText("⚠️ טען נתונים תחילה")

        # עדכון הספינר
        self.spinner_label.setVisible(is_loading)
        if is_loading:
            self.spinner_timer.start(300)  # אנימציה כל 300ms
        else:
            self.spinner_timer.stop()

        # עדכון טקסט הסטטוס
        self.status_label.setText(status_text)

    def create_age_input_group(self, parent_layout):
        age_group = QGroupBox("גיל העץ")
        age_group_layout = QVBoxLayout(age_group)
        self.by_age_radio = QRadioButton("הזן גיל")
        self.by_year_radio = QRadioButton("הזן שנת שתילה")
        radio_layout = QHBoxLayout()
        radio_layout.addWidget(self.by_age_radio)
        radio_layout.addWidget(self.by_year_radio)
        age_group_layout.addLayout(radio_layout)
        self.age_input_stack = QStackedWidget()
        self.age_input = QLineEdit()
        self.age_input.setValidator(QIntValidator(1, 100))
        self.age_input.setPlaceholderText("גיל בשנים (לדוגמה: 8)")
        self.age_input_stack.addWidget(self.age_input)
        self.year_input = QLineEdit()
        self.year_input.setValidator(QIntValidator(1950, datetime.now().year))
        self.year_input.setPlaceholderText(f"שנת שתילה (לדוגמה: {datetime.now().year - 8})")
        self.age_input_stack.addWidget(self.year_input)
        age_group_layout.addWidget(self.age_input_stack)
        self.by_age_radio.setChecked(True)
        self.by_age_radio.toggled.connect(lambda: self.age_input_stack.setCurrentIndex(0))
        self.by_year_radio.toggled.connect(lambda: self.age_input_stack.setCurrentIndex(1))
        parent_layout.addWidget(age_group)

    def create_thinning_tabs(self):
        tab_widget = QTabWidget()
        self.thinning_tabs = tab_widget  # שמירת רפרנס

        # טאב פרוטוקול כללי
        general_tab = QWidget()
        general_layout = QFormLayout(general_tab)
        general_layout.setSpacing(16)
        general_layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self.branches_count_general = QLineEdit("25")
        self.branches_count_general.setValidator(QIntValidator(1, 100))
        self.fronds_count_general = QLineEdit("120")
        self.fronds_count_general.setValidator(QIntValidator(10, 300))
        # שדה נוסף: מספר אשכולות בפרוטוקול הכללי
        self.clusters_count_general = QLineEdit("8")
        self.clusters_count_general.setValidator(QIntValidator(1, 500))
        general_layout.addRow("מספר סנסנים לאשכול:", self.branches_count_general)
        general_layout.addRow("מספר חנטים לסנסן:", self.fronds_count_general)
        general_layout.addRow("מספר אשכולות:", self.clusters_count_general)
        tab_widget.addTab(general_tab, "פרוטוקול כללי")

        # טאב פרוטוקול לפי דור
        generation_tab = QWidget()
        generation_scroll = QScrollArea()
        generation_scroll.setWidgetResizable(True)
        generation_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        generation_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        generation_scroll_widget = QWidget()
        generation_layout = QFormLayout(generation_scroll_widget)
        generation_scroll.setWidget(generation_scroll_widget)

        generation_tab_layout = QVBoxLayout(generation_tab)
        generation_tab_layout.setContentsMargins(0, 0, 0, 0)
        generation_tab_layout.addWidget(generation_scroll)

        generation_layout.setSpacing(12)
        generation_layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self.branches_upper = QLineEdit("22")
        self.fronds_upper = QLineEdit("110")
        self.branches_middle = QLineEdit("25")
        self.fronds_middle = QLineEdit("120")
        self.branches_lower = QLineEdit("28")
        self.fronds_lower = QLineEdit("130")
        for w in [self.branches_upper, self.branches_middle, self.branches_lower]: w.setValidator(QIntValidator(1, 100))
        for w in [self.fronds_upper, self.fronds_middle, self.fronds_lower]: w.setValidator(QIntValidator(10, 300))

        generation_layout.addRow(QLabel("<b>דור עליון:</b>"))
        generation_layout.addRow("  סנסנים לאשכול:", self.branches_upper)
        generation_layout.addRow("  חנטים לסנסן:", self.fronds_upper)
        generation_layout.addRow(QLabel("<b>דור אמצעי:</b>"))
        generation_layout.addRow("  סנסנים לאשכול:", self.branches_middle)
        generation_layout.addRow("  חנטים לסנסן:", self.fronds_middle)
        generation_layout.addRow(QLabel("<b>דור תחתון:</b>"))
        generation_layout.addRow("  סנסנים לאשכול:", self.branches_lower)
        generation_layout.addRow("  חנטים לסנסן:", self.fronds_lower)

        tab_widget.addTab(generation_tab, "פרוטוקול לפי דור")
        return tab_widget

    def _get_tree_age(self):
        if self.by_age_radio.isChecked():
            age_str = self.age_input.text()
            if not age_str: raise ValueError("יש להזין גיל עץ.")
            return int(age_str)
        else:
            year_str = self.year_input.text()
            if not year_str: raise ValueError("יש להזין שנת שתילה.")
            planting_year = int(year_str)
            current_year = datetime.now().year
            age = current_year - planting_year
            if not (0 < age < 100): raise ValueError("הגיל המחושב אינו בטווח הגיוני (1-99).")
            return age

    def load_stations(self):
        self.worker = APIWorker(self.api_client)
        self.worker.data_ready.connect(self.on_stations_loaded)
        self.worker.error_occurred.connect(self.on_api_error)
        self.worker.start()

    def on_stations_loaded(self, stations):
        if isinstance(stations, list):
            self.stations_data = sorted(stations, key=lambda s: s.get('name', ''))
            station_options = [f"{s.get('name')} ({s.get('stationId')})" for s in self.stations_data]
            self.station_combo.clear()
            self.station_combo.addItems(station_options)
            self.station_combo.setPlaceholderText("בחר תחנה מהרשימה")
            self.station_combo.setEnabled(True)
            self.load_data_btn.setEnabled(True)
        else:
            self.on_api_error("פורמט נתוני התחנות אינו תקין.")

    def on_api_error(self, error_msg):
        self.data_display.setPlainText(f"❌ שגיאה: {error_msg}")
        QMessageBox.warning(self, "שגיאת רשת", f"לא ניתן לטעון נתוני תחנות:\n{error_msg}")

    def load_weather_data(self):
        """שלב 4.8: טעינת נתונים מטאורולוגיים היסטוריים"""
        current_index = self.station_combo.currentIndex()
        if current_index < 0:
            return

        station_id = self.stations_data[current_index]['stationId']
        station_name = self.stations_data[current_index]['name']
        current_year = datetime.now().year

        # הקפאת הממשק
        self._set_loading_state(True, f"טוען נתונים היסטוריים עבור תחנת {station_name}...")
        self.data_display.setPlainText(f"🔄 טוען נתונים מטאורולוגיים היסטוריים...\n\nתקופות נטענות:\n" +
                                        f"• התמיינות: {current_year-1}-11-01 עד {current_year}-02-10\n" +
                                        f"• פריחה: {current_year}-02-11 עד {current_year}-03-31\n" +
                                        f"• דילול: {current_year}-04-01 עד {current_year}-05-15")

        # יצירת Worker לטעינת נתונים היסטוריים
        self.historical_worker = HistoricalWeatherWorker(
            self.api_client, station_id, current_year
        )
        self.historical_worker.data_ready.connect(self.on_historical_data_loaded)
        self.historical_worker.error_occurred.connect(self.on_historical_data_error)
        self.historical_worker.progress_update.connect(self.on_progress_update)
        self.historical_worker.start()

    def on_progress_update(self, message: str):
        """שלב 4.9: עדכון סטטוס התקדמות"""
        self.status_label.setText(message)

    def on_historical_data_loaded(self, weather_features: dict):
        """שלב 4.10: טיפול בנתונים מטאורולוגיים שנטענו בהצלחה"""
        self.weather_features = weather_features

        # הצגת הפיצ'רים שחושבו
        features_text = "✅ נתונים מטאורולוגיים נטענו בהצלחה!\n\n"
        features_text += "═══ פיצ'רים מחושבים למודל 1א ═══\n\n"

        period_names = {
            'Inf_differentiation': 'התמיינות (נוב-פבר)',
            'Flowering': 'פריחה (פבר-מרץ)',
            'Thinning': 'דילול (אפר-מאי)'
        }

        for period_key, period_name in period_names.items():
            features_text += f"📅 {period_name}:\n"
            t_val = weather_features.get(f'T_{period_key}', 0)
            h_val = weather_features.get(f'H_{period_key}', 0)
            e_val = weather_features.get(f'E_{period_key}', 0)
            features_text += f"   • שעות חום (T): {t_val:.1f}\n"
            features_text += f"   • לחות ממוצעת (H): {h_val:.1f}%\n"
            features_text += f"   • אידוי כולל (E): {e_val:.2f} מ\"מ\n\n"

        self.data_display.setPlainText(features_text)

        # שחרור הממשק
        self._set_loading_state(False, "✅ נתונים נטענו בהצלחה - ניתן להריץ ניתוח")

    def on_historical_data_error(self, error_msg: str):
        """שלב 4.11: טיפול בשגיאת טעינת נתונים היסטוריים"""
        self.weather_features = None
        self.data_display.setPlainText(f"❌ שגיאה בטעינת נתונים היסטוריים:\n\n{error_msg}")
        self._set_loading_state(False, "❌ שגיאה בטעינת נתונים")
        QMessageBox.warning(self, "שגיאת נתונים",
                           f"לא ניתן היה לטעון נתונים מטאורולוגיים היסטוריים:\n{error_msg}")

    def request_analysis(self):
        """שלב 4.12: בקשת ניתוח והעברת נתונים ל-MainWindow"""
        try:
            # בדיקה שנתונים מטאורולוגיים נטענו
            if self.weather_features is None:
                QMessageBox.warning(self, "חסרים נתונים",
                                   "יש לטעון נתונים מטאורולוגיים לפני הרצת הניתוח.\n\n" +
                                   "בחר תחנה ולחץ על 'טען נתוני תחנה'.")
                return

            data = {'tree_age': self._get_tree_age()}
            data['weather_features'] = self.weather_features  # הוספת הפיצ'רים המטאורולוגיים

            if self.thinning_tabs.currentIndex() == 0:
                data['protocol_type'] = 'general'
                data['thinning'] = {
                    'branches': int(self.branches_count_general.text()),
                    'fronds': int(self.fronds_count_general.text()),
                    'clusters': int(self.clusters_count_general.text())
                }
            else:
                data['protocol_type'] = 'by_generation'
                data['thinning'] = {
                    'upper': {'branches': int(self.branches_upper.text()), 'fronds': int(self.fronds_upper.text())},
                    'middle': {'branches': int(self.branches_middle.text()), 'fronds': int(self.fronds_middle.text())},
                    'lower': {'branches': int(self.branches_lower.text()), 'fronds': int(self.fronds_lower.text())}
                }
            self.analysis_requested.emit(data)
        except (ValueError, TypeError) as e:
            QMessageBox.warning(self, "שגיאת קלט", f"אחד או יותר מהשדות אינו תקין.\n{e}")


class ResultsPage(QWidget):
    """
    שלב 5: דף תוצאות - מציג את תחזית היבול בגרף פעמון
    ללא חלק ההמלצות (כי המודל לא נותן ערך לזה)
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(40, 30, 40, 30)
        self.main_layout.setSpacing(20)

        # שלב 5.1: מסך placeholder לפני הרצת ניתוח
        self.placeholder_widget = QWidget()
        placeholder_layout = QVBoxLayout(self.placeholder_widget)
        placeholder_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        placeholder_label = QLabel("📊\nהזן נתונים בדף הקודם ולחץ 'נתח' כדי לראות כאן את התוצאות.")
        placeholder_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        placeholder_label.setStyleSheet("font-size: 18px; color: #6B7280;")
        placeholder_layout.addWidget(placeholder_label)
        self.main_layout.addWidget(self.placeholder_widget)

        # שלב 5.2: מסך התוצאות עצמו
        self.results_widget = QWidget()
        self.results_layout = QVBoxLayout(self.results_widget)
        self.results_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.addWidget(self.results_widget)
        self.results_widget.setVisible(False)

        title = QLabel("📊 תוצאות ניתוח ותחזית יבול - מודל 1א")
        title.setObjectName("PageTitle")
        apply_shadow(title, blur_radius=5, x_offset=1, y_offset=2, color=QColor(0, 0, 0, 20))
        subtitle = QLabel("תחזית יבול (ק\"ג לעץ) בצומת ההחלטה (אפריל-מאי) מבוססת XGBoost")
        subtitle.setObjectName("PageSubtitle")
        self.results_layout.addWidget(title)
        self.results_layout.addWidget(subtitle)

        # שלב 5.3: כרטיס הגרף
        chart_card = QFrame()
        chart_card.setObjectName("Card")
        apply_shadow(chart_card)
        chart_card_layout = QVBoxLayout(chart_card)
        chart_title = QLabel("🎯 Yield Prediction Distribution (XGBoost Model 1A)")
        chart_title.setObjectName("CardTitle")
        self.yield_dist_chart = YieldDistributionChart()
        chart_card_layout.addWidget(chart_title)
        chart_card_layout.addWidget(self.yield_dist_chart)

        # שלב 5.4: תיבת סיכום מספרי
        summary_layout = QHBoxLayout()
        summary_layout.setSpacing(20)

        self.prediction_label = QLabel()
        self.prediction_label.setStyleSheet(f"""
            font-size: 24px; 
            font-weight: bold; 
            color: {COLORS['primary_dark']};
            padding: 15px;
            background-color: {COLORS['background_darker']};
            border-radius: 10px;
        """)
        self.prediction_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.confidence_label = QLabel()
        self.confidence_label.setStyleSheet(f"""
            font-size: 16px; 
            color: {COLORS['text_secondary']};
            padding: 15px;
            background-color: {COLORS['background_darker']};
            border-radius: 10px;
        """)
        self.confidence_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        summary_layout.addWidget(self.prediction_label, 2)
        summary_layout.addWidget(self.confidence_label, 1)

        chart_card_layout.addLayout(summary_layout)
        self.results_layout.addWidget(chart_card)

    def update_results(self, results):
        """שלב 5.5: עדכון התוצאות בממשק"""
        self.placeholder_widget.setVisible(False)
        self.results_widget.setVisible(True)

        mean_yield = results['yield_mean']
        std_yield = results['yield_std']

        # ציור הגרף
        self.yield_dist_chart.plot(mean=mean_yield, std_dev=std_yield)

        # עדכון תיבות הסיכום
        self.prediction_label.setText(f"🌴 Predicted Yield: {mean_yield:.1f} kg/tree")
        self.confidence_label.setText(
            f"68% Confidence: {mean_yield - std_yield:.1f} - {mean_yield + std_yield:.1f} kg\n"
            f"95% Confidence: {mean_yield - 2*std_yield:.1f} - {mean_yield + 2*std_yield:.1f} kg"
        )


# =====================================================================================
# 4. החלון הראשי של האפליקציה
# =====================================================================================
class MainWindow(QMainWindow):
    """
    שלב 6: החלון הראשי - מנהל את כל הדפים ואת מודל ה-XGBoost
    """
    def __init__(self):
        super().__init__()
        self.API_TOKEN = "1a901e45-9028-44ff-bd2c-35e82407fb9b"
        self.api_client = WeatherAPIClient(self.API_TOKEN)
        self.data_processor = DataProcessor()

        # שלב 6.1: טעינת מודל XGBoost
        self.xgb_model = None
        self._load_xgboost_model()

        self.setWindowTitle("דילול חנטי תמרים - מודל 1א")
        self.setWindowIcon(QIcon('volcani_logo.png'))

        self.setMinimumSize(960, 720)
        self.resize(1280, 800)

        self.setup_ui()
        self.center_window()

    def _load_xgboost_model(self):
        """שלב 6.2: טעינת מודל XGBoost מקובץ JSON"""
        try:
            model_path = 'xgboost_yield_model_1a.json'
            self.xgb_model = xgb.Booster()
            self.xgb_model.load_model(model_path)
            print(f"✅ מודל XGBoost נטען בהצלחה: {model_path}")
        except Exception as e:
            print(f"❌ שגיאה בטעינת מודל XGBoost: {e}")
            self.xgb_model = None

    def center_window(self):
        screen = self.screen().geometry()
        self.move((screen.width() - self.width()) // 2, (screen.height() - self.height()) // 2)

    def setup_ui(self):
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.setCentralWidget(main_splitter)

        self.nav_bar = QListWidget()
        self.nav_bar.setMinimumWidth(200)
        self.nav_bar.setMaximumWidth(320)
        self.nav_bar.addItem(QListWidgetItem("🏠   עמוד הבית"))
        self.nav_bar.addItem(QListWidgetItem("📝   הזנת נתונים"))
        self.nav_bar.addItem(QListWidgetItem("📊   תוצאות וניתוח"))
        self.nav_bar.setCurrentRow(0)

        self.stacked_widget = QStackedWidget()
        self.home_page = HomePage()
        self.data_entry_page = DataEntryPage(self.api_client)
        self.results_page = ResultsPage()
        self.stacked_widget.addWidget(self.home_page)
        self.stacked_widget.addWidget(self.data_entry_page)
        self.stacked_widget.addWidget(self.results_page)

        main_splitter.addWidget(self.nav_bar)
        main_splitter.addWidget(self.stacked_widget)
        main_splitter.setSizes([260, 1020])
        main_splitter.setStretchFactor(1, 1)

        self.nav_bar.currentRowChanged.connect(self.stacked_widget.setCurrentIndex)
        self.data_entry_page.analysis_requested.connect(self.run_analysis)

    def run_analysis(self, data):
        """שלב 6.3: הרצת ניתוח עם מודל XGBoost"""
        results = self.calculate_results(data)
        self.results_page.update_results(results)
        self.nav_bar.setCurrentRow(2)
        self.statusBar().showMessage("✅ ניתוח הושלם בהצלחה. מציג תוצאות.", 5000)

    def calculate_results(self, data):
        """
        שלב 6.4: חישוב תחזית יבול באמצעות מודל XGBoost

        הפיצ'רים הנדרשים למודל 1א (לפי השמות שהמודל אומן עליהם):
        - Tree age, year
        - Thinning_Upper_Fruits Bunch-1, Thinning_Center_Fruits Bunch-1, Thinning_Lower_Fruits Bunch-1
        - Thinning_Bunches, Thinning_Fruits Tree-1
        - T/H/E עבור שלוש תקופות פיזיולוגיות
        """
        age = data['tree_age']
        weather_features = data.get('weather_features', {})

        # שלב 6.4.1: הכנת נתוני החקלאי
        if data['protocol_type'] == 'general':
            p = data['thinning']
            # בפרוטוקול כללי - אותם ערכי חנטים לכל הדורות
            # חנטים לאשכול = סנסנים * חנטים לסנסן
            fruits_per_bunch = p['branches'] * p['fronds']
            user_inputs = {
                'branches': p['branches'],
                'clusters': p['clusters'],
                'upper_fronds': fruits_per_bunch,
                'middle_fronds': fruits_per_bunch,
                'lower_fronds': fruits_per_bunch,
            }
        else:
            # פרוטוקול לפי דור - ערכים שונים לכל דור
            p = data['thinning']
            user_inputs = {
                'branches': int((p['upper']['branches'] + p['middle']['branches'] + p['lower']['branches']) / 3),
                'clusters': 8,  # ברירת מחדל
                'upper_fronds': p['upper']['branches'] * p['upper']['fronds'],
                'middle_fronds': p['middle']['branches'] * p['middle']['fronds'],
                'lower_fronds': p['lower']['branches'] * p['lower']['fronds'],
            }

        # שלב 6.4.2: בניית וקטור הקלט למודל
        current_year = datetime.now().year
        input_df = self.data_processor.prepare_input_vector(
            user_inputs, weather_features, age, current_year
        )

        print(f"DEBUG: Input features for model: {input_df.to_dict('records')[0]}")

        # שלב 6.4.3: הרצת המודל
        if self.xgb_model is not None:
            try:
                dmatrix = xgb.DMatrix(input_df)
                prediction = self.xgb_model.predict(dmatrix)
                mean_yield = float(prediction[0])

                # סטיית תקן משוערת (בהתאם לאי-ודאות המודל)
                # ניתן לשפר זאת עם quantile regression או bootstrap
                std_yield = mean_yield * 0.15  # 15% אי-ודאות

                print(f"✅ תחזית מודל XGBoost: {mean_yield:.2f} ± {std_yield:.2f} ק\"ג/עץ")

            except Exception as e:
                print(f"❌ שגיאה בהרצת המודל: {e}")
                mean_yield = self._fallback_prediction(data)
                std_yield = mean_yield * 0.20
        else:
            # שלב 6.4.4: חישוב גיבוי אם המודל לא נטען
            mean_yield = self._fallback_prediction(data)
            std_yield = mean_yield * 0.20

        return {'yield_mean': mean_yield, 'yield_std': std_yield}

    def _fallback_prediction(self, data):
        """שלב 6.5: חיזוי גיבוי (פשוט) אם המודל לא זמין"""
        age = data['tree_age']
        if data['protocol_type'] == 'general':
            p = data['thinning']
            fruitlets_per_tree = p['clusters'] * p['branches'] * p['fronds']
        else:
            p = data['thinning']
            avg_branches = (p['upper']['branches'] + p['middle']['branches'] + p['lower']['branches']) / 3
            avg_fronds = (p['upper']['fronds'] + p['middle']['fronds'] + p['lower']['fronds']) / 3
            fruitlets_per_tree = 8 * avg_branches * avg_fronds

        # חישוב גס: 10 גרם לפרי בממוצע
        estimated_yield = (fruitlets_per_tree * 10) / 1000  # בק"ג

        # התאמה לפי גיל
        if age < 5:
            estimated_yield *= 0.6
        elif age > 20:
            estimated_yield *= 0.85

        return max(20, min(200, estimated_yield))


# =====================================================================================
# 5. פונקציית הרצה ראשית
# =====================================================================================
def main():
    """
    שלב 7: נקודת הכניסה הראשית לאפליקציה

    סדר הפעולות:
    1. יצירת אפליקציית PyQt6
    2. החלת עיצוב RTL וסגנונות
    3. הצגת מסך פתיחה (Splash)
    4. טעינת החלון הראשי עם מודל XGBoost
    """
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLES)
    app.setLayoutDirection(Qt.LayoutDirection.RightToLeft)
    app.setApplicationName("מערכת תמיכת החלטה לחקלאים")
    app.setApplicationVersion("5.0-xgboost-model-1a")
    app.setOrganizationName("מכון וולקני - ARO")

    splash = AnimatedSplashScreen()
    splash.show()

    main_window = MainWindow()
    QTimer.singleShot(4000, lambda: (splash.close(), main_window.show()))

    sys.exit(app.exec())


if __name__ == "__main__":
    main()

