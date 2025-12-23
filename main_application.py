# main_application.py
import sys
import json
import requests
import numpy as np
from datetime import datetime
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from scipy.stats import norm

# =====================================================================================
#  ייבוא מסך הפתיחה מהקובץ הנפרד
# =====================================================================================
from splash_screen import AnimatedSplashScreen

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
    def __init__(self, api_token: str):
        self.api_token = api_token
        self.base_url = "https://api.ims.gov.il/v1/envista"
        self.headers = {"Authorization": f"ApiToken {api_token}"}

    def get_stations(self):
        try:
            response = requests.get(f"{self.base_url}/stations", headers=self.headers, timeout=15)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"שגיאה בקריאת נתוני תחנות: {e}")

    def get_station_data(self, station_id: int):
        try:
            url = f"{self.base_url}/stations/{station_id}/data/latest"
            response = requests.get(url, headers=self.headers, timeout=15)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"שגיאה בקריאת נתונים מטאורולוגיים: {e}")


class APIWorker(QThread):
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


class YieldDistributionChart(FigureCanvas):
    def __init__(self):
        self.fig = Figure(figsize=(8, 4.5), facecolor=COLORS['card'])
        super().__init__(self.fig)
        self.setMinimumHeight(350)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor(COLORS['background'])
        self.fig.patch.set_alpha(0)

    def plot(self, mean, std_dev):
        self.ax.clear()
        x = np.linspace(mean - 4 * std_dev, mean + 4 * std_dev, 400)
        y = norm.pdf(x, mean, std_dev)
        self.ax.plot(x, y, color=COLORS['primary'], linewidth=2.5, label='התפלגות התחזית')
        self.ax.fill_between(x, y, where=(x >= mean - std_dev) & (x <= mean + std_dev),
                             color=COLORS['primary_light'], alpha=0.6, label='68% ביטחון')
        self.ax.fill_between(x, y, where=(x >= mean - 2 * std_dev) & (x <= mean + 2 * std_dev),
                             color=COLORS['primary_light'], alpha=0.3, label='95% ביטחון')
        self.ax.axvline(mean, color=COLORS['accent'], linestyle='--', linewidth=2, label=f'יבול ממוצע: {mean:.1f} ק"ג')
        self.ax.set_xlabel('תחזית יבול (ק"ג לעץ)', fontsize=12, color=COLORS['text_secondary'])
        self.ax.set_ylabel('צפיפות הסתברות', fontsize=12, color=COLORS['text_secondary'])
        self.ax.tick_params(colors=COLORS['text_secondary'])
        for spine in self.ax.spines.values():
            spine.set_visible(False)
        self.ax.grid(True, axis='y', alpha=0.3, linestyle='--')
        self.ax.get_yaxis().set_ticks([])
        self.ax.legend(loc='upper right', frameon=False)
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
    analysis_requested = pyqtSignal(dict)

    def __init__(self, api_client, parent=None):
        super().__init__(parent)
        self.api_client = api_client
        self.stations_data = []

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(40, 30, 40, 30)
        main_layout.setSpacing(20)

        title = QLabel("📝 הזנת נתונים ופרוטוקול דילול")
        title.setObjectName("PageTitle")
        apply_shadow(title, blur_radius=5, x_offset=1, y_offset=2, color=QColor(0, 0, 0, 20))
        subtitle = QLabel("מלאו את הנתונים הבאים כדי לקבל תחזית יבול והמלצות.")
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
        weather_title = QLabel("נתוני אקלים עדכניים")
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
        self.data_display = QTextEdit()
        self.data_display.setPlainText("בחר תחנה ולחץ על טען נתונים...")
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
        self.analyze_btn = QPushButton("🔬 נתח וצור תחזית")
        self.analyze_btn.setObjectName("AnalyzeButton")
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

        self.branches_upper = QLineEdit("22");
        self.fronds_upper = QLineEdit("110")
        self.branches_middle = QLineEdit("25");
        self.fronds_middle = QLineEdit("120")
        self.branches_lower = QLineEdit("28");
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
        current_index = self.station_combo.currentIndex()
        if current_index < 0: return
        station_id = self.stations_data[current_index]['stationId']
        station_name = self.stations_data[current_index]['name']
        self.data_display.setPlainText(f"🔄 טוען נתונים עבור תחנת {station_name}...")
        self.load_data_btn.setEnabled(False)
        self.data_worker = APIWorker(self.api_client, station_id)
        self.data_worker.data_ready.connect(self.on_data_loaded)
        self.data_worker.error_occurred.connect(self.on_data_error)
        self.data_worker.start()

    def on_data_loaded(self, data):
        formatted_data = json.dumps(data.get('data', {}), indent=2, ensure_ascii=False)
        self.data_display.setPlainText(formatted_data)
        self.load_data_btn.setEnabled(True)

    def on_data_error(self, error_msg):
        self.data_display.setPlainText(f"❌ שגיאה בטעינת נתונים:\n{error_msg}")
        self.load_data_btn.setEnabled(True)
        QMessageBox.warning(self, "שגיאת נתונים", f"לא ניתן היה לטעון נתונים מהתחנה:\n{error_msg}")

    def request_analysis(self):
        try:
            data = {'tree_age': self._get_tree_age()}
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
    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(40, 30, 40, 30)
        self.main_layout.setSpacing(20)
        self.placeholder_widget = QWidget()
        placeholder_layout = QVBoxLayout(self.placeholder_widget)
        placeholder_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        placeholder_label = QLabel("📊\nהזן נתונים בדף הקודם ולחץ 'נתח' כדי לראות כאן את התוצאות.")
        placeholder_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        placeholder_label.setStyleSheet("font-size: 18px; color: #6B7280;")
        placeholder_layout.addWidget(placeholder_label)
        self.main_layout.addWidget(self.placeholder_widget)
        self.results_widget = QWidget()
        self.results_layout = QVBoxLayout(self.results_widget)
        self.results_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.addWidget(self.results_widget)
        self.results_widget.setVisible(False)
        title = QLabel("📊 תוצאות ניתוח ותחזית יבול")
        title.setObjectName("PageTitle")
        apply_shadow(title, blur_radius=5, x_offset=1, y_offset=2, color=QColor(0, 0, 0, 20))
        subtitle = QLabel("מבוסס על הנתונים שהוזנו וניתוח מודלים חקלאיים.")
        subtitle.setObjectName("PageSubtitle")
        self.results_layout.addWidget(title)
        self.results_layout.addWidget(subtitle)
        cards_layout = QGridLayout()
        cards_layout.setSpacing(25)
        self.results_layout.addLayout(cards_layout)
        chart_card = QFrame()
        chart_card.setObjectName("Card")
        apply_shadow(chart_card)
        chart_card_layout = QVBoxLayout(chart_card)
        chart_title = QLabel("🎯 התפלגות תחזית היבול (Model Confidence)")
        chart_title.setObjectName("CardTitle")
        self.yield_dist_chart = YieldDistributionChart()
        chart_card_layout.addWidget(chart_title)
        chart_card_layout.addWidget(self.yield_dist_chart)
        cards_layout.addWidget(chart_card, 0, 0, 2, 1)
        rec_card = QFrame()
        rec_card.setObjectName("Card")
        apply_shadow(rec_card)
        rec_card_layout = QVBoxLayout(rec_card)
        rec_title = QLabel("💡 המלצות מותאמות אישית")
        rec_title.setObjectName("CardTitle")
        self.rec_text = QTextEdit()
        self.rec_text.setReadOnly(True)
        rec_card_layout.addWidget(rec_title)
        rec_card_layout.addWidget(self.rec_text)
        cards_layout.addWidget(rec_card, 0, 1)
        cards_layout.setColumnStretch(0, 2)
        cards_layout.setColumnStretch(1, 1)

    def update_results(self, results):
        self.placeholder_widget.setVisible(False)
        self.results_widget.setVisible(True)
        self.yield_dist_chart.plot(mean=results['yield_mean'], std_dev=results['yield_std'])
        recommendations_html = ""
        for rec in results['recommendations']:
            recommendations_html += f"<p style='margin: 5px 0;'><b>•</b> {rec}</p>"
        self.rec_text.setHtml(f"<div style='font-size: 14px; line-height: 1.6;'>{recommendations_html}</div>")


# =====================================================================================
# 4. החלון הראשי של האפליקציה
# =====================================================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.API_TOKEN = "1a901e45-9028-44ff-bd2c-35e82407fb9b"
        self.api_client = WeatherAPIClient(self.API_TOKEN)

        self.setWindowTitle("דילול חנטי תמרים")
        self.setWindowIcon(QIcon('volcani_logo.png'))

        self.setMinimumSize(960, 720)
        self.resize(1280, 800)

        self.setup_ui()
        self.center_window()

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
        results = self.calculate_results(data)
        self.results_page.update_results(results)
        self.nav_bar.setCurrentRow(2)
        self.statusBar().showMessage("✅ ניתוח הושלם בהצלחה. מציג תוצאות.", 5000)

    def calculate_results(self, data):
        age = data['tree_age']
        if data['protocol_type'] == 'general':
            p = data['thinning']
            structure_factor = (p['branches'] * 0.1 + p['fronds'] * 0.05) / 10
        else:
            p = data['thinning']
            upper_factor = (p['upper']['branches'] * 0.1 + p['upper']['fronds'] * 0.05)
            middle_factor = (p['middle']['branches'] * 0.1 + p['middle']['fronds'] * 0.05)
            lower_factor = (p['lower']['branches'] * 0.1 + p['lower']['fronds'] * 0.05)
            structure_factor = (upper_factor * 0.5 + middle_factor * 0.3 + lower_factor * 0.2) / 10
        if age < 5:
            base_yield = 60
        elif age < 10:
            base_yield = 100
        elif age < 20:
            base_yield = 120
        else:
            base_yield = 100
        age_factor = min(age / 10, 2.0)
        mean_yield = max(25, min(200, base_yield * age_factor * structure_factor))
        std_dev_factor = 0.20 if age < 7 else 0.12
        std_yield = mean_yield * std_dev_factor
        recommendations = []
        if age < 5:
            recommendations.append("עץ צעיר: בצע דילול עדין לתמיכה בפיתוח מבנה העץ.")
        elif age > 20:
            recommendations.append("עץ בוגר: ניתן לבצע דילול אגרסיבי יותר לחידוש הצמיחה.")
        else:
            recommendations.append("עץ בוגר: דילול סטנדרטי מתאים לשמירה על יבול.")
        if data['protocol_type'] == 'general':
            avg_fronds = data['thinning']['fronds']
        else:
            p = data['thinning']; avg_fronds = (p['upper']['fronds'] + p['middle']['fronds'] + p['lower']['fronds']) / 3
        if avg_fronds > 150:
            recommendations.append("עומס חנטים ממוצע גבוה: בצע דילול נוסף לשיפור איכות הפרי ואוורור.")
        elif avg_fronds < 80:
            recommendations.append("עומס חנטים ממוצע נמוך: בדוק את מצב בריאות העץ וההשקיה.")
        if mean_yield > 150:
            recommendations.append("תחזית יבול גבוהה: הקפד על השקיה ודישון לתמיכה ביבול.")
        elif mean_yield < 60:
            recommendations.append("תחזית יבול נמוכה: שקול התאמת פרוטוקול הדישון וההשקיה.")
        current_month = datetime.now().month
        if current_month in [11, 12, 1, 2]:
            recommendations.append("המלצה עונתית: חורף הוא זמן מתאים לגיזום ודילול.")
        elif current_month in [3, 4, 5]:
            recommendations.append("המלצה עונתית: הימנע מגיזום כבד בתקופת הפריחה והחנטה.")
        else:
            recommendations.append("המלצה עונתית: בקיץ, התמקד בדילול פרי עדין במידת הצורך.")
        return {'yield_mean': mean_yield, 'yield_std': std_yield, 'recommendations': recommendations}


# =====================================================================================
# 5. פונקציית הרצה ראשית
# =====================================================================================
def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLES)
    app.setLayoutDirection(Qt.LayoutDirection.RightToLeft)
    app.setApplicationName("מערכת תמיכת החלטה לחקלאים")
    app.setApplicationVersion("4.0-responsive-button")  # Updated version
    app.setOrganizationName("חקלאות חכמה בע\"מ")
    splash = AnimatedSplashScreen()
    splash.show()
    main_window = MainWindow()
    QTimer.singleShot(4000, lambda: (splash.close(), main_window.show()))
    sys.exit(app.exec())


if __name__ == "__main__":
    main()