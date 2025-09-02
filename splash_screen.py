# splash_screen.py
import sys
import numpy as np
import random
import math
from PyQt6.QtWidgets import QApplication, QSplashScreen, QGraphicsBlurEffect
from PyQt6.QtCore import Qt, QTimer, QRect
from PyQt6.QtGui import QPixmap, QPainter, QLinearGradient, QColor, QFont, QPen


class NeuralNode:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.original_x = x
        self.original_y = y
        self.size = random.uniform(3, 8)
        self.pulse_phase = random.uniform(0, 2 * math.pi)
        self.drift_phase = random.uniform(0, 2 * math.pi)

    def update(self, time):
        # תנועה עדינה של הנוירונים
        self.x = self.original_x + 15 * math.sin(time * 0.001 + self.drift_phase)
        self.y = self.original_y + 10 * math.cos(time * 0.0008 + self.drift_phase)

        # פעימת הנוירון
        pulse = math.sin(time * 0.003 + self.pulse_phase)
        self.size = 4 + 2 * pulse


class AnimatedSplashScreen(QSplashScreen):
    def __init__(self):
        super().__init__()
        self.setFixedSize(600, 450)
        self.setWindowFlags(Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.FramelessWindowHint)
        screen_geometry = QApplication.primaryScreen().geometry()
        self.move((screen_geometry.width() - self.width()) // 2, (screen_geometry.height() - self.height()) // 2)

        # נסה לטעון את הלוגו, אם לא קיים, צור Pixmap ריק
        try:
            self.logo = QPixmap('volcani_logo.png')
        except:
            self.logo = QPixmap()

        self.loading_states = ["מאתחל רכיבים...", "יוצר קשר עם השירות המטאורולוגי...", "טוען מודלים חכמים...",
                               "בונה ממשק ויזואלי...", "כמעט סיימנו..."]
        self.current_state_index = 0
        self.animation_time = 0
        self.logo_pulse_phase = 0

        # יצירת רשת נוירונים
        self.create_neural_network()

        self.animation_timer = QTimer(self, timeout=self.update_animation)
        self.animation_timer.start(16)  # ~60 FPS

        self.text_timer = QTimer(self, timeout=self.next_loading_state)
        self.text_timer.start(800)

    def create_neural_network(self):
        """יצירת רשת נוירונים עם קשרים ביניהם"""
        self.nodes = []
        self.connections = []

        # יצירת נוירונים במיקומים רנדומליים
        for _ in range(25):
            x = random.uniform(50, 550)
            y = random.uniform(50, 400)
            self.nodes.append(NeuralNode(x, y))

        # יצירת קשרים בין נוירונים קרובים
        for i, node1 in enumerate(self.nodes):
            for j, node2 in enumerate(self.nodes[i + 1:], i + 1):
                distance = math.sqrt((node1.x - node2.x) ** 2 + (node1.y - node2.y) ** 2)
                if distance < 120 and random.random() < 0.3:  # רק קשרים קרובים וסיכוי של 30%
                    self.connections.append((i, j, random.uniform(0, 2 * math.pi)))

    def update_animation(self):
        self.animation_time += 16  # 16ms per frame
        self.logo_pulse_phase = (self.logo_pulse_phase + 0.05) % (2 * math.pi)

        # עדכון מיקום הנוירונים
        for node in self.nodes:
            node.update(self.animation_time)

        self.update()

    def next_loading_state(self):
        self.current_state_index = (self.current_state_index + 1) % len(self.loading_states)

    def draw_neural_network(self, painter):
        """ציור רשת הנוירונים המונפשת"""
        w, h = self.width(), self.height()

        # ציור הקשרים (הקווים) בין הנוירונים
        painter.setOpacity(0.15)
        for i, j, phase in self.connections:
            node1, node2 = self.nodes[i], self.nodes[j]

            # אנימציה של עוצמת הקשר
            pulse = math.sin(self.animation_time * 0.002 + phase)
            alpha = int(50 + 30 * pulse)

            # צבע דינמי לקשר
            hue = (self.animation_time * 0.1 + phase * 50) % 360
            color = QColor.fromHsv(int(hue), 100, 200, alpha)

            painter.setPen(QPen(color, 1.5))
            painter.drawLine(int(node1.x), int(node1.y), int(node2.x), int(node2.y))

            # אפקט של "פעימה" הנוסעת לאורך הקשר
            if pulse > 0.7:  # רק כשהפעימה חזקה
                mid_x = (node1.x + node2.x) / 2
                mid_y = (node1.y + node2.y) / 2
                painter.setOpacity(0.6)
                painter.setBrush(QColor(255, 255, 255, 150))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawEllipse(int(mid_x - 2), int(mid_y - 2), 4, 4)

        # ציור הנוירונים
        painter.setOpacity(0.8)
        for node in self.nodes:
            # צבע דינמי לנוירון
            hue = (self.animation_time * 0.05 + node.pulse_phase * 100) % 360
            saturation = int(80 + 20 * math.sin(self.animation_time * 0.003 + node.pulse_phase))
            color = QColor.fromHsv(int(hue), saturation, 255)

            painter.setBrush(color)
            painter.setPen(QPen(QColor(255, 255, 255, 100), 1))
            painter.drawEllipse(int(node.x - node.size / 2), int(node.y - node.size / 2),
                                int(node.size), int(node.size))

            # אפקט זוהר חיצוני לנוירונים פעילים
            if math.sin(self.animation_time * 0.004 + node.pulse_phase) > 0.5:
                painter.setOpacity(0.3)
                painter.setBrush(QColor(255, 255, 255))
                painter.setPen(Qt.PenStyle.NoPen)
                glow_size = node.size * 1.8
                painter.drawEllipse(int(node.x - glow_size / 2), int(node.y - glow_size / 2),
                                    int(glow_size), int(glow_size))
                painter.setOpacity(0.8)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h = self.width(), self.height()

        # רקע גרדיאנט גוונים כהים של כחול
        gradient = QLinearGradient(0, 0, w, h)
        gradient.setColorAt(0, QColor('#1a1a2e'))  # כחול כהה מאוד
        gradient.setColorAt(0.3, QColor('#16213e'))  # כחול כהה עמוק
        gradient.setColorAt(0.7, QColor('#0f3460'))  # כחול כהה ים
        gradient.setColorAt(1, QColor('#0e4b99'))  # כחול כהה מעט בהיר יותר
        painter.fillRect(self.rect(), gradient)

        # ציור רשת הנוירונים
        self.draw_neural_network(painter)

        # ציור הלוגו עם אפקט הבהוב
        painter.setOpacity(1.0)
        if not self.logo.isNull():
            scaled_logo = self.logo.scaled(120, 120, Qt.AspectRatioMode.KeepAspectRatio,
                                           Qt.TransformationMode.SmoothTransformation)
            logo_x = (w - scaled_logo.width()) // 2
            logo_y = 80

            # אפקט הבהוב ללוגו
            pulse = math.sin(self.logo_pulse_phase)
            logo_opacity = 0.7 + 0.3 * pulse

            # אפקט זוהר סביב הלוגו
            painter.setOpacity(0.4 * pulse)
            painter.setBrush(QColor(255, 255, 255))
            painter.setPen(Qt.PenStyle.NoPen)
            glow_size = 140 + 20 * pulse
            painter.drawEllipse(int(logo_x - (glow_size - 120) / 2),
                                int(logo_y - (glow_size - 120) / 2),
                                int(glow_size), int(glow_size))

            # ציור הלוגו עם השקיפות המשתנה
            painter.setOpacity(logo_opacity)
            painter.drawPixmap(logo_x, logo_y, scaled_logo)

        # טקסט כותרת עם אפקט צל
        painter.setOpacity(1.0)
        painter.setPen(QColor('white'))
        painter.setFont(QFont('Inter', 24, QFont.Weight.Black))
        painter.setOpacity(0.3)
        painter.drawText(QRect(52, 222, 500, 40), Qt.AlignmentFlag.AlignCenter, 'מערכת AI לחקלאים')
        painter.setOpacity(1.0)
        painter.drawText(QRect(50, 220, 500, 40), Qt.AlignmentFlag.AlignCenter, 'מערכת AI לחקלאים')

        # תת כותרת
        painter.setFont(QFont('Inter', 16, QFont.Weight.Medium))
        painter.setOpacity(0.9)
        painter.drawText(QRect(50, 260, 500, 30), Qt.AlignmentFlag.AlignCenter, 'טכנולוגיה מתקדמת לחקלאות חכמה')

        # מחוון טעינה מעגלי מונפש
        painter.save()
        painter.translate(w // 2, h - 100)
        colors = [QColor('#FF6B6B'), QColor('#4ECDC4'), QColor('#45B7D1'), QColor('#96CEB4')]

        for i, color in enumerate(colors):
            rotation_speed = 2 + i * 0.5
            painter.rotate(self.animation_time * 0.01 * rotation_speed * ((-1) ** i))

            # שינוי עוצמת הצבע בהתאם לזמן
            alpha = int(150 + 100 * math.sin(self.animation_time * 0.003 + i))
            color.setAlpha(alpha)

            painter.setPen(QPen(color, 6, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            start_angle = i * 90 * 16
            span_angle = 60 * 16
            painter.drawArc(-25, -25, 50, 50, start_angle, span_angle)
        painter.restore()

        # טקסט מצב הטעינה
        painter.setFont(QFont('Inter', 13, QFont.Weight.Medium))
        painter.setOpacity(0.8)
        painter.drawText(QRect(50, h - 50, 500, 25), Qt.AlignmentFlag.AlignCenter,
                         self.loading_states[self.current_state_index])


# דוגמה לשימוש
if __name__ == "__main__":
    app = QApplication(sys.argv)
    splash = AnimatedSplashScreen()
    splash.show()

    # הצגת מסך הפתיחה למשך 5 שניות
    QTimer.singleShot(5000, splash.close)
    QTimer.singleShot(5000, app.quit)

    sys.exit(app.exec())