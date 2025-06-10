# ver10_customtkinter.py
# Decision support system for farmers - Advanced date palm pruning with CustomTkinter
import sys
import json
import threading
import queue
import requests
import numpy as np
import math
from datetime import datetime
import customtkinter as ctk
from tkinter import messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# --- Modern Color Palette ---
COLORS = {
    'primary': '#3B82F6',
    'primary_light': '#60A5FA',
    'primary_dark': '#2563EB',
    'secondary': '#10B981',
    'accent': '#F59E0B',
    'background': '#F1F5F9',  # Light Gray
    'surface': '#FFFFFF',
    'card': '#FFFFFF',
    'text': '#0F172A',  # Slate 900
    'text_secondary': '#64748B',  # Slate 500
    'success': '#10B981',
    'error': '#EF4444',
}


# --- API Client (No changes needed) ---
class WeatherAPIClient:
    def __init__(self, api_token: str):
        self.api_token = api_token
        self.base_url = "https://api.ims.gov.il/v1/envista"
        self.headers = {"Authorization": f"ApiToken {api_token}"}

    def get_stations(self):
        try:
            response = requests.get(f"{self.base_url}/stations", headers=self.headers, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error fetching station data: {e}")

    def get_station_data(self, station_id: int):
        try:
            url = f"{self.base_url}/stations/{station_id}/data/latest"
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error fetching meteorological data: {e}")


# --- Threading Worker for Non-Blocking API Calls ---
class APIWorker(threading.Thread):
    def __init__(self, api_client, operation='stations', station_id=None, q=None):
        super().__init__()
        self.api_client = api_client
        self.operation = operation
        self.station_id = station_id
        self.queue = q

    def run(self):
        try:
            if self.operation == 'stations':
                data = self.api_client.get_stations()
                self.queue.put(('stations_success', data))
            else:
                data = self.api_client.get_station_data(self.station_id)
                self.queue.put(('data_success', data))
        except Exception as e:
            self.queue.put(('error', str(e)))


# --- CustomTkinter Widgets ---

class AnimatedSplashScreen(ctk.CTkToplevel):
    def __init__(self):
        super().__init__()
        self.geometry("600x400")
        self.overrideredirect(True)  # Frameless window
        self.lift()
        self.attributes("-topmost", True)

        # Center the window
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        x = (screen_width - 600) // 2
        y = (screen_height - 400) // 2
        self.geometry(f"+{x}+{y}")

        self.canvas = ctk.CTkCanvas(self, width=600, height=400, highlightthickness=0)
        self.canvas.pack()

        # State
        self.loading_states = ["Initializing system...", "Connecting to weather service...", "Loading station data...",
                               "Preparing UI...", "Finishing up..."]
        self.current_state_index = 0
        self.loading_angle = 0
        self.draw_splash()
        self.update_animation()
        self.update_text()

    def draw_splash(self):
        self.canvas.create_rectangle(0, 0, 600, 400, fill=COLORS['primary_dark'], outline="")
        self.canvas.create_text(300, 120, text='🌴', font=("Segoe UI Emoji", 60), fill='white')
        self.canvas.create_text(300, 200, text='Farmer Decision Support System', font=("Segoe UI", 22, "bold"),
                                fill='white')
        self.canvas.create_text(300, 235, text='Advanced Date Pruning', font=("Segoe UI", 16), fill='white')
        self.status_text = self.canvas.create_text(300, 330, text=self.loading_states[0], font=("Segoe UI", 12),
                                                   fill='white')
        self.arc = self.canvas.create_arc(275, 275, 325, 325, start=0, extent=120, style='arc', outline='white',
                                          width=4)

    def update_animation(self):
        self.loading_angle = (self.loading_angle - 15) % 360
        self.canvas.itemconfig(self.arc, start=self.loading_angle)
        self.after(50, self.update_animation)

    def update_text(self):
        self.current_state_index = (self.current_state_index + 1) % len(self.loading_states)
        self.canvas.itemconfig(self.status_text, text=self.loading_states[self.current_state_index])
        self.after(800, self.update_text)


class GrowthChart(ctk.CTkFrame):
    def __init__(self, master, data):
        super().__init__(master, fg_color="white", corner_radius=12)

        fig = Figure(figsize=(7, 3.5), facecolor='white')
        ax = fig.add_subplot(111)

        months = data['months']
        growth_rates = data['growth_rate']
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

        colors = [COLORS['primary'], COLORS['primary_light']] * 6
        ax.bar(months, growth_rates, color=colors[:len(months)], alpha=0.8, width=0.6)
        ax.plot(months, growth_rates, color=COLORS['error'], linewidth=2, marker='o', markersize=5)

        ax.set_ylabel('Relative growth rate (%)', fontsize=10, color=COLORS['text_secondary'])
        ax.set_xlabel('Month', fontsize=10, color=COLORS['text_secondary'])
        ax.set_xticks(months)
        ax.set_xticklabels(month_names, fontsize=9)
        ax.grid(True, axis='y', alpha=0.3, linestyle='--')
        ax.set_facecolor('#F8FAFC')

        for spine in ax.spines.values():
            spine.set_visible(False)
        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=self)
        canvas.draw()
        canvas.get_tk_widget().pack(expand=True, fill='both', padx=10, pady=10)


class ManualProtocolDialog(ctk.CTkToplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Manual Dilution Protocol")
        self.geometry("500x520")
        self.transient(parent)
        self.grab_set()
        self.result = None

        self.configure(fg_color=COLORS['background'])

        # --- Title ---
        title_label = ctk.CTkLabel(self, text="🔧 Manual Dilution Protocol", font=("Segoe UI", 20, "bold"),
                                   text_color="white", fg_color=COLORS['primary'], height=60)
        title_label.pack(fill='x', padx=15, pady=15)

        # --- Form Frame ---
        form_frame = ctk.CTkFrame(self, fg_color=COLORS['card'], corner_radius=12)
        form_frame.pack(fill='both', expand=True, padx=15, pady=0)
        form_frame.grid_columnconfigure(1, weight=1)

        # --- Input Fields ---
        labels = ["Tree Age:", "Upper Clusters:", "Middle Clusters:", "Lower Clusters:",
                  "Number of Spadices (Sansanim):", "Number of Fronds (Chantim):"]
        self.entries = {}
        placeholders = ["e.g., 8", "1/4", "1/2", "1/4", "25", "120"]

        for i, (text, placeholder) in enumerate(zip(labels, placeholders)):
            label = ctk.CTkLabel(form_frame, text=text, font=("Segoe UI", 13), text_color=COLORS['text_secondary'])
            label.grid(row=i, column=0, padx=20, pady=10, sticky='w')
            entry = ctk.CTkEntry(form_frame, placeholder_text=placeholder, font=("Segoe UI", 13))
            entry.insert(0, placeholder)  # Pre-fill with default values
            entry.grid(row=i, column=1, padx=20, pady=10, sticky='ew')
            self.entries[text] = entry

        # --- Buttons ---
        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.pack(fill='x', padx=15, pady=15)
        btn_frame.grid_columnconfigure((0, 1), weight=1)

        cancel_btn = ctk.CTkButton(btn_frame, text="❌ Cancel", command=self.cancel, fg_color=COLORS['accent'],
                                   hover_color='#D97706')
        cancel_btn.grid(row=0, column=0, padx=5, sticky='ew')

        analyze_btn = ctk.CTkButton(btn_frame, text="🔬 Calculate Forecast", command=self.analyze,
                                    fg_color=COLORS['success'], hover_color='#059669')
        analyze_btn.grid(row=0, column=1, padx=5, sticky='ew')

    def analyze(self):
        try:
            self.result = {
                'tree_age': int(self.entries["Tree Age:"].get()),
                'clusters': {
                    'upper': self.entries["Upper Clusters:"].get(),
                    'middle': self.entries["Middle Clusters:"].get(),
                    'lower': self.entries["Lower Clusters:"].get()
                },
                'branches': int(self.entries["Number of Spadices (Sansanim):"].get()),
                'fronds': int(self.entries["Number of Fronds (Chantim):"].get())
            }
            self.destroy()
        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid numerical values.", parent=self)

    def cancel(self):
        self.result = None
        self.destroy()


class ResultsDialog(ctk.CTkToplevel):
    def __init__(self, parent, protocol_data, results):
        super().__init__(parent)
        self.title("Analysis Results")
        self.geometry("800x650")
        self.transient(parent)
        self.grab_set()
        self.configure(fg_color=COLORS['background'])

        # --- Main Scrollable Frame ---
        scroll_frame = ctk.CTkScrollableFrame(self, fg_color="transparent")
        scroll_frame.pack(fill='both', expand=True)

        # --- Title ---
        title = ctk.CTkLabel(scroll_frame, text="📊 Protocol Analysis Results", text_color="white",
                             font=("Segoe UI", 20, "bold"), fg_color=COLORS['success'], corner_radius=12, height=60)
        title.pack(fill='x', padx=15, pady=15)

        # --- Yield Forecast Card ---
        yield_card = ctk.CTkFrame(scroll_frame, fg_color=COLORS['card'], corner_radius=12)
        yield_card.pack(fill='x', padx=15, pady=7.5)

        ctk.CTkLabel(yield_card, text="🎯 Yield Forecast", font=("Segoe UI", 16, "bold"),
                     text_color=COLORS['primary']).pack(pady=(10, 5))

        yield_frame = ctk.CTkFrame(yield_card, fg_color="transparent")
        yield_frame.pack(pady=10, fill='x', expand=True)
        yield_frame.columnconfigure((0, 1), weight=1)

        yield_value = ctk.CTkLabel(yield_frame, text=f"{results['expected_yield']:.1f}", font=("Segoe UI", 48, "bold"),
                                   text_color=COLORS['success'])
        yield_value.grid(row=0, column=0, sticky='e', padx=(0, 10))
        yield_unit = ctk.CTkLabel(yield_frame, text="kg / tree", font=("Segoe UI", 18, "normal"),
                                  text_color=COLORS['text_secondary'])
        yield_unit.grid(row=0, column=1, sticky='w')

        # --- Growth Chart Card ---
        chart_card = ctk.CTkFrame(scroll_frame, fg_color=COLORS['card'], corner_radius=12)
        chart_card.pack(fill='x', padx=15, pady=7.5)
        ctk.CTkLabel(chart_card, text="📈 Annual Growth Distribution", font=("Segoe UI", 16, "bold"),
                     text_color=COLORS['primary']).pack(pady=(10, 5))
        GrowthChart(chart_card, results['growth_distribution']).pack(fill='x', padx=10, pady=(0, 10))

        # --- Recommendations Card ---
        rec_card = ctk.CTkFrame(scroll_frame, fg_color=COLORS['card'], corner_radius=12)
        rec_card.pack(fill='x', padx=15, pady=7.5)
        ctk.CTkLabel(rec_card, text="💡 Customized Recommendations", font=("Segoe UI", 16, "bold"),
                     text_color=COLORS['primary']).pack(pady=(10, 5))

        recommendations_text = "\n".join([f"• {rec}" for rec in results['recommendations']])
        rec_textbox = ctk.CTkTextbox(rec_card, fg_color="#F0F9FF", border_color="#BAE6FD", border_width=2,
                                     text_color="#0C4A6E", corner_radius=8, font=("Segoe UI", 13))
        rec_textbox.insert("1.0", recommendations_text)
        rec_textbox.configure(state="disabled")
        rec_textbox.pack(fill='x', expand=True, padx=15, pady=(0, 15))

        # --- Close Button ---
        close_btn = ctk.CTkButton(self, text="✅ Close", command=self.destroy, fg_color=COLORS['primary'], height=35)
        close_btn.pack(pady=15, padx=15, fill='x')


class MainWindow(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.API_TOKEN = "1a901e45-9028-44ff-bd2c-35e82407fb9b"  # Your API Token
        self.api_client = WeatherAPIClient(self.API_TOKEN)
        self.stations_data = []
        self.queue = queue.Queue()

        self.title("Farmer DSS - Date Pruning")
        self.geometry("1000x700")
        self.minsize(900, 600)
        self.configure(fg_color=COLORS['background'])

        self.setup_ui()
        self.load_stations()
        self.process_queue()

    def setup_ui(self):
        # --- Header ---
        header = ctk.CTkLabel(self, text="🌴 Decision Support System for Date Pruning", font=("Segoe UI", 24, "bold"),
                              text_color="white", fg_color=COLORS['primary'], corner_radius=12, height=70)
        header.pack(fill="x", padx=15, pady=15)

        # --- Main Content Frame ---
        main_frame = ctk.CTkFrame(self, fg_color="transparent")
        main_frame.pack(fill="both", expand=True, padx=15, pady=0)
        main_frame.grid_columnconfigure(1, weight=1)
        main_frame.grid_rowconfigure(0, weight=1)

        # --- Left Column (Actions) ---
        left_column = ctk.CTkFrame(main_frame, fg_color="transparent", width=350)
        left_column.grid(row=0, column=0, padx=(0, 10), sticky='ns')

        actions_card = ctk.CTkFrame(left_column, fg_color=COLORS['card'], corner_radius=12)
        actions_card.pack(fill='x', pady=0)

        ctk.CTkLabel(actions_card, text="⚙️ System Actions", font=("Segoe UI", 16, "bold"),
                     text_color=COLORS['primary']).pack(pady=(15, 10))

        desc_label = ctk.CTkLabel(actions_card,
                                  text="Enter a manual dilution protocol to receive an accurate yield forecast and personalized recommendations for your date palm.",
                                  wraplength=300, justify="left", font=("Segoe UI", 13), fg_color="#EBF8FF",
                                  text_color="#1E40AF", corner_radius=8)
        desc_label.pack(fill='x', padx=15, pady=10)

        protocol_btn = ctk.CTkButton(actions_card, text="🔧 Enter Dilution Protocol", command=self.open_protocol_dialog,
                                     fg_color=COLORS['success'], hover_color='#059669', font=("Segoe UI", 14, "bold"),
                                     height=40)
        protocol_btn.pack(fill='x', padx=15, pady=15)

        # --- Right Column (Weather Data) ---
        weather_card = ctk.CTkFrame(main_frame, fg_color=COLORS['card'], corner_radius=12)
        weather_card.grid(row=0, column=1, sticky='nsew')
        weather_card.grid_rowconfigure(2, weight=1)
        weather_card.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(weather_card, text="🌤️ Current Climate Data", font=("Segoe UI", 16, "bold"),
                     text_color=COLORS['primary']).grid(row=0, column=0, columnspan=2, pady=(15, 10))

        # Station Selector
        station_frame = ctk.CTkFrame(weather_card, fg_color='transparent')
        station_frame.grid(row=1, column=0, columnspan=2, padx=15, pady=5, sticky='ew')
        station_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(station_frame, text="Select Station:", font=("Segoe UI", 13, "bold")).grid(row=0, column=0,
                                                                                                padx=(0, 10))
        self.station_combo = ctk.CTkComboBox(station_frame, state="disabled")
        self.station_combo.grid(row=0, column=1, sticky='ew')

        self.load_data_btn = ctk.CTkButton(station_frame, text="📡 Load Data", state="disabled",
                                           command=self.load_weather_data, width=120)
        self.load_data_btn.grid(row=0, column=2, padx=(10, 0))

        # Data Display
        self.data_display = ctk.CTkTextbox(weather_card, font=("Consolas", 12), fg_color="#F8FAFC",
                                           border_color="#D1D5DB", border_width=1)
        self.data_display.insert("1.0", "🔄 Waiting for stations to load...")
        self.data_display.grid(row=2, column=0, columnspan=2, padx=15, pady=10, sticky='nsew')

        # --- Status Bar ---
        self.status_bar = ctk.CTkLabel(self, text="🔄 Initializing system...", font=("Segoe UI", 12), anchor='w',
                                       height=30, fg_color=COLORS['background'], text_color=COLORS['text_secondary'])
        self.status_bar.pack(fill='x', padx=15, pady=(5, 10))

    def load_stations(self):
        self.status_bar.configure(text="🔄 Loading station list...")
        APIWorker(self.api_client, 'stations', q=self.queue).start()

    def load_weather_data(self):
        try:
            station_text = self.station_combo.get()
            station_id = int(station_text.split('(')[-1].replace(')', ''))
        except (IndexError, ValueError):
            messagebox.showwarning("Selection Missing", "Please select a valid station from the list.")
            return

        self.data_display.delete("1.0", "end")
        self.data_display.insert("1.0", f"🔄 Loading data for station {station_text.split('(')[0]}...")
        self.load_data_btn.configure(state="disabled")
        self.status_bar.configure(text="🔄 Loading meteorological data...")
        APIWorker(self.api_client, 'data', station_id, q=self.queue).start()

    def process_queue(self):
        try:
            message_type, data = self.queue.get_nowait()

            if message_type == 'stations_success':
                self.stations_data = sorted(data, key=lambda s: s.get('name', ''))
                station_options = [f"{s.get('name')} ({s.get('stationId')})" for s in self.stations_data]
                self.station_combo.configure(values=station_options, state="readonly")
                self.station_combo.set(station_options[0])
                self.load_data_btn.configure(state="normal")
                self.data_display.delete("1.0", "end")
                self.data_display.insert("1.0",
                                         "✅ Stations loaded successfully.\n\n🔍 Select a station and load current data.")
                self.status_bar.configure(text=f"✅ Loaded {len(station_options)} stations. System is ready.")

            elif message_type == 'data_success':
                formatted_data = json.dumps(data, indent=2, ensure_ascii=False)
                self.data_display.delete("1.0", "end")
                self.data_display.insert("1.0", formatted_data)
                self.load_data_btn.configure(state="normal")
                self.status_bar.configure(text="✅ Meteorological data loaded successfully.")

            elif message_type == 'error':
                self.status_bar.configure(text=f"❌ Error: {data}")
                self.data_display.insert("end", f"\n\n❌ Error: {data}")
                messagebox.showerror("API Error", f"Could not connect to service:\n{data}")
                self.load_data_btn.configure(state="normal")

        except queue.Empty:
            pass  # No new messages
        finally:
            self.after(100, self.process_queue)  # Check again in 100ms

    def open_protocol_dialog(self):
        dialog = ManualProtocolDialog(self)
        self.wait_window(dialog)  # Wait for dialog to close

        if dialog.result:
            protocol_data = dialog.result
            results = self.calculate_results(protocol_data)
            ResultsDialog(self, protocol_data, results)

    def calculate_results(self, data):
        # This logic remains identical to the original script
        age_factor = min(data['tree_age'] / 10, 2.0)
        structure_factor = (data['branches'] * 0.1 + data['fronds'] * 0.05) / 10

        if data['tree_age'] < 5:
            base_yield = 60
        elif data['tree_age'] < 10:
            base_yield = 100
        elif data['tree_age'] < 20:
            base_yield = 120
        else:
            base_yield = 100

        expected_yield = max(25, min(200, base_yield * age_factor * structure_factor))

        months = np.arange(1, 13)
        growth_curve = np.where(months <= 6, 30 + 40 * np.sin(np.pi * months / 6),
                                70 + 25 * np.sin(np.pi * (months - 6) / 6))
        growth_curve = np.maximum(growth_curve, 10)

        recommendations = []
        if data['tree_age'] < 5:
            recommendations.append("Young tree: Perform gentle thinning to support structural development.")
        elif data['tree_age'] > 20:
            recommendations.append("Mature tree: More aggressive thinning can be done to rejuvenate growth.")
        else:
            recommendations.append("Adult tree: Standard thinning is suitable for maintaining yield.")

        if data['fronds'] > 150:
            recommendations.append("High fruit load: Perform additional thinning to improve fruit quality.")
        elif data['fronds'] < 80:
            recommendations.append("Low fruit load: Check the tree's health and irrigation status.")

        if expected_yield > 150:
            recommendations.append(
                "High yield forecast: Ensure adequate irrigation and fertilization to support the yield.")
        elif expected_yield < 60:
            recommendations.append("Low yield forecast: Consider adjusting the fertilization and irrigation protocol.")

        current_month = datetime.now().month
        if current_month in [11, 12, 1, 2]:
            recommendations.append("Seasonal tip: Winter is a suitable time for pruning and thinning.")
        elif current_month in [3, 4, 5]:
            recommendations.append("Seasonal tip: Avoid heavy pruning during the flowering period.")
        else:
            recommendations.append("Seasonal tip: Focus on gentle fruit thinning during the summer.")

        return {
            'expected_yield': expected_yield,
            'growth_distribution': {'months': months.tolist(), 'growth_rate': growth_curve.tolist()},
            'recommendations': recommendations
        }


# החלף את הפונקציה הקודמת בזו
def main():
    ctk.set_appearance_mode("Light")
    app = MainWindow() # יוצר את החלון הראשי
    app.mainloop()     # מפעיל אותו מיד, בלי מסך פתיחה

    # זו הגרסה הנוכחית שלך
    def main():
        ctk.set_appearance_mode("Light")

        app = MainWindow()

        splash = AnimatedSplashScreen()

        def show_main_window():
            splash.destroy()
            app.deiconify()  # Show the main window
            app.lift()
            app.attributes("-topmost", True)
            app.after(100, lambda: app.attributes("-topmost", False))

        app.withdraw()  # Hide main window initially
        app.after(3000, show_main_window)

        app.mainloop()