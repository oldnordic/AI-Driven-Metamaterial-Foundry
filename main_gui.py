import sys
import os
import pandas as pd
import json
import sqlite3
import threading
import time
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLineEdit, QLabel, QTextEdit, QProgressBar,
    QTabWidget, QGroupBox, QStatusBar, QMessageBox, QTableWidget,
    QTableWidgetItem, QHeaderView, QFormLayout
)
from PyQt5.QtCore import QObject, QThread, pyqtSignal, Qt

from ai_foundry import data_acquisition as da
from ai_foundry import ai_engine as ae

# --- Worker Class (No changes needed here) ---
class Worker(QObject):
    finished = pyqtSignal()
    progress = pyqtSignal(int, int, str, str)
    log_message = pyqtSignal(str)
    error_signal = pyqtSignal(str, str)
    result_ready = pyqtSignal(dict)

    def __init__(self, func, *args, **kwargs):
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def run(self):
        try:
            self.kwargs['log_message_callback'] = lambda msg: self.log_message.emit(msg)
            if 'progress_callback' in self.func.__code__.co_varnames:
                 self.kwargs['progress_callback'] = lambda value, maximum, msg, mode="determinate": self.progress.emit(value, maximum, msg, mode)
            
            result = self.func(*self.args, **self.kwargs)
            if result:
                self.result_ready.emit(result)
        except Exception as e:
            import traceback
            self.log_message.emit(f"Error in background task: {traceback.format_exc()}")
            self.error_signal.emit("Operation Failed", str(e))
        finally:
            self.finished.emit()

# --- Main Application Window ---
class AIFoundryApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI-Driven Metamaterial Foundry Control")
        self.setGeometry(100, 100, 1200, 800)
        self.data_for_ai = pd.DataFrame()
        self.init_ui()
        self.check_data_dir()

    def init_ui(self):
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        self.tabs = QTabWidget()
        self.main_layout.addWidget(self.tabs)

        self.setup_data_tab()
        self.setup_data_view_tab()
        self.setup_prediction_tab()
        self.setup_fabrication_tab()
        
        self.log_group = QGroupBox("Application Log")
        self.log_layout = QVBoxLayout(self.log_group)
        self.log_text_edit = QTextEdit()
        self.log_text_edit.setReadOnly(True)
        self.log_layout.addWidget(self.log_text_edit)
        self.main_layout.addWidget(self.log_group)
        self.setStatusBar(QStatusBar(self))

    # --- Methods for setting up UI tabs ---
    # (setup_data_tab, setup_data_view_tab, and setup_fabrication_tab remain the same)
    def setup_data_tab(self):
        """Sets up the UI for the Data Acquisition & Pre-processing tab."""
        self.data_tab = QWidget()
        self.tabs.addTab(self.data_tab, "Data Acquisition & Pre-processing")
        layout = QVBoxLayout(self.data_tab)

        # 1. Acquire Raw Data
        acquire_group = QGroupBox("1. Acquire Raw Data from Materials Project")
        acquire_layout = QVBoxLayout(acquire_group)
        self.acquire_button = QPushButton("Start Data Acquisition")
        self.acquire_button.clicked.connect(lambda: self.start_task(
            self._run_data_acquisition, 'acquire'
        ))
        acquire_layout.addWidget(self.acquire_button)
        self.acquire_progress_bar = QProgressBar()
        self.acquire_progress_bar.setTextVisible(True)
        acquire_layout.addWidget(self.acquire_progress_bar)
        self.acquire_status_label = QLabel("Ready")
        acquire_layout.addWidget(self.acquire_status_label)
        layout.addWidget(acquire_group)

        # 2. Parse & Extract Features
        parse_group = QGroupBox("2. Parse Downloaded Data and Save to Database")
        parse_layout = QVBoxLayout(parse_group)
        self.parse_button = QPushButton("Start Data Parsing")
        self.parse_button.clicked.connect(lambda: self.start_task(
            self._run_data_parsing, 'parse'
        ))
        parse_layout.addWidget(self.parse_button)
        self.parse_progress_bar = QProgressBar()
        self.parse_progress_bar.setTextVisible(True)
        parse_layout.addWidget(self.parse_progress_bar)
        self.parse_status_label = QLabel("Ready")
        parse_layout.addWidget(self.parse_status_label)
        layout.addWidget(parse_group)

        # 3. AI Pre-training
        ai_pretrain_group = QGroupBox("3. Train Foundational AI Model")
        ai_pretrain_layout = QVBoxLayout(ai_pretrain_group)
        self.ai_pretrain_button = QPushButton("Start AI Pre-training")
        self.ai_pretrain_button.clicked.connect(lambda: self.start_task(
            self._run_ai_pretrain, 'ai_train'
        ))
        ai_pretrain_layout.addWidget(self.ai_pretrain_button)
        self.ai_progress_bar = QProgressBar()
        self.ai_progress_bar.setTextVisible(True)
        ai_pretrain_layout.addWidget(self.ai_progress_bar)
        self.ai_status_label = QLabel("Ready")
        ai_pretrain_layout.addWidget(self.ai_status_label)
        layout.addWidget(ai_pretrain_group)

        layout.addStretch(1)

    def setup_data_view_tab(self):
        """Sets up the UI for the View Parsed Data tab."""
        self.data_view_tab = QWidget()
        self.tabs.addTab(self.data_view_tab, "View Parsed Data")
        layout = QVBoxLayout(self.data_view_tab)
        
        control_layout = QHBoxLayout()
        self.load_data_button = QPushButton("Load & Display Data from Database")
        self.load_data_button.clicked.connect(lambda: self.start_task(
            self._load_and_display_data, 'load_display'
        ))
        control_layout.addWidget(self.load_data_button)
        control_layout.addStretch(1)
        layout.addLayout(control_layout)

        self.data_table = QTableWidget()
        layout.addWidget(self.data_table)
        self.table_headers = [
            "Material ID", "Formula", "Space Group", "Lattice A", "Num Sites", "Elements", 
            "Band Gap (eV)", "Formation Energy (eV/atom)", "Magnetization (µB)", "Is Metal"
        ]
        self.data_table.setColumnCount(len(self.table_headers))
        self.data_table.setHorizontalHeaderLabels(self.table_headers)
        self.data_table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.data_table.setEditTriggers(QTableWidget.NoEditTriggers)

    def setup_prediction_tab(self):
        """Sets up the UI for the AI Prediction tab."""
        self.prediction_tab = QWidget()
        self.tabs.addTab(self.prediction_tab, "AI-Powered Prediction")
        main_layout = QHBoxLayout(self.prediction_tab)

        form_group = QGroupBox("Input Hypothetical Material Features")
        form_layout = QFormLayout()
        
        self.prediction_inputs = {
            "lattice_a": QLineEdit("3.5"), "lattice_b": QLineEdit("3.5"), "lattice_c": QLineEdit("3.5"),
            "lattice_alpha": QLineEdit("90"), "lattice_beta": QLineEdit("90"), "lattice_gamma": QLineEdit("90"),
            "volume": QLineEdit("42.875"), "density": QLineEdit("8.9"), "num_sites": QLineEdit("4"),
            "space_group": QLineEdit("Fm-3m")
        }
        for name, widget in self.prediction_inputs.items():
            form_layout.addRow(QLabel(f"{name.replace('_', ' ').title()}:"), widget)
        form_group.setLayout(form_layout)

        results_group = QGroupBox("Prediction")
        results_layout = QVBoxLayout()
        self.predict_button = QPushButton("✨ Predict Properties")
        self.predict_button.clicked.connect(self._run_prediction)
        results_layout.addWidget(self.predict_button)
        
        self.prediction_results_display = QTextEdit()
        self.prediction_results_display.setReadOnly(True)
        self.prediction_results_display.setPlaceholderText("Predicted properties and analysis will appear here...")
        results_layout.addWidget(self.prediction_results_display)
        results_group.setLayout(results_layout)

        main_layout.addWidget(form_group, 1)
        main_layout.addWidget(results_group, 1)

    def setup_fabrication_tab(self):
        """Placeholder for the Fabrication tab."""
        self.fabrication_tab = QWidget()
        self.tabs.addTab(self.fabrication_tab, "Fabrication & Functionalization")
        self.fabrication_tab.setLayout(QVBoxLayout())
        self.fabrication_tab.layout().addWidget(QLabel("Controls for 3D Printing & Vapor Loading (Coming Soon!)"))


    # --- NEW METHOD for analysis ---
    def analyze_predictions(self, prediction_dict):
        """
        Generates a human-readable analysis of the predicted material properties.
        """
        analysis_parts = []

        # 1. Analyze Stability
        formation_energy = prediction_dict.get('formation_energy_per_atom')
        if formation_energy is not None:
            if formation_energy < -0.05: # Use a small threshold
                stability = f"**Likely Stable**: The negative formation energy ({formation_energy:.4f} eV/atom) suggests the material is thermodynamically stable and could likely be synthesized."
            else:
                stability = f"**Likely Unstable**: The positive or near-zero formation energy ({formation_energy:.4f} eV/atom) suggests the material may not be stable."
            analysis_parts.append(stability)

        # 2. Analyze Magnetic Properties
        magnetization = prediction_dict.get('total_magnetization')
        if magnetization is not None:
            if abs(magnetization) > 0.02: # Use a small threshold
                magnetic = f"**Magnetic**: With a total magnetization of {magnetization:.4f} µB, the material is predicted to be magnetic."
            else:
                magnetic = "**Non-Magnetic**: The magnetization is negligible, suggesting a non-magnetic material."
            analysis_parts.append(magnetic)

        # 3. Analyze Electronic Properties
        band_gap = prediction_dict.get('band_gap')
        is_metal_raw = prediction_dict.get('is_metal')
        if is_metal_raw is not None and band_gap is not None:
            if is_metal_raw > 0.5:
                electronic_summary = f"**Metallic Conductor**: The model predicts this is a metal with high confidence ({is_metal_raw:.1%}). The very low predicted band gap ({band_gap:.4f} eV) is consistent with metallic behavior."
            else:
                if band_gap <= 3.0:
                    electronic_summary = f"**Semiconductor**: Predicted to be a semiconductor with a band gap of {band_gap:.4f} eV, making it potentially useful for electronic applications."
                else:
                    electronic_summary = f"**Insulator**: The large predicted band gap ({band_gap:.4f} eV) suggests this material is an electrical insulator."
            analysis_parts.append(electronic_summary)

        # Build the final string using Markdown
        if not analysis_parts:
            return "Could not generate an analysis from the prediction results."

        final_analysis = "### AI-Powered Analysis\n"
        final_analysis += "-------------------------\n\n- "
        final_analysis += "\n- ".join(analysis_parts)
        
        return final_analysis

    # --- MODIFIED METHOD for updating GUI ---
    def _update_prediction_results(self, results):
        """Updates the GUI with the prediction results and analysis."""
        self.log_message(f"Prediction successful: {results}")
        
        # Format the raw numerical output
        result_text = "### Predicted Properties (Raw Values)\n"
        result_text += "-----------------------------------\n"
        for prop, value in results.items():
            if prop == "is_metal":
                 verdict = "Yes" if value > 0.5 else "No"
                 result_text += f"- **Is Metal**: {verdict} (raw value: {value:.4f})\n"
            else:
                 prop_name = prop.replace('_', ' ').title()
                 result_text += f"- **{prop_name}**: {value:.4f}\n"

        # Generate and append the human-readable analysis
        analysis_text = self.analyze_predictions(results)
        result_text += "\n" + analysis_text

        # Use setMarkdown to render the formatted text
        self.prediction_results_display.setMarkdown(result_text)

    # --- Other methods ---
    # (check_data_dir, set_ui_busy_state, _update_progress, start_task, 
    # _run_*, _load_and_display_data, etc. remain the same)
    def _run_prediction(self):
        """Gathers input data and starts the prediction task."""
        try:
            input_data = {}
            for name, widget in self.prediction_inputs.items():
                text_value = widget.text()
                if not text_value:
                    raise ValueError(f"Input for '{name}' cannot be empty.")
                if name != 'space_group':
                    input_data[name] = float(text_value)
                else:
                    input_data[name] = text_value
            
            self.prediction_results_display.setText("Predicting...")
            self.start_task(ae.predict_properties, 'predict', input_data)
        except ValueError as e:
            QMessageBox.critical(self, "Invalid Input", f"Please check your inputs. Error: {e}")
            self.prediction_results_display.setText(f"Error: Invalid input.\n{e}")
        except FileNotFoundError as e:
            QMessageBox.critical(self, "Model Not Found", str(e))
            self.prediction_results_display.setText(f"Error: Model not found. Please train it first.")

    def log_message(self, message):
        """Appends a message to the log text edit in a thread-safe way."""
        self.log_text_edit.append(message)

    def show_error_messagebox(self, title, message):
        """Displays a critical error message box."""
        QMessageBox.critical(self, title, message)
        
    def check_data_dir(self):
        """Ensures the data directory exists, creating it if necessary."""
        if not os.path.exists(da.DATA_DIR):
            os.makedirs(da.DATA_DIR)
            self.log_message(f"Created data directory: {da.DATA_DIR}/")
        else:
            self.log_message(f"Data directory already exists: {da.DATA_DIR}/")

    def set_ui_busy_state(self, is_busy):
        """Disables or enables all task buttons to prevent concurrent operations."""
        self.acquire_button.setEnabled(not is_busy)
        self.parse_button.setEnabled(not is_busy)
        self.ai_pretrain_button.setEnabled(not is_busy)
        self.load_data_button.setEnabled(not is_busy)
        if hasattr(self, 'predict_button'):
            self.predict_button.setEnabled(not is_busy)

    def _update_progress(self, value, maximum, message, mode="determinate"):
        """Updates the appropriate progress bar based on the currently running task."""
        progress_bars = {
            'acquire': self.acquire_progress_bar,
            'parse': self.parse_progress_bar,
            'ai_train': self.ai_progress_bar
        }
        status_labels = {
            'acquire': self.acquire_status_label,
            'parse': self.parse_status_label,
            'ai_train': self.ai_status_label
        }
        
        if self.current_task in progress_bars:
            pb = progress_bars[self.current_task]
            sl = status_labels[self.current_task]
            
            if mode == "indeterminate":
                pb.setRange(0, 0)
            else:
                pb.setRange(0, maximum)
                pb.setValue(value)
            
            pb.setFormat(message)
            sl.setText(message)
        
        self.log_message(message)
        
    def start_task(self, func, task_name, *args):
        self.current_task = task_name
        self.set_ui_busy_state(True)
        
        self.thread = QThread()
        self.worker = Worker(func, *args)
        self.worker.moveToThread(self.thread)
        
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        
        self.worker.log_message.connect(self.log_message)
        self.worker.error_signal.connect(self.show_error_messagebox)
        
        if task_name in ['acquire', 'parse', 'ai_train', 'load_display']:
            self.worker.progress.connect(self._update_progress)
        elif task_name == 'predict':
            self.worker.result_ready.connect(self._update_prediction_results)
            
        self.thread.finished.connect(lambda: self.set_ui_busy_state(False))
        self.thread.start()

    def _run_data_acquisition(self, progress_callback, log_message_callback):
        log_message_callback("Initiating data acquisition...")
        da.download_cif_data(
            progress_callback=progress_callback,
            log_message_callback=log_message_callback
        )
        log_message_callback("Data acquisition process completed.")

    def _run_data_parsing(self, progress_callback, log_message_callback):
        log_message_callback("Initiating data parsing...")
        start_time = time.time()
        try:
            parsed_df = da.parse_and_extract_features(
                progress_callback=progress_callback,
                log_message_callback=log_message_callback
            )
            if not parsed_df.empty:
                self.data_for_ai = parsed_df
                log_message_callback(f"Data parsing completed. Stored {len(self.data_for_ai)} materials for AI training.")
            else:
                log_message_callback("Data parsing completed, but no features were extracted.")
        except Exception as e:
            log_message_callback(f"Data parsing error: {e}")
            raise
        finally:
            elapsed_time = time.time() - start_time
            log_message_callback(f"Parsing finished in {elapsed_time:.2f} seconds.")

    def _run_ai_pretrain(self, progress_callback, log_message_callback):
        log_message_callback("Initiating AI pre-training...")
        start_time = time.time()
        if self.data_for_ai.empty:
            log_message_callback("Error: No data available for AI pre-training. Please run 'Start Data Parsing' first.")
            QMessageBox.warning(self, "No Data", "No parsed data is loaded in memory. Please run data parsing first.")
            return
        try:
            ae.train_ai_model(
                self.data_for_ai,
                progress_callback=progress_callback,
                log_message_callback=log_message_callback
            )
        except Exception as e:
            log_message_callback(f"AI pre-training error: {e}")
            raise
        finally:
            elapsed_time = time.time() - start_time
            log_message_callback(f"AI training finished in {elapsed_time:.2f} seconds.")

    def _load_and_display_data(self, progress_callback, log_message_callback):
        log_message_callback("Loading parsed data from database...")
        db_filepath = da.DATABASE_FILE
        if not os.path.exists(db_filepath):
            log_message_callback(f"Error: Database file not found at {db_filepath}. Please run 'Start Data Parsing' first.")
            return
        conn = sqlite3.connect(db_filepath)
        df = pd.read_sql_query("SELECT * FROM materials", conn)
        conn.close()
        self.data_for_ai = df
        log_message_callback(f"Loaded {len(df)} entries from database into memory.")
        self.data_table.setRowCount(0)
        if df.empty:
            log_message_callback("Database is empty or no materials table found.")
            return
        self.data_table.setRowCount(len(df))
        display_cols = [
            "material_id", "formula", "space_group", "lattice_a", "num_sites", 
            "elements_present", "band_gap", "formation_energy_per_atom", 
            "total_magnetization", "is_metal"
        ]
        for i, row in df.iterrows():
            for j, col_name in enumerate(display_cols):
                val = row.get(col_name)
                display_val = ""
                if pd.notna(val):
                    if isinstance(val, str) and col_name == 'elements_present':
                        try:
                            display_val = ', '.join(json.loads(val))
                        except (json.JSONDecodeError, TypeError):
                            display_val = val
                    elif isinstance(val, float):
                        display_val = f"{val:.4f}"
                    else:
                        display_val = str(val)
                self.data_table.setItem(i, j, QTableWidgetItem(display_val))
            if i % 100 == 0:
                progress_callback(i + 1, len(df), f"Displaying {i+1}/{len(df)} rows...")
        progress_callback(len(df), len(df), f"Successfully displayed {len(df)} entries.")
        log_message_callback("Data display complete.")

def main():
    app = QApplication(sys.argv)
    window = AIFoundryApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()