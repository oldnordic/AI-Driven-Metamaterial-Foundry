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
    QTableWidgetItem, QHeaderView
)
from PyQt5.QtCore import QObject, QThread, pyqtSignal, Qt

# Import our custom modules
from ai_foundry import data_acquisition as da
from ai_foundry import ai_engine as ae

# --- Worker for background tasks ---
class Worker(QObject):
    finished = pyqtSignal()
    progress = pyqtSignal(int, int, str, str) # value, max, message, mode
    log_message = pyqtSignal(str)
    error_signal = pyqtSignal(str, str)
    
    def __init__(self, func, *args, **kwargs):
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def run(self):
        try:
            # Pass the progress_callback and log_message signals to the function
            self.kwargs['progress_callback'] = lambda value, maximum, msg, mode="determinate": self.progress.emit(value, maximum, msg, mode)
            self.kwargs['log_message_callback'] = lambda msg: self.log_message.emit(msg) # Separate log message signal
            self.func(*self.args, **self.kwargs)
        except Exception as e:
            self.error_signal.emit("Operation Failed", str(e))
            self.log_message.emit(f"Error in background task: {e}")
        finally:
            self.finished.emit()

# --- Main Application Window ---
class AIFoundryApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI-Driven Metamaterial Foundry Control")
        self.setGeometry(100, 100, 1200, 800) # x, y, width, height

        self.parsed_data_df = pd.DataFrame() # To store the parsed DataFrame

        self.init_ui()
        self.check_data_dir() # This should be safe now

    def init_ui(self):
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        # Tab Widget
        self.tabs = QTabWidget()
        self.main_layout.addWidget(self.tabs)

        # --- Data Acquisition & Pre-processing Tab ---
        self.data_tab = QWidget()
        self.tabs.addTab(self.data_tab, "Data Acquisition & Pre-processing")
        self.setup_data_tab()

        # --- View Parsed Data Tab ---
        self.data_view_tab = QWidget()
        self.tabs.addTab(self.data_view_tab, "View Parsed Data")
        self.setup_data_view_tab()

        # --- Fabrication & Functionalization Tab (Placeholder) ---
        self.fabrication_tab = QWidget()
        self.tabs.addTab(self.fabrication_tab, "Fabrication & Functionalization")
        self.fabrication_tab.setLayout(QVBoxLayout())
        self.fabrication_tab.layout().addWidget(QLabel("Controls for 3D Printing & Vapor Loading (Coming Soon!)"))

        # --- AI Engine & Inverse Design Tab (Placeholder) ---
        self.ai_tab = QWidget()
        self.tabs.addTab(self.ai_tab, "AI Engine & Inverse Design")
        self.ai_tab.setLayout(QVBoxLayout())
        self.ai_tab.layout().addWidget(QLabel("AI Model Training & Design Generation (Coming Soon!)"))

        # --- Status Log ---
        self.log_group = QGroupBox("Application Log")
        self.log_layout = QVBoxLayout(self.log_group)
        self.log_text_edit = QTextEdit()
        self.log_text_edit.setReadOnly(True)
        self.log_layout.addWidget(self.log_text_edit)
        self.main_layout.addWidget(self.log_group)

        self.status_bar = self.statusBar()

    def log_message(self, message):
        # Using a signal to update QTextEdit from any thread
        self.log_text_edit.append(message)

    def show_error_messagebox(self, title, message):
        QMessageBox.critical(self, title, message)

    def setup_data_tab(self):
        layout = QVBoxLayout(self.data_tab)

        # 1. Acquire Raw Data
        acquire_group = QGroupBox("1. Acquire Raw Data")
        acquire_layout = QVBoxLayout(acquire_group)
        acquire_layout.addWidget(QLabel("Database URL (e.g., COD, Materials Project):"))
        self.db_url_entry = QLineEdit("https://www.crystallography.net/cod/")
        acquire_layout.addWidget(self.db_url_entry)
        self.acquire_button = QPushButton("Start Data Acquisition")
        self.acquire_button.clicked.connect(self.start_task_thread(self._run_data_acquisition, 'acquire'))
        acquire_layout.addWidget(self.acquire_button)
        
        self.acquire_progress_bar = QProgressBar()
        self.acquire_progress_bar.setTextVisible(True)
        acquire_layout.addWidget(self.acquire_progress_bar)
        self.acquire_status_label = QLabel("Ready")
        acquire_layout.addWidget(self.acquire_status_label)
        layout.addWidget(acquire_group)

        # 2. Parse & Extract Features
        parse_group = QGroupBox("2. Parse & Extract Features")
        parse_layout = QVBoxLayout(parse_group)
        self.parse_button = QPushButton("Start Data Parsing")
        self.parse_button.clicked.connect(self.start_task_thread(self._run_data_parsing, 'parse'))
        parse_layout.addWidget(self.parse_button)

        self.parse_progress_bar = QProgressBar()
        self.parse_progress_bar.setTextVisible(True)
        parse_layout.addWidget(self.parse_progress_bar)
        self.parse_status_label = QLabel("Ready")
        parse_layout.addWidget(self.parse_status_label)
        layout.addWidget(parse_group)

        # 3. AI Pre-training (Conceptual)
        ai_pretrain_group = QGroupBox("3. AI Pre-training (Conceptual)")
        ai_pretrain_layout = QVBoxLayout(ai_pretrain_group)
        self.ai_pretrain_button = QPushButton("Start AI Pre-training (Conceptual)")
        self.ai_pretrain_button.clicked.connect(self.start_task_thread(self._run_ai_pretrain, 'ai_train'))
        ai_pretrain_layout.addWidget(self.ai_pretrain_button)

        self.ai_progress_bar = QProgressBar()
        self.ai_progress_bar.setTextVisible(True)
        ai_pretrain_layout.addWidget(self.ai_progress_bar)
        self.ai_status_label = QLabel("Ready")
        ai_pretrain_layout.addWidget(self.ai_status_label)
        layout.addWidget(ai_pretrain_group)

        layout.addStretch(1) # Push content to top

    def setup_data_view_tab(self):
        layout = QVBoxLayout(self.data_view_tab)
        
        control_layout = QHBoxLayout()
        self.load_data_button = QPushButton("Load & Display Parsed Data")
        self.load_data_button.clicked.connect(self.start_task_thread(self._load_and_display_data, 'load_display'))
        control_layout.addWidget(self.load_data_button)
        control_layout.addStretch(1) # Push button to left
        layout.addLayout(control_layout)

        # QTableWidget for displaying data
        self.data_table = QTableWidget()
        layout.addWidget(self.data_table)

        # Define columns and headers
        self.table_headers = [
            "Material ID", "Formula", "Space Group", "Lattice A", "Num Sites", "Elements", 
            "Band Gap (eV)", "Formation Energy (eV/atom)", "Magnetization (µB)", "Is Metal"
        ]
        self.data_table.setColumnCount(len(self.table_headers))
        self.data_table.setHorizontalHeaderLabels(self.table_headers)
        self.data_table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive) # Allow resizing
        self.data_table.setEditTriggers(QTableWidget.NoEditTriggers) # Make table read-only

    def check_data_dir(self):
        # Log to the QTextEdit, which should be initialized by now
        if not os.path.exists(da.DATA_DIR):
            os.makedirs(da.DATA_DIR)
            self.log_message(f"Created data directory: {da.DATA_DIR}/")
        else:
            self.log_message(f"Data directory already exists: {da.DATA_DIR}/")

    def set_ui_busy_state(self, task_name, is_busy):
        # General method to control button and progress bar states
        buttons = {
            'acquire': self.acquire_button,
            'parse': self.parse_button,
            'ai_train': self.ai_pretrain_button,
            'load_display': self.load_data_button
        }
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

        if is_busy:
            # Disable all buttons
            for btn in buttons.values():
                btn.setEnabled(False)
            # Start indeterminate progress for specific task
            if task_name in progress_bars:
                progress_bars[task_name].setRange(0,0) # Indeterminate mode
                progress_bars[task_name].setValue(0)
                status_labels[task_name].setText("Running...")
        else:
            # Enable all buttons
            for btn in buttons.values():
                btn.setEnabled(True)
            # Reset specific task progress bar
            if task_name in progress_bars:
                progress_bars[task_name].setRange(0,100) # Determinate mode
                progress_bars[task_name].setValue(0)
                status_labels[task_name].setText("Ready")
            
    def _update_progress(self, value, maximum, message, mode="determinate"):
        # Update progress bar and label based on which task is running
        if self.current_task == 'acquire':
            pb = self.acquire_progress_bar
            sl = self.acquire_status_label
        elif self.current_task == 'parse':
            pb = self.parse_progress_bar
            sl = self.parse_status_label
        elif self.current_task == 'ai_train':
            pb = self.ai_progress_bar
            sl = self.ai_status_label
        else: # For load_display, it's quick usually, no dedicated bar
            return 
        
        pb.setRange(0, maximum)
        pb.setValue(value)
        pb.setFormat(message) # Display message directly on the bar
        sl.setText(message)
        self.log_message(message) # Also log to main log

    # --- Threading Management ---
    def start_task_thread(self, func, task_name):
        def wrapper():
            self.current_task = task_name # Store current task name
            self.set_ui_busy_state(task_name, True)
            
            self.thread = QThread()
            self.worker = Worker(func, self.db_url_entry.text() if task_name == 'acquire' else self.parsed_data_df) # Pass relevant data
            
            # Connect signals
            self.worker.moveToThread(self.thread)
            self.thread.started.connect(self.worker.run)
            self.worker.finished.connect(self.thread.quit)
            self.worker.finished.connect(self.worker.deleteLater)
            self.thread.finished.connect(self.thread.deleteLater)
            self.worker.progress.connect(self._update_progress)
            self.worker.log_message.connect(self.log_message) # Connect worker's log signal
            self.worker.error_signal.connect(self.show_error_messagebox)
            
            self.thread.finished.connect(lambda: self.set_ui_busy_state(task_name, False))
            self.thread.start()
        return wrapper

    # --- Background Task Functions (called by Worker) ---
    def _run_data_acquisition(self, database_url, progress_callback, log_message_callback):
        log_message_callback("Initiating data acquisition...")
        da.download_cif_data(
            database_url=database_url, 
            progress_callback=progress_callback
        )
        log_message_callback("Data acquisition process completed.")

    def _run_data_parsing(self, dummy_arg, progress_callback, log_message_callback): # dummy_arg is parsed_data_df from start_task_thread, not used here
        log_message_callback("Initiating data parsing...")
        start_time = time.time()
        try:
            # Ensure the DataFrame is loaded or parsed to self.parsed_data_df if needed
            # For data parsing, we need to pass a valid database_file path, not parsed_data_df
            # Re-read or ensure data is available before parsing. This needs re-thinking based on current flow
            # Let's read from database directly if parsed_data_df is not set or empty (robustness)
            if self.parsed_data_df.empty: # Only load from DB if it's not already loaded from a fresh parse
                 conn = sqlite3.connect(da.DATABASE_FILE)
                 self.parsed_data_df = pd.read_sql_query("SELECT * FROM materials", conn)
                 conn.close()

            # Now, truly run the parsing and save to DB
            self.parsed_data_df = da.parse_and_extract_features(
                cif_folder=da.DATA_DIR, 
                properties_file=da.PROPERTIES_FILE, 
                database_file=da.DATABASE_FILE,
                progress_callback=progress_callback # Pass correct progress callback
            )
            if not self.parsed_data_df.empty:
                log_message_callback(f"Data parsing completed. Extracted {len(self.parsed_data_df)} material features and saved to database.")
            else:
                log_message_callback("Data parsing completed, but no features were extracted. Check logs for errors.")
        except Exception as e:
            log_message_callback(f"Data parsing error: {e}")
            raise # Re-raise to be caught by worker's error_signal
        finally:
            elapsed_time = time.time() - start_time
            log_message_callback(f"Parsing finished in {elapsed_time:.2f} seconds.")


    def _run_ai_pretrain(self, dataframe, progress_callback, log_message_callback):
        log_message_callback("Initiating conceptual AI pre-training...")
        start_time = time.time()
        
        # Ensure data is loaded to 'dataframe' parameter
        if dataframe.empty:
            log_message_callback("Error: No parsed data available for AI pre-training. Please run 'Start Data Parsing' first.")
            return # Exit early

        try:
            ae.train_ai_model(
                dataframe, # Pass the DataFrame received
                progress_callback=progress_callback, # Pass correct progress callback
                log_message_callback=log_message_callback # Pass log message callback
            )
            log_message_callback("Conceptual AI pre-training process completed.")
        except Exception as e:
            log_message_callback(f"AI pre-training error: {e}")
            raise # Re-raise to be caught by worker's error_signal
        finally:
            elapsed_time = time.time() - start_time
            log_message_callback(f"AI training finished in {elapsed_time:.2f} seconds.")


    def _load_and_display_data(self, dummy_arg, progress_callback, log_message_callback): # dummy_arg not used
        log_message_callback("Loading and displaying parsed data from database...")
        start_time = time.time()
        try:
            db_filepath = da.DATABASE_FILE
            if not os.path.exists(db_filepath):
                log_message_callback(f"Error: Database file not found at {db_filepath}. Please run 'Start Data Parsing' first.")
                QMessageBox.critical(self, "Error", "Database file not found. Please parse data first.") # Directly use QMessageBox
                return

            conn = sqlite3.connect(db_filepath)
            df = pd.read_sql_query("SELECT * FROM materials", conn)
            conn.close()

            if df.empty:
                log_message_callback("Database is empty or no materials table found.")
                QMessageBox.information(self, "Info", "Database is empty or no materials table found.")
                return

            # Clear existing data in QTableWidget (must be done on main thread)
            self.data_table.setRowCount(0) # Clear existing rows
            
            # Set up columns if not already set (e.g., on first load)
            self.data_table.setColumnCount(len(self.table_headers))
            self.data_table.setHorizontalHeaderLabels(self.table_headers)

            # Populate QTableWidget with new data
            display_cols_df = [
                "material_id", "formula", "space_group", "lattice_a", 
                "num_sites", "elements_present", "band_gap", "formation_energy_per_atom", 
                "total_magnetization", "is_metal"
            ]
            
            self.data_table.setRowCount(len(df)) # Set row count for new data
            
            for i, (_, row) in enumerate(df.iterrows()):
                for j, col_name in enumerate(display_cols_df):
                    val = row.get(col_name, None) 
                    
                    display_val = "N/A"
                    if pd.notna(val) and val is not None:
                        if col_name == 'elements_present' and isinstance(val, str):
                            try:
                                elements_list = json.loads(val) 
                                display_val = ', '.join(elements_list)
                            except (json.JSONDecodeError, TypeError):
                                display_val = str(val) # Fallback to raw string
                        elif isinstance(val, float): 
                            display_val = f"{val:.3f}"
                        elif isinstance(val, bool):
                            display_val = str(val)
                        else:
                            display_val = str(val)
                    
                    self.data_table.setItem(i, j, QTableWidgetItem(display_val))
                
                # Update progress visually in table loading as well, though usually fast
                if i % 100 == 0: # Update every 100 rows to avoid too many GUI updates
                    progress_callback(i + 1, len(df), f"Displaying {i+1}/{len(df)} rows...")

            progress_callback(len(df), len(df), f"Successfully loaded and displayed {len(df)} entries from database.") # Final progress update
            log_message_callback(f"Successfully loaded and displayed {len(df)} entries from database.")

        except Exception as e:
            log_message_callback(f"Error loading/displaying data from database: {e}")
            raise # Re-raise to be caught by worker's error_signal
        finally:
            elapsed_time = time.time() - start_time
            log_message_callback(f"Display finished in {elapsed_time:.2f} seconds.")
            # QProgressBar for display is not directly tied to self.current_task for a bar.
            # We'll just reset the main button state.

# --- Main Application Execution ---
def main():
    app = QApplication(sys.argv)
    window = AIFoundryApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()