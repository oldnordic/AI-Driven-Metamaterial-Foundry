# AI-Driven Metamaterial Foundry: A Blueprint for Accelerated Materials Innovation

## Overview
This project represents the foundational steps towards an **AI-Driven Metamaterial Foundry**, a revolutionary system aiming to transform materials discovery. It combines advanced computational tools with experimental data pipelines to design and create novel materials with precisely engineered properties, atom by atom.

The core vision is to move from laborious, serendipitous discovery to an on-demand, AI-guided paradigm for creating materials with unprecedented functionalities. This project focuses on the initial pillar: **Data Acquisition and Pre-processing** for AI training.

## Key Features Implemented (Initial Phase)

-   **Modular Python Application:** Built with a clean, modular structure.
-   **Graphical User Interface (GUI):** Developed using PyQt5 for interactive control.
-   **Multi-threaded Processing:** Ensures a responsive GUI during long-running tasks.
-   **Materials Data Acquisition:** Connects to the **Materials Project API** to download known crystal structures (CIF files) and their associated properties (e.g., band gap, formation energy, magnetization).
-   **Robust Data Parsing:** Processes downloaded CIF files, extracts key crystallographic features (lattice parameters, space group, elemental composition, etc.), and combines them with material properties.
-   **SQLite Database Backend:** Stores all parsed and processed material data in a local SQLite database for efficient management and querying, replacing flat CSV files.
-   **Conceptual AI Pre-training:** Demonstrates a foundational AI training loop using PyTorch, leveraging local AMD GPUs (via ROCm) to learn relationships between material structures and properties.
-   **Progress Indicators:** Provides visual progress bars and status updates for long-running processes.

## Vision of the Full Foundry (Beyond this Project's Scope)
This project is the first step towards a complete AI-Driven Metamaterial Foundry, envisioned around four interconnected pillars:
1.  **Lattice Format:** Advanced nanoscale 3D printing of programmable lattice scaffolds.
2.  **Vapor Loading:** Precision atomic-level functionalization of scaffolds with desired elements.
3.  **Frequency Stimulation:** Quantum-state probing of materials under cryogenic conditions.
4.  **AI Predictive Engine:** The intelligent core orchestrating the entire closed-loop discovery and inverse design process.

## Installation and Setup

### Prerequisites
-   Linux operating system (e.g., CachyOS / Arch Linux, Ubuntu)
-   Python 3.10, 3.11, 3.12, or 3.13 (Python 3.13 is confirmed to work with recent PyTorch ROCm builds).
-   `pip` (Python package installer)
-   `git` (Version control system)
-   **AMD GPU** (e.g., Radeon RX 7000 series / RDNA 3 architecture) for AI acceleration.
-   **ROCm Installation:** Ensure [AMD ROCm platform](https://rocm.docs.amd.com/en/latest/deploy/linux/index.html) is correctly installed and configured for your specific GPU and Linux distribution. This is critical for PyTorch GPU acceleration.

### Project Setup

1.  **Clone the Repository:**
    ```bash
    git clone git@github.com:YOUR_USERNAME/AI-Driven-Metamaterial-Foundry.git
    cd AI-Driven-Metamaterial-Foundry
    ```
    (Replace `YOUR_USERNAME` and `AI-Driven-Metamaterial-Foundry` with your actual GitHub details).

2.  **Create and Activate Virtual Environment:**
    It is highly recommended to use a virtual environment to manage dependencies.
    ```bash
    python3.13 -m venv .venv # Or python3.11 -m venv .venv if you used Python 3.11
    source .venv/bin/activate
    ```

3.  **Install Python Dependencies:**
    ```bash
    pip install PyQt5 python-dotenv requests pymatgen pandas mp-api scikit-learn
    # Install PyTorch with ROCm (Crucial: Check PyTorch website for exact command for your ROCm version)
    # Example for ROCm 6.3 on Python 3.13:
    pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/rocm6.3](https://download.pytorch.org/whl/rocm6.3)
    ```

4.  **Set Up Materials Project API Key:**
    -   Go to [materialsproject.org](https://materialsproject.org/) and register for a free account.
    -   Find your API key in your account dashboard.
    -   Create a file named `.env` in the root of your `AI-Driven-Metamaterial-Foundry` directory (the same directory as `main_gui.py`):
        ```
        MP_API_KEY="YOUR_MATERIALS_PROJECT_API_KEY_HERE"
        ```
        **Remember to replace `YOUR_MATERIALS_PROJECT_API_KEY_HERE` with your actual key, keeping the quotes.**

## Running the Application

1.  **Activate your virtual environment (if not already active):**
    ```bash
    cd ~/AI-Driven-Metamaterial-Foundry # Navigate to your project root
    source .venv/bin/activate
    ```

2.  **Launch the GUI:**
    ```bash
    python -m ai_foundry.main_gui
    ```

### Usage Steps (within the GUI)
1.  **Data Acquisition & Pre-processing Tab:**
    -   Click "Start Data Acquisition" to download CIF files and properties from Materials Project.
    -   Click "Start Data Parsing" to process the downloaded data and save it to the local SQLite database.
    -   Click "Start AI Pre-training (Conceptual)" to train the foundational AI model on your GPU.
2.  **View Parsed Data Tab:**
    -   Click "Load & Display Parsed Data" to view the contents of your SQLite database in a table.

## Contributing
This project is open-source and contributions are welcome! Feel free to open issues or pull requests.

## License
This project is licensed under the MIT License.

## Contact
[Luiz Spies/https://github.com/oldnordic]
