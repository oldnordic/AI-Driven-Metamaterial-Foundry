import os
import requests
from pymatgen.core import Structure
from pymatgen.io.cif import CifParser
from mp_api.client import MPRester
import pandas as pd
import json
from dotenv import load_dotenv
import sqlite3

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
DATA_DIR = "material_lattice_data"
PROPERTIES_FILE = os.path.join(DATA_DIR, "material_properties.json")
DATABASE_FILE = os.path.join(DATA_DIR, "materials_foundry.db")

# Your Materials Project API Key - now loaded from .env
MP_API_KEY = os.getenv("MP_API_KEY")

# --- Data Acquisition Function ---
# Modified to accept log_message_callback
def download_cif_data(database_url=None, save_path=DATA_DIR, progress_callback=None, log_message_callback=None):
    """
    Downloads CIF files and associated properties from the Materials Project database.

    Args:
        database_url (str, optional): Not directly used for MPRester, but kept for signature consistency.
        save_path (str): Directory to save the data.
        progress_callback (callable, optional): A function to call with (value, max, message, mode).
        log_message_callback (callable, optional): A function to call with general log messages.
    """
    if not MP_API_KEY:
        error_msg = "Error: Materials Project API key not found. Please ensure MP_API_KEY is set in your .env file or environment variables."
        if log_message_callback:
            log_message_callback(error_msg)
        raise ValueError(error_msg)
        
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        if log_message_callback:
            log_message_callback(f"Created data directory: {save_path}/")
    else:
        if log_message_callback:
            log_message_callback(f"Data directory already exists: {save_path}/")

    if log_message_callback:
        log_message_callback(f"Attempting to acquire data from Materials Project using API...")

    downloaded_count = 0
    acquired_properties = {}

    try:
        mpr = MPRester(MP_API_KEY)
        
        target_elements = ["Cu", "Fe"]
        target_properties_fields = [
            "material_id", 
            "structure", 
            "band_gap", 
            "formation_energy_per_atom", 
            "total_magnetization",
            "is_metal"
        ]

        # Materials Project API does not give total count for summary.search, so progress bar will be indeterminate
        if progress_callback:
            progress_callback(0, 0, "Querying MP...", mode="indeterminate") # Start indeterminate mode

        total_materials_queried = 0
        for element in target_elements:
            if log_message_callback:
                log_message_callback(f"Querying Materials Project for structures & properties with {element} (default API limit)...")
            
            data_summaries = mpr.summary.search(elements=[element], fields=target_properties_fields) 
            
            # Loop through summaries to download CIFs and collect properties
            for doc_idx, doc in enumerate(data_summaries): # Added doc_idx for potential progress if needed
                total_materials_queried += 1
                try: 
                    mp_id = doc.material_id
                    
                    props = {field: getattr(doc, field, None) for field in target_properties_fields if field != "structure"}
                    acquired_properties[mp_id] = props

                    if doc.structure:
                        if isinstance(doc.structure, Structure):
                            structure_obj = doc.structure
                        else:
                            structure_obj = Structure.from_dict(doc.structure)
                        
                        cif_string = structure_obj.to(fmt="cif")
                        cif_filepath = os.path.join(save_path, f"{mp_id}.cif")
                        
                        if not os.path.exists(cif_filepath):
                            with open(cif_filepath, "w") as f:
                                f.write(cif_string)
                            downloaded_count += 1
                            if log_message_callback:
                                log_message_callback(f"Downloaded and saved CIF for: {mp_id}")
                        else:
                            if log_message_callback:
                                log_message_callback(f"File already exists: {mp_id}.cif (skipped)")
                    else:
                        if log_message_callback:
                            log_message_callback(f"Skipping {mp_id}: No structure data available.")
                except Exception as inner_e:
                    if log_message_callback:
                        log_message_callback(f"Error processing doc {mp_id}: {inner_e}")
                    print(f"Error processing doc {mp_id}: {inner_e}") # Keep console print for debugging
            
        if downloaded_count == 0 and not acquired_properties:
            if log_message_callback:
                log_message_callback("No new files or properties downloaded from Materials Project (check existing files or query results).")
        else:
            with open(PROPERTIES_FILE, 'w') as f:
                json.dump(acquired_properties, f, indent=4)
            if log_message_callback:
                log_message_callback(f"Successfully downloaded {downloaded_count} new CIF files and saved properties for {len(acquired_properties)} materials.")
            print(f"Data acquisition from Materials Project complete. Total new files: {downloaded_count}, Properties saved: {len(acquired_properties)}")

    except Exception as e:
        if log_message_callback:
            log_message_callback(f"Error during Materials Project API call: {e}. Please check your API key, internet, or query parameters.")
        print(f"Error during Materials Project API call: {e}. Falling back to dummy data.")
        
        # --- Corrected & Simplified Dummy CIFs ---
        dummy_cif_content_cu = """
data_Cu_simple
_chemical_name_common                   'Copper (Simple FCC)'
_cell_length_a                          3.615
_cell_length_b                          3.615
_cell_length_c                          3.615
_cell_angle_alpha                       90.000
_cell_angle_beta                        90.000
_cell_angle_gamma                       90.000
_symmetry_space_group_name_H-M          'F m -3 m'
_symmetry_Int_Tables_number             225
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
Cu1 Cu 0.0 0.0 0.0 1.0
"""
        dummy_cif_content_nacl = """
data_NaCl_simple
_chemical_name_common 'Sodium Chloride (Simple Cubic)'
_cell_length_a 5.640
_cell_length_b 5.640
_cell_length_c 5.640
_cell_angle_alpha 90.00
_cell_angle_beta 90.00
_cell_angle_gamma 90.00
_symmetry_space_group_name_H-M 'F m -3 m'
_symmetry_Int_Tables_number 225
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
Na1 Na 0.0 0.0 0.0 1.0
Cl1 Cl 0.5 0.0 0.0 1.0
"""
        dummy_cu_filepath = os.path.join(save_path, "dummy_cu_perfected.cif")
        dummy_nacl_filepath = os.path.join(save_path, "dummy_nacl_perfected.cif")

        if not os.path.exists(dummy_cu_filepath):
            with open(dummy_cu_filepath, "w") as f:
                f.write(dummy_cif_content_cu)
            if log_message_callback:
                log_message_callback("Created fallback dummy CIF (dummy_cu_perfected.cif).")
        else:
            if log_message_callback:
                log_message_callback("Fallback dummy CIF (dummy_cu_perfected.cif) exists.")
        
        if not os.path.exists(dummy_nacl_filepath):
            with open(dummy_nacl_filepath, "w") as f:
                f.write(dummy_cif_content_nacl)
            if log_message_callback:
                log_message_callback("Created fallback dummy CIF (dummy_nacl_perfected.cif).")
        else:
            if log_message_callback:
                log_message_callback("Fallback dummy CIF (dummy_nacl_perfected.cif) exists.")

    if log_message_callback:
        log_message_callback("Data acquisition process finished.")


# --- Data Parsing Function (MODIFIED for progress and log callbacks) ---
def parse_and_extract_features(cif_folder=DATA_DIR, properties_file=PROPERTIES_FILE, 
                               database_file=DATABASE_FILE, progress_callback=None, log_message_callback=None):
    """
    Parses CIF files, extracts relevant features for AI, combines with downloaded properties,
    and saves to an SQLite database.

    Args:
        cif_folder (str): Directory containing CIF files.
        properties_file (str): Path to the JSON file containing material properties.
        database_file (str): Path to the SQLite database file.
        progress_callback (callable, optional): A function to call with (value, max, message).
        log_message_callback (callable, optional): A function to call with general log messages.
    """
    material_data = []
    cif_files = [f for f in os.listdir(cif_folder) if f.endswith(".cif")]
    total_files = len(cif_files)

    if total_files == 0:
        if progress_callback:
            progress_callback(0, 1, "No CIF files found to parse. Ensure data acquisition ran.")
        if log_message_callback:
            log_message_callback("No CIF files found to parse.")
        return pd.DataFrame()

    # Load properties
    all_properties = {}
    if os.path.exists(properties_file):
        try:
            with open(properties_file, 'r') as f:
                all_properties = json.load(f)
            if log_message_callback:
                log_message_callback(f"Loaded properties for {len(all_properties)} materials from {os.path.basename(properties_file)}.")
        except Exception as e:
            if log_message_callback:
                log_message_callback(f"Error loading properties from {os.path.basename(properties_file)}: {e}. Proceeding without properties.")
            print(f"Error loading properties: {e}")

    if log_message_callback:
        log_message_callback(f"Starting parsing of {total_files} CIF files...")

    for i, filename in enumerate(cif_files):
        filepath = os.path.join(cif_folder, filename)
        mp_id = os.path.splitext(os.path.basename(filename))[0]

        try:
            parser = CifParser(filepath)
            structures = parser.parse_structures(primitive=True) 
            if not structures:
                raise ValueError("No valid structure parsed from CIF file.")
            structure = structures[0]

            features = {
                "material_id": mp_id,
                "formula": structure.formula.replace(" ", ""),
                "space_group": structure.get_space_group_info()[0],
                "lattice_a": structure.lattice.a,
                "lattice_b": structure.lattice.b,
                "lattice_c": structure.lattice.c,
                "lattice_alpha": structure.lattice.alpha,
                "lattice_beta": structure.lattice.beta,
                "lattice_gamma": structure.lattice.gamma,
                "volume": structure.volume,
                "density": structure.density,
                "num_sites": structure.num_sites,
                "elements_present": json.dumps(sorted([str(e) for e in structure.composition.elements])),
                "composition_dict": json.dumps(structure.composition.as_dict()),
                "band_gap": None,
                "formation_energy_per_atom": None,
                "total_magnetization": None,
                "is_metal": None
            }
            
            if mp_id in all_properties:
                for prop_key, prop_value in all_properties[mp_id].items():
                    if prop_key in features:
                         features[prop_key] = prop_value
            else:
                if log_message_callback:
                    log_message_callback(f"Warning: No properties found for {mp_id} in JSON. Proceeding without them.")

            material_data.append(features)
            if progress_callback:
                progress_callback(i + 1, total_files, f"Parsed {i+1}/{total_files}: {filename}")
        except Exception as e:
            if log_message_callback:
                log_message_callback(f"Error parsing {filename}: {e}")
            print(f"Error parsing {filename}: {e}") # Keep console print for debugging

    df = pd.DataFrame(material_data)
    
    if not df.empty:
        try:
            conn = sqlite3.connect(database_file)
            df.to_sql('materials', conn, if_exists='replace', index=False)
            conn.close()
            if log_message_callback:
                log_message_callback(f"Successfully saved {len(df)} materials to {os.path.basename(database_file)}.")
        except Exception as e:
            if log_message_callback:
                log_message_callback(f"Error saving to database: {e}")
            print(f"Error saving to database: {e}")

    if log_message_callback:
        log_message_callback(f"Finished parsing. Extracted features for {len(df)} materials.")
    print("Data parsing complete.")
    return df

# --- Conceptual AI Pre-training Function (MODIFIED for progress and log callbacks) ---
def pre_train_ai_model(dataframe_of_features, progress_callback=None, log_message_callback=None):
    """
    This is a conceptual step for the AI model's initial training.
    In a real scenario, this would involve defining, training, and saving
    a machine learning model based on the extracted features and
    corresponding known material properties (which would need to be loaded).
    """
    if dataframe_of_features.empty:
        if progress_callback:
            progress_callback(0, 1, "No data to pre-train on. AI pre-training skipped.")
        if log_message_callback:
            log_message_callback("No data for AI pre-training.")
        return

    # Filter out rows where target properties are NaN or None, as these cannot be trained on
    target_cols = ["band_gap", "formation_energy_per_atom", "total_magnetization", "is_metal"]
    initial_rows = len(dataframe_of_features)
    dataframe_of_features_cleaned = dataframe_of_features.dropna(subset=target_cols)
    
    if len(dataframe_of_features_cleaned) == 0:
        if progress_callback:
            progress_callback(0, 1, "No valid data (features or targets) after cleaning for AI training. Skipping.")
        if log_message_callback:
            log_message_callback("No valid data after cleaning for AI training. Skipping.")
        return
    elif len(dataframe_of_features_cleaned) < initial_rows:
        if log_message_callback:
            log_message_callback(f"Removed {initial_rows - len(dataframe_of_features_cleaned)} rows with missing target properties for AI training.")


    if log_message_callback:
        log_message_callback(f"Starting AI model training with {len(dataframe_of_features_cleaned)} cleaned material entries...")

    all_feature_cols = [
        "lattice_a", "lattice_b", "lattice_c", "lattice_alpha", "lattice_beta",
        "lattice_gamma", "volume", "density", "num_sites", "space_group",
        "unique_elements_count",
        "elements_present"
    ]
    available_feature_cols = [col for col in all_feature_cols if col in dataframe_of_features_cleaned.columns]

    available_target_cols = [col for col in target_cols if col in dataframe_of_features_cleaned.columns]

    if not available_feature_cols or not available_target_cols:
        if progress_callback:
            progress_callback(0, 1, "Not enough valid features or targets for AI training after cleaning. Skipping.")
        if log_message_callback:
            log_message_callback("Not enough valid features or targets for AI training after cleaning. Skipping.")
        return

    # --- Preprocessing before splitting (CRITICAL for LabelEncoder) ---
    processed_df = dataframe_of_features_cleaned.copy()
    
    label_encoders = {}
    scalers = {}

    if 'elements_present' in processed_df.columns:
        processed_df['elements_present'] = processed_df['elements_present'].apply(
            lambda x: tuple(json.loads(x)) if isinstance(x, str) else tuple()
        )
    
    for col in ['space_group', 'elements_present']:
        if col in processed_df.columns and (pd.api.types.is_string_dtype(processed_df[col]) or pd.api.types.is_object_dtype(processed_df[col])):
            le = LabelEncoder()
            processed_df[col] = le.fit_transform(processed_df[col])
            label_encoders[col] = le
        else:
            if col in processed_df.columns:
                processed_df[col] = processed_df[col].fillna(0)


    numerical_features = [col for col in available_feature_cols if col not in ['space_group', 'elements_present']]
    for col in numerical_features:
        if col in processed_df.columns:
            processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce').fillna(0)
            scaler = StandardScaler()
            processed_df[col] = scaler.fit_transform(processed_df[[col]])
            scalers[col] = scaler
        else:
            processed_df[col] = 0.0

    for col in available_target_cols:
        processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce').fillna(0)
    
    features_tensor = torch.tensor(processed_df[available_feature_cols].values, dtype=torch.float32)
    targets_tensor = torch.tensor(processed_df[available_target_cols].values, dtype=torch.float32)

    # --- Data Splitting and DataLoader Creation ---
    X_train, X_val, y_train, y_val = train_test_split(
        features_tensor, targets_tensor, test_size=0.2, random_state=42
    )

    train_dataset = MaterialDataset(X_train, y_train)
    val_dataset = MaterialDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # --- Model Initialization ---
    input_dim = len(available_feature_cols)
    output_dim = len(available_target_cols)
    model = SimpleMaterialPredictor(input_dim, output_dim)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if log_message_callback:
        log_message_callback(f"Using device: {device}")

    # --- Loss Function and Optimizer ---
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # --- Training Loop ---
    num_epochs = 20
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(train_dataset)

        # --- Validation Loop ---
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_running_loss += loss.item() * inputs.size(0)
        
        val_epoch_loss = val_running_loss / len(val_dataset)

        if progress_callback:
            progress_callback(epoch + 1, num_epochs, f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_epoch_loss:.4f}")
        if log_message_callback:
            log_message_callback(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_epoch_loss:.4f}")

    # --- Save the trained model ---
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    if log_message_callback:
        log_message_callback(f"AI model saved to {MODEL_SAVE_PATH}")

    if log_message_callback:
        log_message_callback("AI model training complete. Model's foundational understanding established.")