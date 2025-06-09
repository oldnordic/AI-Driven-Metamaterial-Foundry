import os
import requests
from pymatgen.core import Structure
from pymatgen.io.cif import CifParser
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from mp_api.client import MPRester
import pandas as pd
import json
from dotenv import load_dotenv
import sqlite3

load_dotenv()

# --- Configuration ---
DATA_DIR = "material_lattice_data"
PROPERTIES_FILE = os.path.join(DATA_DIR, "material_properties.json")
DATABASE_FILE = os.path.join(DATA_DIR, "materials_foundry.db")
MP_API_KEY = os.getenv("MP_API_KEY")

# --- Data Acquisition Function ---
def download_cif_data(database_url=None, save_path=DATA_DIR, progress_callback=None, log_message_callback=None):
    """
    Downloads CIF files and associated properties from the Materials Project database.
    """
    if not MP_API_KEY:
        error_msg = "Error: Materials Project API key not found. Please ensure MP_API_KEY is set in your .env file."
        if log_message_callback: log_message_callback(error_msg)
        raise ValueError(error_msg)
        
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if log_message_callback: log_message_callback(f"Data directory: {save_path}/")

    if log_message_callback: log_message_callback("Attempting to acquire data from Materials Project using API...")

    downloaded_count = 0
    acquired_properties = {}

    try:
        with MPRester(MP_API_KEY) as mpr:
            target_elements = ["Cu", "Fe", "Si", "O"]
            
            target_properties_fields = [
                "material_id", "structure", "band_gap", "formation_energy_per_atom", 
                "total_magnetization", "is_metal", "formula_pretty"
            ]

            if progress_callback: progress_callback(0, 0, "Querying MP...", mode="indeterminate")

            for element in target_elements:
                if log_message_callback: log_message_callback(f"Querying Materials Project for structures with {element}...")
                data_summaries = mpr.summary.search(elements=[element], fields=target_properties_fields)
                
                for doc in data_summaries:
                    try: 
                        mp_id = str(doc.material_id)
                        props = {field: getattr(doc, field, None) for field in target_properties_fields if field != "structure"}
                        if 'formula_pretty' in props:
                            props['pretty_formula'] = props.pop('formula_pretty')

                        acquired_properties[mp_id] = props

                        if doc.structure:
                            structure_obj = doc.structure
                            cif_string = structure_obj.to(fmt="cif")
                            cif_filepath = os.path.join(save_path, f"{mp_id}.cif")
                            
                            if not os.path.exists(cif_filepath):
                                with open(cif_filepath, "w") as f: f.write(cif_string)
                                downloaded_count += 1
                    except Exception as inner_e:
                        if log_message_callback: log_message_callback(f"Error processing doc {getattr(doc, 'material_id', 'N/A')}: {inner_e}")

            if downloaded_count > 0 or acquired_properties:
                with open(PROPERTIES_FILE, 'w') as f: json.dump(acquired_properties, f, indent=4)
                if log_message_callback: log_message_callback(f"Successfully downloaded {downloaded_count} new CIFs and saved properties for {len(acquired_properties)} materials.")
            else:
                if log_message_callback: log_message_callback("No new files or properties downloaded from Materials Project.")

    except Exception as e:
        if log_message_callback: log_message_callback(f"Error during API call: {e}. Falling back to creating dummy data.")
        
        dummy_cif_content_cu = """
data_Cu_simple
_chemical_name_common 'Copper (FCC)'
_cell_length_a 3.615
_cell_length_b 3.615
_cell_length_c 3.615
_cell_angle_alpha 90.0
_cell_angle_beta 90.0
_cell_angle_gamma 90.0
_symmetry_space_group_name_H-M 'F m -3 m'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Cu1 Cu 0.0 0.0 0.0
"""
        dummy_cif_content_nacl = """
data_NaCl_simple
_chemical_name_common 'Sodium Chloride'
_cell_length_a 5.640
_cell_length_b 5.640
_cell_length_c 5.640
_cell_angle_alpha 90.0
_cell_angle_beta 90.0
_cell_angle_gamma 90.0
_symmetry_space_group_name_H-M 'F m -3 m'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Na1 Na 0.0 0.0 0.0
Cl1 Cl 0.5 0.0 0.0
"""
        dummy_cu_filepath = os.path.join(save_path, "dummy_cu.cif")
        dummy_nacl_filepath = os.path.join(save_path, "dummy_nacl.cif")

        if not os.path.exists(dummy_cu_filepath):
            with open(dummy_cu_filepath, "w") as f: f.write(dummy_cif_content_cu)
            if log_message_callback: log_message_callback("Created fallback dummy CIF (dummy_cu.cif).")
        
        if not os.path.exists(dummy_nacl_filepath):
            with open(dummy_nacl_filepath, "w") as f: f.write(dummy_cif_content_nacl)
            if log_message_callback: log_message_callback("Created fallback dummy CIF (dummy_nacl.cif).")

    if log_message_callback: log_message_callback("Data acquisition process finished.")

# --- Data Parsing Function ---
def parse_and_extract_features(cif_folder=DATA_DIR, properties_file=PROPERTIES_FILE, 
                               database_file=DATABASE_FILE, progress_callback=None, log_message_callback=None):
    material_data = []
    cif_files = [f for f in os.listdir(cif_folder) if f.endswith(".cif")]
    if not cif_files:
        if log_message_callback: log_message_callback("No CIF files found to parse.")
        return pd.DataFrame()
    all_properties = {}
    if os.path.exists(properties_file):
        with open(properties_file, 'r') as f: all_properties = json.load(f)
    if log_message_callback: log_message_callback(f"Parsing {len(cif_files)} CIF files...")
    for i, filename in enumerate(cif_files):
        filepath = os.path.join(cif_folder, filename)
        mp_id_str = os.path.splitext(filename)[0]
        try:
            structure = CifParser(filepath).get_structures(primitive=True)[0]
            sga = SpacegroupAnalyzer(structure)
            features = {
                "material_id": mp_id_str, "pretty_formula": None, "crystal_system": sga.get_crystal_system(),
                "formula": structure.formula.replace(" ", ""), "space_group": sga.get_space_group_info()[0],
                "lattice_a": structure.lattice.a, "lattice_b": structure.lattice.b, "lattice_c": structure.lattice.c,
                "lattice_alpha": structure.lattice.alpha, "lattice_beta": structure.lattice.beta, "lattice_gamma": structure.lattice.gamma,
                "volume": structure.volume, "density": structure.density, "num_sites": structure.num_sites,
                "elements_present": json.dumps(sorted([str(e) for e in structure.composition.elements])),
                "band_gap": None, "formation_energy_per_atom": None, "total_magnetization": None, "is_metal": None,
                "predicted_band_gap": None, "predicted_formation_energy_per_atom": None, 
                "predicted_total_magnetization": None, "predicted_is_metal": None
            }
            if mp_id_str in all_properties:
                for prop_key, prop_value in all_properties[mp_id_str].items():
                    if prop_key in features:
                         features[prop_key] = prop_value
            if not features["pretty_formula"]:
                features["pretty_formula"] = features["formula"]
            material_data.append(features)
        except Exception as e:
            if log_message_callback: log_message_callback(f"Error parsing {filename}: {e}")
        if progress_callback: progress_callback(i + 1, len(cif_files), f"Parsed {i+1}/{len(cif_files)}")
    df = pd.DataFrame(material_data)
    if not df.empty:
        with sqlite3.connect(database_file) as conn:
            df.to_sql('materials', conn, if_exists='replace', index=False)
        if log_message_callback: log_message_callback(f"Saved {len(df)} materials to database.")
    return df