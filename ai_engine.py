import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import json
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.exceptions import NotFittedError

# --- Configuration for AI Engine ---
MODEL_SAVE_PATH = "ai_foundry_model.pth"

# --- 1. Custom Dataset Class ---
class MaterialDataset(Dataset):
    """
    A custom PyTorch Dataset for material features and properties.
    Expects features_tensor and targets_tensor to be pre-processed.
    """
    def __init__(self, features_tensor, targets_tensor):
        self.features = features_tensor
        self.targets = targets_tensor

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

# --- 2. Simple Neural Network Model ---
class SimpleMaterialPredictor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleMaterialPredictor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

# --- 3. Training Function ---
def train_ai_model(dataframe_of_features, progress_callback=None, log_message_callback=None):
    if dataframe_of_features.empty:
        if log_message_callback:
            log_message_callback("No data for AI pre-training. Skipping model training.")
        return

    # Filter out rows where target properties are NaN or None, as these cannot be trained on
    target_cols = ["band_gap", "formation_energy_per_atom", "total_magnetization", "is_metal"]
    initial_rows = len(dataframe_of_features)
    dataframe_of_features_cleaned = dataframe_of_features.dropna(subset=target_cols)
    
    if len(dataframe_of_features_cleaned) == 0:
        if log_message_callback:
            log_message_callback("No valid data (features or targets) after cleaning for AI training. Skipping.")
        return
    elif len(dataframe_of_features_cleaned) < initial_rows:
        if log_message_callback:
            log_message_callback(f"Removed {initial_rows - len(dataframe_of_features_cleaned)} rows with missing target properties for AI training.")

    if log_message_callback:
        log_message_callback(f"Starting AI model training with {len(dataframe_of_features_cleaned)} cleaned material entries...")

    # Define all possible feature columns
    all_feature_cols = [
        "lattice_a", "lattice_b", "lattice_c", "lattice_alpha", "lattice_beta",
        "lattice_gamma", "volume", "density", "num_sites", "space_group",
        "unique_elements_count",
        "elements_present"
    ]
    
    # Filter to only include columns actually present in the DataFrame
    available_feature_cols = [col for col in all_feature_cols if col in dataframe_of_features_cleaned.columns]
    available_target_cols = [col for col in target_cols if col in dataframe_of_features_cleaned.columns]

    if not available_feature_cols or not available_target_cols:
        if log_message_callback:
            log_message_callback("Not enough valid features or targets for AI training after cleaning. Skipping.")
        return

    # --- Preprocessing before splitting ---
    processed_df = dataframe_of_features_cleaned.copy()
    
    label_encoders = {}
    scalers = {}

    # Identify categorical and numerical columns based on availability
    categorical_cols = []
    numerical_cols = []

    for col in available_feature_cols:
        if col in ['space_group', 'elements_present']:
            categorical_cols.append(col)
        else:
            numerical_cols.append(col)

    # 1. Handle Categorical Features
    for col in categorical_cols:
        if col in processed_df.columns:
            if col == 'elements_present':
                # Convert string representation of lists to tuples for LabelEncoder
                processed_df[col] = processed_df[col].apply(
                    lambda x: tuple(json.loads(x)) if isinstance(x, str) else (x if isinstance(x, tuple) else ())
                )
            
            le = LabelEncoder()
            # Fit and transform the entire column. Handle potential NaNs by filling before fit_transform
            # Also ensure the column is treated as object dtype for LabelEncoder if it contains mixed types
            processed_df[col] = processed_df[col].astype(str).fillna('__MISSING__') # Convert all to string for consistent encoding
            processed_df[col] = le.fit_transform(processed_df[col])
            label_encoders[col] = le
        else:
            processed_df[col] = -1 # Default numerical value for missing categorical column


    # 2. Handle Numerical Features
    for col in numerical_cols:
        if col in processed_df.columns:
            processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce').fillna(0).astype(float) # Ensure float
            scaler = StandardScaler()
            processed_df[col] = scaler.fit_transform(processed_df[[col]])
            scalers[col] = scaler
        else:
            processed_df[col] = 0.0 # Add missing numerical feature columns with zeros

    # Ensure target columns are numeric and handle NaNs
    for col in available_target_cols:
        processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce').fillna(0).astype(float) # Ensure float
    
    # --- Final check before tensor conversion ---
    final_features_df = processed_df[available_feature_cols].copy() 
    
    # --- NEW DEBUGGING CODE START ---
    if log_message_callback:
        log_message_callback("--- Final Feature DataFrame dtypes and problematic values ---")
        for col in final_features_df.columns:
            col_dtype = final_features_df[col].dtype
            log_message_callback(f"Column '{col}': dtype = {col_dtype}")
            if col_dtype == object:
                # Find non-numeric/non-boolean items if it's an object dtype
                non_numeric_rows = final_features_df[col].apply(lambda x: not isinstance(x, (int, float, bool))).sum()
                if non_numeric_rows > 0:
                    log_message_callback(f"  --> WARNING: Contains {non_numeric_rows} non-numeric/non-boolean objects. Sample:")
                    log_message_callback(f"    {final_features_df[col][~final_features_df[col].apply(lambda x: isinstance(x, (int, float, bool)))].head()}")
                
                # Attempt to convert again, explicitly to float
                try:
                    final_features_df[col] = final_features_df[col].astype(float)
                    log_message_callback(f"  --> Successfully converted '{col}' to float.")
                except Exception as e:
                    log_message_callback(f"  --> ERROR: Failed to convert '{col}' to float even after previous steps: {e}. Values will be problematic.")
            elif not pd.api.types.is_numeric_dtype(final_features_df[col]) and not pd.api.types.is_bool_dtype(final_features_df[col]):
                log_message_callback(f"  --> WARNING: Column '{col}' is not numeric or boolean. dtype = {col_dtype}")
                
    log_message_callback("--------------------------------------------------")
    # --- NEW DEBUGGING CODE END ---

    features_tensor = torch.tensor(final_features_df.values, dtype=torch.float32)
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