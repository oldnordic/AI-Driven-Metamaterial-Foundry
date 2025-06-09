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
import joblib
import os

# --- Configuration for AI Engine ---
MODEL_SAVE_PATH = "ai_foundry_model.pth"
PREPROCESSOR_SAVE_PATH = "ai_foundry_preprocessors.joblib"

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

# --- 3. Training Function (MODIFIED) ---
def train_ai_model(dataframe_of_features, progress_callback=None, log_message_callback=None):
    if dataframe_of_features.empty:
        if log_message_callback:
            log_message_callback("No data for AI pre-training. Skipping model training.")
        return

    # Filter out rows where target properties are NaN or None
    target_cols = ["band_gap", "formation_energy_per_atom", "total_magnetization", "is_metal"]
    initial_rows = len(dataframe_of_features)
    dataframe_of_features_cleaned = dataframe_of_features.dropna(subset=target_cols)
    
    if len(dataframe_of_features_cleaned) == 0:
        if log_message_callback:
            log_message_callback("No valid data for AI training after cleaning. Skipping.")
        return

    if log_message_callback:
        log_message_callback(f"Starting AI model training with {len(dataframe_of_features_cleaned)} cleaned material entries...")

    all_feature_cols = [
        "lattice_a", "lattice_b", "lattice_c", "lattice_alpha", "lattice_beta",
        "lattice_gamma", "volume", "density", "num_sites", "space_group"
    ]
    
    available_feature_cols = [col for col in all_feature_cols if col in dataframe_of_features_cleaned.columns]
    available_target_cols = [col for col in target_cols if col in dataframe_of_features_cleaned.columns]

    if not available_feature_cols or not available_target_cols:
        if log_message_callback:
            log_message_callback("Not enough valid features or targets for AI training. Skipping.")
        return

    processed_df = dataframe_of_features_cleaned.copy()
    label_encoders = {}
    scalers = {}

    categorical_cols = ['space_group']
    numerical_cols = [col for col in available_feature_cols if col not in categorical_cols]

    # 1. Handle Categorical Features
    for col in categorical_cols:
        if col in processed_df.columns:
            le = LabelEncoder()
            processed_df[col] = le.fit_transform(processed_df[col].astype(str))
            label_encoders[col] = le

    # 2. Handle Numerical Features
    for col in numerical_cols:
        if col in processed_df.columns:
            processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce').fillna(0)
            scaler = StandardScaler()
            processed_df[col] = scaler.fit_transform(processed_df[[col]]).flatten()
            scalers[col] = scaler

    # 3. Create Tensors
    features_tensor = torch.tensor(processed_df[available_feature_cols].values, dtype=torch.float32)
    
    # --- NEW AND FINAL FIX HERE ---
    # Select the target columns and forcefully convert the entire block to float32
    targets_df = processed_df[available_target_cols].astype(np.float32)
    targets_tensor = torch.tensor(targets_df.values, dtype=torch.float32)
    # --- END OF FIX ---

    # 4. Split data and create DataLoaders
    X_train, X_val, y_train, y_val = train_test_split(
        features_tensor, targets_tensor, test_size=0.2, random_state=42
    )

    train_dataset = MaterialDataset(X_train, y_train)
    val_dataset = MaterialDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # 5. Initialize and train the model
    input_dim = len(available_feature_cols)
    output_dim = len(available_target_cols)
    model = SimpleMaterialPredictor(input_dim, output_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if log_message_callback: log_message_callback(f"Using device: {device}")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 20

    for epoch in range(num_epochs):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, targets).item()
        
        val_epoch_loss = val_loss / len(val_loader)
        if progress_callback:
            progress_callback(epoch + 1, num_epochs, f"Epoch {epoch+1}/{num_epochs}, Val Loss: {val_epoch_loss:.4f}")

    # 6. Save model and pre-processors
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    joblib.dump({
        'scalers': scalers,
        'label_encoders': label_encoders,
        'feature_order': available_feature_cols,
        'target_order': available_target_cols
    }, PREPROCESSOR_SAVE_PATH)

    if log_message_callback:
        log_message_callback(f"AI model saved to {MODEL_SAVE_PATH}")
        log_message_callback(f"Pre-processors saved to {PREPROCESSOR_SAVE_PATH}")

# --- Prediction Function (No changes needed) ---
def predict_properties(input_data_dict, log_message_callback=None):
    if not os.path.exists(MODEL_SAVE_PATH) or not os.path.exists(PREPROCESSOR_SAVE_PATH):
        raise FileNotFoundError("Model or pre-processor file not found. Please train the model first.")

    preprocessors = joblib.load(PREPROCESSOR_SAVE_PATH)
    scalers = preprocessors['scalers']
    label_encoders = preprocessors['label_encoders']
    feature_order = preprocessors['feature_order']
    target_order = preprocessors['target_order']

    input_dim = len(feature_order)
    output_dim = len(target_order)
    model = SimpleMaterialPredictor(input_dim, output_dim)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model.eval()

    input_df = pd.DataFrame([input_data_dict])
    processed_features = []
    for col in feature_order:
        value = input_df.loc[0, col]
        if col in scalers:
            scaled_val = scalers[col].transform(np.array([[value]]))[0, 0]
            processed_features.append(scaled_val)
        elif col in label_encoders:
            try:
                encoded_val = label_encoders[col].transform([str(value)])[0]
                processed_features.append(encoded_val)
            except ValueError:
                if log_message_callback:
                    log_message_callback(f"Warning: Unseen category '{value}' for '{col}'. Using -1 as default.")
                processed_features.append(-1)
        else:
             raise ValueError(f"Feature '{col}' not found in any pre-processor.")
    
    input_tensor = torch.tensor([processed_features], dtype=torch.float32)

    with torch.no_grad():
        predicted_tensor = model(input_tensor)
    
    predictions = predicted_tensor.numpy().flatten()

    result_dict = {target_order[i]: float(predictions[i]) for i in range(len(target_order))}
    
    return result_dict