import os
import librosa
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

DATA_PATH = "dataset"

X = []
y = []

# 🔥 Faster feature extraction
def extract_features(file_path):
    try:
        # Load only first 3 seconds (huge speed boost)
        audio, sr = librosa.load(file_path, sr=16000, duration=3)

        # Reduce MFCC count (20 → 13)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)

        # Use mean + variance (stronger features)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_var = np.var(mfcc, axis=1)

        features = np.concatenate((mfcc_mean, mfcc_var))
        return features

    except Exception as e:
        return None


# ----------------------------
# Load HUMAN files
# ----------------------------
print("Extracting HUMAN features...")
human_files = os.listdir(os.path.join(DATA_PATH, "human"))[:3000]

for file in tqdm(human_files):
    path = os.path.join(DATA_PATH, "human", file)
    features = extract_features(path)
    if features is not None:
        X.append(features)
        y.append(0)  # HUMAN


# ----------------------------
# Load AI files
# ----------------------------
print("Extracting AI features...")
ai_files = os.listdir(os.path.join(DATA_PATH, "ai"))[:3000]

for file in tqdm(ai_files):
    path = os.path.join(DATA_PATH, "ai", file)
    features = extract_features(path)
    if features is not None:
        X.append(features)
        y.append(1)  # AI


X = np.array(X)
y = np.array(y)

print("Dataset shape:", X.shape)

# ----------------------------
# Split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------
# Train
# ----------------------------
print("Training model...")

model = RandomForestClassifier(
    n_estimators=150,   # Reduced from 200 (faster)
    n_jobs=-1,          # 🔥 USE ALL CPU CORES
    random_state=42
)

model.fit(X_train, y_train)

# ----------------------------
# Evaluate
# ----------------------------
y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ----------------------------
# Save
# ----------------------------
joblib.dump(model, "voice_model.pkl")

print("\nModel saved as voice_model.pkl")
