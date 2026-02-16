import os
import librosa
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import joblib

DATA_PATH = "dataset"
SAMPLE_RATE = 16000
DURATION = 3

X = []
y = []

# -----------------------------------------
# Feature Extraction (More Robust)
# -----------------------------------------
def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)

        if len(audio) < 1000:
            return None

        # MFCC
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_var = np.var(mfcc, axis=1)

        # Spectral centroid
        spectral_centroid = np.mean(
            librosa.feature.spectral_centroid(y=audio, sr=sr)
        )

        # Zero crossing rate
        zcr = np.mean(librosa.feature.zero_crossing_rate(audio))

        # Spectral bandwidth
        spectral_bandwidth = np.mean(
            librosa.feature.spectral_bandwidth(y=audio, sr=sr)
        )

        features = np.concatenate(
            (
                mfcc_mean,
                mfcc_var,
                [spectral_centroid],
                [zcr],
                [spectral_bandwidth],
            )
        )

        return features

    except Exception:
        return None


# -----------------------------------------
# Load HUMAN
# -----------------------------------------
print("Extracting HUMAN features...")
human_path = os.path.join(DATA_PATH, "human")
human_files = os.listdir(human_path)[:3000]

for file in tqdm(human_files):
    path = os.path.join(human_path, file)
    features = extract_features(path)
    if features is not None:
        X.append(features)
        y.append(0)

# -----------------------------------------
# Load AI
# -----------------------------------------
print("Extracting AI features...")
ai_path = os.path.join(DATA_PATH, "ai")
ai_files = os.listdir(ai_path)[:3000]

for file in tqdm(ai_files):
    path = os.path.join(ai_path, file)
    features = extract_features(path)
    if features is not None:
        X.append(features)
        y.append(1)

X = np.array(X)
y = np.array(y)

print("Dataset shape:", X.shape)

# -----------------------------------------
# Stratified Split
# -----------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------------------------
# Pipeline (Scaling + Model)
# -----------------------------------------
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        n_jobs=-1,
        random_state=42
    ))
])

print("Training model...")
pipeline.fit(X_train, y_train)

# -----------------------------------------
# Cross Validation (Stability Check)
# -----------------------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(pipeline, X, y, cv=cv)

print("\nCross-validation scores:", cv_scores)
print("Mean CV Accuracy:", np.mean(cv_scores))

# -----------------------------------------
# Evaluation
# -----------------------------------------
y_pred = pipeline.predict(X_test)

print("\nTest Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# -----------------------------------------
# Save
# -----------------------------------------
joblib.dump(pipeline, "voice_model.pkl")

print("\nModel saved as voice_model.pkl")
