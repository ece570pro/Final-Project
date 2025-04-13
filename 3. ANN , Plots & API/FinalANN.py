import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import os


# Load Data
df = pd.read_csv("data.csv")
print("Data columns:", df.columns)

# Extract input features
X = df[['vf', 'width', 'thickness', 'p', 't', 'h']].values

# Extract Targets labels for k11 and k33
y_k11 = df['k11'].values
y_k33 = df['k33'].values

# Here a custom scaling feature is defined to scale the geometries in the physical range given in the paper
def custom_scale(X):
    X_scaled = X.copy()
    # Scale 'vf' from [0,1] to [0.2, 0.8]
    X_scaled[:, 0] = 0.2 + X[:, 0] * (0.8 - 0.2)
    # Scale 'width' (interpreted as Wy/Sy) from [0,1] to [0.2, 0.95]
    X_scaled[:, 1] = 0.2 + X[:, 1] * (0.95 - 0.2)
    # Scale 'thickness' (interpreted as Ty/Sy) from [0,1] to [0.008, 0.5]
    X_scaled[:, 2] = 0.008 + X[:, 2] * (0.5 - 0.008)
    return X_scaled

# Create a common train-test split - (train/validate) and test split
indices = np.arange(len(X))
train_val_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=42)
train_indices, val_indices = train_test_split(train_val_indices, test_size=0.2, random_state=42)


X_train_common = X[train_indices]
X_val_common = X[val_indices]
X_test_common = X[test_indices]

# Targets
y_k11_train = y_k11[train_indices]
y_k11_val = y_k11[val_indices]
y_k11_test = y_k11[test_indices]

y_k33_train = y_k33[train_indices]
y_k33_val = y_k33[val_indices]
y_k33_test = y_k33[test_indices]

X_train_scaled = custom_scale(X_train_common)
X_val_scaled = custom_scale(X_val_common)
X_test_scaled = custom_scale(X_test_common)

# Callbacks
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=15,
    verbose=1,
    min_lr=1e-6
)

#Model for k11
model_k11 = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(6,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)  # Output layer for regression
])
model_k11.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss='mse',
    metrics=['mae']
)
print("Training model for k11...")
history_k11 = model_k11.fit(
    X_train_scaled, y_k11_train,
    epochs=180, batch_size=64,
    validation_data=(X_val_scaled, y_k11_val),callbacks=[callback,lr_scheduler]
)

np.save("history_k11.npy", history_k11.history)
test_loss, test_mae = model_k11.evaluate(X_test_scaled, y_k11_test)
print(f"k11 Test MAE: {test_mae:.4f}")

# Save the k11 model
save_dir = os.path.join(os.path.dirname(__file__), "saved_model")
os.makedirs(save_dir, exist_ok=True)
model_k11.save(os.path.join(save_dir, "my_model_k11.keras"))




# k33 model
model_k33 = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(6,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])
model_k33.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss='mse',
    metrics=['mae']
)


history_k33 = model_k33.fit(
    X_train_scaled, y_k33_train,
    epochs=150,
    batch_size=64,
    validation_data=(X_val_scaled, y_k33_val),
    callbacks=[callback, lr_scheduler]
)


np.save("history_k33.npy", history_k33.history)
test_loss, test_mae = model_k33.evaluate(X_test_scaled, y_k33_test)
print(f"k33 Test MAE: {test_mae:.4f}")

# Save the k33 model
model_k33.save(os.path.join(save_dir, "my_model_k33.keras"))



y_k11_pred = model_k11.predict(X_test_scaled)
y_k33_pred = model_k33.predict(X_test_scaled)

np.save("y_k11_test.npy", y_k11_test)
np.save("y_k11_pred.npy", y_k11_pred)

np.save("y_k33_test.npy", y_k33_test)
np.save("y_k33_pred.npy", y_k33_pred)


# For R^2 Values
y_k11_pred_r2 = y_k11_pred.flatten()
y_k33_pred_r2 = y_k33_pred.flatten()

r2_k11 = r2_score(y_k11_test, y_k11_pred_r2)
print(f"R² score for k11: {r2_k11:.4f}")

r2_k33 = r2_score(y_k33_test, y_k33_pred_r2)
print(f"R² score for k33: {r2_k33:.4f}")
