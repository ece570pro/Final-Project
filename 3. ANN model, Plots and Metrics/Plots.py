import numpy as np
import matplotlib.pyplot as plt

# Load saved histories
history_k11 = np.load("history_k11.npy", allow_pickle=True).item()
history_k33 = np.load("history_k33.npy", allow_pickle=True).item()

# Load true and predicted values
y_k11_test = np.load("y_k11_test.npy")
y_k11_pred = np.load("y_k11_pred.npy")

y_k33_test = np.load("y_k33_test.npy")
y_k33_pred = np.load("y_k33_pred.npy")

# -------- Plot 1: MSE vs Epochs for k11 --------
plt.figure()
plt.plot(history_k11['loss'], label='Training', color='tab:blue')
plt.plot(history_k11['val_loss'], label='Validation', color='tab:orange')
plt.xlabel('Epoch')
plt.ylabel('MSE [$k_{11}$]')
plt.title('MSE vs Epochs for $k_{11}$')
plt.ylim(0, 0.005)  # Force y-axis to zoom into this range
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("mse_vs_epoch_k11.jpg", dpi=300)



# -------- Plot 2: MSE vs Epochs for k33 --------
plt.figure()
plt.plot(history_k33['loss'], label='Training', color='tab:blue')
plt.plot(history_k33['val_loss'], label='Validation', color='tab:orange')
plt.xlabel('Epoch')
plt.ylabel('MSE [$k_{33}$]')
plt.title('MSE vs Epochs for $k_{33}$')
plt.ylim(0, 0.0001)  # Force y-axis to zoom into this range
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("mse_vs_epoch_k33.jpg", dpi=300)



# -------- Plot 3: Predicted vs True for k11 --------
plt.figure()
plt.scatter(y_k11_test, y_k11_pred, s=10, alpha=0.8)
plt.plot([y_k11_test.min(), y_k11_test.max()],
         [y_k11_test.min(), y_k11_test.max()], 'k--')
plt.xlabel("True Values [$k_{11}$]")
plt.ylabel("Predicted Values [$k_{11}$]")
plt.title("Predicted vs True for $k_{11}$")
plt.grid(True)
plt.tight_layout()
plt.savefig("predicted_vs_true_k11.jpg", dpi=300)


# -------- Plot 4: Predicted vs True for k33 --------
plt.figure()
plt.scatter(y_k33_test, y_k33_pred, s=10, alpha=0.8)
plt.plot([y_k33_test.min(), y_k33_test.max()],
         [y_k33_test.min(), y_k33_test.max()], 'k--')
plt.xlabel("True Values [$k_{33}$]")
plt.ylabel("Predicted Values [$k_{33}$]")
plt.title("Predicted vs True for $k_{33}$")
plt.grid(True)
plt.tight_layout()
plt.savefig("predicted_vs_true_k33.jpg", dpi=300)

