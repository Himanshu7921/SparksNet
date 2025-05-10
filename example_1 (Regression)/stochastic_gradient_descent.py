import numpy as np
import warnings

def calulate_loss(y_preds: np.ndarray, y_true: np.ndarray) -> float:
    return np.mean((y_preds - y_true) ** 2)

def update_weight_bias(weight: float, dw: float, bias: float, db: float, lr: float = 0.01) -> tuple[float, float]:
    weight -= dw * lr
    bias -= db * lr
    return weight, bias

def update_weight_bias_momentum(weight: float, dw: float, bias: float, db: float, lr: float = 0.01, momentum = 0.9, v_w: float = 0, v_b: float = 0) -> tuple[float, float]:
    v_w = v_w * momentum + lr * dw
    v_b = v_b * momentum + lr * db

    weight -= v_w
    bias -= v_b
    return weight, bias, v_w, v_b

def grad_of_loss_wrt_y_preds(y_preds: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    N = y_preds.shape[0]
    return (2/N) * (y_preds - y_true)

def grad_of_y_preds_wrt_w(X: np.ndarray) -> np.ndarray:
    return X

def predict(X: np.ndarray, w: float, b: float) -> np.ndarray:
    return w * X + b

X = np.array([1, 2, 3, 4, 5, 6])
y_true = 2 * X + 3

w, b = 0.0, 0.0
v_w, v_b = 0.0, 0.0
for epoch in range(300):
    print(f"\nEpoch: {epoch + 1}: ")
    for i in range(len(X)):
        x_i = X[i]
        y_true_i = y_true[i]
        y_pred_i = predict(x_i, w, b)

        # Loss Calculation for each sample in y_preds and y_true
        loss = y_pred_i - y_true_i

        # calculate dw, db
        dw, db = (2) * (loss) * grad_of_y_preds_wrt_w(x_i), (2) * (loss)

        # update the weights and bias
        # w, b  = update_weight_bias(w, dw, b, db, lr = 0.001)
        w, b, v_w, v_b = update_weight_bias_momentum(w, dw, b, db, lr = 0.001, momentum=0.9, v_w = v_w, v_b = v_b)

    # Predict for full dataset just to show loss after epoch
    y_preds = w * X + b
    loss = calulate_loss(y_preds, y_true)
    print(f"Loss after epoch {epoch + 1}: {loss:.4f}")
    print(f"Updated w = {w:.4f}, b = {b:.4f}")

    if abs(dw) < 1e-2 and abs(db) < 1e-2:
        warnings.warn(f"Stopping at Epoch {epoch + 1} due to small gradient updates: dw={dw:.4f}, db={db:.4f}")
        break



