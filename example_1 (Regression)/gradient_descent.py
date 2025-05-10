import numpy as np

def calulate_loss(y_preds: np.ndarray, y_true: np.ndarray) -> float:
    return np.mean((y_preds - y_true) ** 2)

def update_weight(weight: float, dw: float, lr: float = 0.01) -> float:
    weight -= dw * lr
    return weight

def grad_of_loss_wrt_y_preds(y_preds: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    N = y_preds.shape[0]
    return (2/N) * (y_preds - y_true)

def grad_of_y_preds_wrt_w(X: np.ndarray) -> np.ndarray:
    return X

def predict(X: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    return w * X + b

X = np.array([1, 2, 3, 4])
y_true = np.array([3, 6, 9, 12])

# for w = 0; b = 0
w, b = 0, 0
y_preds = predict(X, w, b)

# for w = 0; b = 0
for epoch in range(32):
    print(f"\nEpoch: {epoch + 1}: ")
    w, b = w, b
    y_preds = predict(X, w, b)
    print(y_preds - y_true)
    print(f"Predicted Values for weight = {w}, bias = {b} is {y_preds}")
    print(f"    (y_preds - y_true)^2 = {(y_preds - y_true) ** 2}")
    print(f"    (y_preds - y_true)^2 / N = Loss for w = {w}, b = {b}")
    print(f"The Loss for w = {w}, b = {b} is: {calulate_loss(y_preds, y_true):.2f}")
    dw = grad_of_loss_wrt_y_preds(y_preds, y_true) @ grad_of_y_preds_wrt_w(X).T
    print(f"    dLoss/dy_preds = {grad_of_loss_wrt_y_preds(y_preds, y_true)}")
    print(f"    dy_preds/dw = {grad_of_y_preds_wrt_w(X)}")
    print(f"    dw = dLoss/dw = dLoss/dy_preds * dy_preds/dw = {dw}")
    w = update_weight(weight = w, dw = dw, lr = 0.01)
    print(f"Updated value of w = w - dw * lr is {w:.2f}")
    print("----" * 30)
