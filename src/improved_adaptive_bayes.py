# improved_adaptive_bayes.py
import os
import math
import time
import numpy as np
from numba import njit, prange

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except Exception:
    CUPY_AVAILABLE = False

# ---------------------------
# Enhanced Numba kernels
# ---------------------------
@njit(cache=True, fastmath=True)
def _robust_transform(x, transform_type=0):
    """Более стабильные преобразования признаков"""
    if transform_type == 0:  # log1p with clipping
        if x <= -0.99:
            return math.log(0.01)
        return math.log1p(x)
    elif transform_type == 1:  # tanh normalization
        return math.tanh(x)
    elif transform_type == 2:  # arctan normalization
        return math.atan(x) * 2.0 / math.pi
    else:  # identity
        return x

@njit(cache=True, fastmath=True)
def _stable_sigmoid(z):
    """Более стабильная сигмоида"""
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    else:
        ez = math.exp(z)
        return ez / (1.0 + ez)

@njit(cache=True, fastmath=True)
def _compute_hessian_diag(p):
    """Диагональ гессиана для второго порядка"""
    return p * (1.0 - p)

@njit(parallel=True, cache=True, fastmath=True)
def _score_batch_enhanced(X, w, eps, transform_type):
    n, d = X.shape
    out = np.empty(n, dtype=np.float64)
    for i in prange(n):
        s = w[0]  # bias term
        for j in range(d):
            x = X[i, j]
            if abs(x) > eps:
                x_transformed = _robust_transform(x, np.int32(transform_type))
                s += w[j + 1] * x_transformed
        out[i] = s
    return out

@njit(parallel=True, cache=True, fastmath=True)
def _update_batch_enhanced(X, y, w, momentum, base_lr, eps, l1_reg, l2_reg, 
                          transform_type, use_newton):
    n, d = X.shape
    
    for i in prange(n):
        # Forward pass
        s = w[0]  # bias
        x_transformed = np.empty(d, dtype=np.float64)
        
        for j in range(d):
            x = X[i, j]
            if abs(x) > eps:
                x_transformed[j] = _robust_transform(x, np.int32(transform_type))
                s += w[j + 1] * x_transformed[j]
            else:
                x_transformed[j] = 0.0
        
        p = _stable_sigmoid(s)
        err = y[i] - p
        
        # Adaptive learning rate с улучшенной формулой
        confidence = abs(p - 0.5) * 2.0  # [0, 1]
        uncertainty = 1.0 - confidence
        adaptive_factor = uncertainty * (1.0 + abs(err))
        lr = base_lr * adaptive_factor
        
        # Newton's method adjustment
        if use_newton:
            hess_diag = _compute_hessian_diag(p)
            if hess_diag > 1e-8:
                lr = lr / (1.0 + hess_diag)
        
        # Update bias
        momentum[0] = 0.9 * momentum[0] + lr * err
        w[0] += momentum[0]
        
        # Update weights with regularization
        for j in range(d):
            if abs(X[i, j]) > eps:
                # L1 regularization (soft thresholding)
                l1_penalty = l1_reg * (1.0 if w[j + 1] > 0 else -1.0)
                # L2 regularization
                l2_penalty = l2_reg * w[j + 1]
                
                grad = err * x_transformed[j] - l1_penalty - l2_penalty
                momentum[j + 1] = 0.9 * momentum[j + 1] + lr * grad
                w[j + 1] += momentum[j + 1]
                
                # L1 soft thresholding
                if abs(w[j + 1]) < l1_reg * lr:
                    w[j + 1] = 0.0

class AdaptiveBayes:
    def __init__(self, base_lr=1e-2, eps=1e-8, l1_reg=1e-5, l2_reg=1e-4,
                 transform_type='log1p', use_newton=True, init_method='xavier',
                 early_stopping=True, patience=5, device=None):
        """
        Улучшенная версия AdaptiveBayes с повышенной точностью
        
        Parameters:
        -----------
        base_lr : float, learning rate
        eps : float, threshold for feature selection
        l1_reg : float, L1 regularization strength
        l2_reg : float, L2 regularization strength  
        transform_type : str, feature transform ('log1p', 'tanh', 'arctan', 'identity')
        use_newton : bool, use Newton's method approximation
        init_method : str, weight initialization ('xavier', 'he', 'normal')
        early_stopping : bool, use early stopping
        patience : int, early stopping patience
        """
        self.base_lr = float(base_lr)
        self.eps = float(eps)
        self.l1_reg = float(l1_reg)
        self.l2_reg = float(l2_reg)
        self.use_newton = use_newton
        self.early_stopping = early_stopping
        self.patience = patience
        self.init_method = init_method
        
        # Transform type mapping
        self.transform_map = {'log1p': 0, 'tanh': 1, 'arctan': 2, 'identity': 3}
        self.transform_type = self.transform_map.get(transform_type, 0)
        
        self.device = device
        self._is_gpu = False
        self.w = None
        self.momentum = None
        self._xp = np
        self._backend = 'cpu'
        self.best_weights = None
        self.best_score = -np.inf

        # GPU setup
        if device is None and CUPY_AVAILABLE:
            self._is_gpu = True
            self._xp = cp
            self._backend = 'gpu'
        elif device == 'gpu' and CUPY_AVAILABLE:
            self._is_gpu = True
            self._xp = cp
            self._backend = 'gpu'

    def _initialize_weights(self, d):
        """Улучшенная инициализация весов"""
        # d+1 because we include bias term
        total_params = d + 1
        
        if self.init_method == 'xavier':
            # Xavier/Glorot initialization
            limit = math.sqrt(6.0 / (d + 1))
            if self._is_gpu:
                self.w = self._xp.random.uniform(-limit, limit, total_params).astype(self._xp.float64)
                self.momentum = self._xp.zeros(total_params, dtype=self._xp.float64)
            else:
                self.w = np.random.uniform(-limit, limit, total_params).astype(np.float64)
                self.momentum = np.zeros(total_params, dtype=np.float64)
                
        elif self.init_method == 'he':
            # He initialization
            std = math.sqrt(2.0 / d)
            if self._is_gpu:
                self.w = self._xp.random.normal(0, std, total_params).astype(self._xp.float64)
                self.momentum = self._xp.zeros(total_params, dtype=self._xp.float64)
            else:
                self.w = np.random.normal(0, std, total_params).astype(np.float64)
                self.momentum = np.zeros(total_params, dtype=np.float64)
        else:
            # Normal initialization (fallback)
            if self._is_gpu:
                self.w = self._xp.random.normal(0, 0.01, total_params).astype(self._xp.float64)
                self.momentum = self._xp.zeros(total_params, dtype=self._xp.float64)
            else:
                self.w = np.random.normal(0, 0.01, total_params).astype(np.float64)
                self.momentum = np.zeros(total_params, dtype=np.float64)

    def _score_cpu(self, X):
        # Convert cupy arrays to numpy if needed
        if hasattr(X, 'get'):  # cupy array
            X = X.get()
        if hasattr(self.w, 'get'):  # cupy array
            w = self.w.get()
        else:
            w = self.w
        return _score_batch_enhanced(X, w, self.eps, np.int32(self.transform_type))

    def _update_cpu(self, X, y):
        # Convert cupy arrays to numpy if needed
        if hasattr(X, 'get'):
            X = X.get()
        if hasattr(y, 'get'):
            y = y.get()
        if hasattr(self.w, 'get'):
            w = self.w.get()
            momentum = self.momentum.get()
        else:
            w = self.w
            momentum = self.momentum
        
        _update_batch_enhanced(X, y, w, momentum, self.base_lr, 
                              self.eps, self.l1_reg, self.l2_reg, 
                              np.int32(self.transform_type), self.use_newton)
        
        # Copy back if needed
        if hasattr(self.w, 'get'):
            self.w = self._xp.asarray(w)
            self.momentum = self._xp.asarray(momentum)
        else:
            self.w = w
            self.momentum = momentum

    def _compute_validation_score(self, X_val, y_val):
        """Вычисление метрики качества для early stopping"""
        probs = self.predict_proba(X_val)
        # Log-likelihood
        eps = 1e-15
        probs = np.clip(probs, eps, 1 - eps)
        ll = np.mean(y_val * np.log(probs) + (1 - y_val) * np.log(1 - probs))
        return ll

    def fit(self, X, y, X_val=None, y_val=None, epochs=10, batch_size=32768, shuffle=True):
        """
        Обучение с улучшениями точности
        """
        xp = self._xp
        
        # Convert to proper arrays
        if not self._is_gpu:
            X = np.asarray(X, dtype=np.float64)
            y = np.asarray(y, dtype=np.float64)
            if X_val is not None:
                X_val = np.asarray(X_val, dtype=np.float64)
                y_val = np.asarray(y_val, dtype=np.float64)
        else:
            X = cp.asarray(X, dtype=cp.float64)
            y = cp.asarray(y, dtype=cp.float64)
            if X_val is not None:
                X_val = cp.asarray(X_val, dtype=cp.float64)
                y_val = cp.asarray(y_val, dtype=cp.float64)

        n, d = X.shape
        if self.w is None:
            self._initialize_weights(d)

        best_val_score = -np.inf
        patience_counter = 0

        for epoch in range(epochs):
            # Shuffle data
            idx = xp.arange(n)
            if shuffle:
                if self._is_gpu:
                    idx = cp.random.permutation(n)
                else:
                    idx = np.random.permutation(n)

            # Mini-batch training
            for start in range(0, n, batch_size):
                end = min(n, start + batch_size)
                bidx = idx[start:end]
                Xb = X[bidx]
                yb = y[bidx]
                
                if self._is_gpu:
                    # GPU implementation would go here
                    pass
                else:
                    Xb_cpu = np.ascontiguousarray(Xb)
                    yb_cpu = np.ascontiguousarray(yb)
                    self._update_cpu(Xb_cpu, yb_cpu)

            # Early stopping check
            if self.early_stopping and X_val is not None and y_val is not None:
                val_score = self._compute_validation_score(X_val, y_val)
                
                if val_score > best_val_score:
                    best_val_score = val_score
                    self.best_weights = self.w.copy()
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                if patience_counter >= self.patience:
                    print(f"Early stopping at epoch {epoch}")
                    if self.best_weights is not None:
                        self.w = self.best_weights
                    break

        return self

    def predict_proba(self, X, batch_size=262144):
        """Предсказание вероятностей с улучшенной стабильностью"""
        xp = self._xp
        if self._is_gpu and not isinstance(X, cp.ndarray):
            X = cp.asarray(X, dtype=cp.float64)
        if (not self._is_gpu) and not isinstance(X, np.ndarray):
            X = np.asarray(X, dtype=np.float64)

        n = X.shape[0]
        probs = []
        
        for start in range(0, n, batch_size):
            end = min(n, start + batch_size)
            Xb = X[start:end]
            
            if self._is_gpu:
                # Ensure numpy array for CPU computation
                Xb_cpu = cp.asnumpy(Xb) if hasattr(Xb, 'get') else Xb
                scores = self._score_cpu(Xb_cpu)
                pb = 1.0 / (1.0 + np.exp(-np.clip(scores, -500, 500)))
                probs.append(pb)
            else:
                scores = self._score_cpu(np.ascontiguousarray(Xb))
                # Более стабильная сигмоида с клиппингом
                pb = 1.0 / (1.0 + np.exp(-np.clip(scores, -500, 500)))
                probs.append(pb)
                
        out = np.concatenate(probs, axis=0)
        # Always return numpy array, not cupy
        if hasattr(out, 'get'):
            return out.get()
        return out

    def predict(self, X, threshold=0.5):
        p = self.predict_proba(X)
        # Ensure numpy array
        if hasattr(p, 'get'):
            p = p.get()
        return (p >= threshold).astype(np.int32)

    def get_feature_importance(self):
        """Получить важность признаков"""
        if self.w is None:
            return None
        # Skip bias term (first weight)
        return np.abs(self.w[1:])
        
    def set_params(self, **params):
        """Установить гиперпараметры (для совместимости с sklearn)"""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self