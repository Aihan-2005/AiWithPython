import itertools
import numpy as np


# ============================================================
# Basic Metrics
# ============================================================

def mean_squared_error(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean((y_true - y_pred) ** 2)


def residual_sum_of_squares(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.sum((y_true - y_pred) ** 2)


def r2_score(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    rss = np.sum((y_true - y_pred) ** 2)
    tss = np.sum((y_true - np.mean(y_true)) ** 2)

    return 1 - rss / tss


# ============================================================
# Train Test Split From Scratch
# ============================================================

def train_test_split_scratch(X, y, test_size=0.2, random_state=42):
    rng = np.random.default_rng(random_state)

    X = np.asarray(X)
    y = np.asarray(y)

    n = X.shape[0]
    indices = np.arange(n)
    rng.shuffle(indices)

    n_test = int(n * test_size)

    test_indices = indices[:n_test]
    train_indices = indices[n_test:]

    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


# ============================================================
# K-Fold Split From Scratch
# ============================================================

def kfold_indices(n_samples, k=5, random_state=42, shuffle=True):
    indices = np.arange(n_samples)

    if shuffle:
        rng = np.random.default_rng(random_state)
        rng.shuffle(indices)

    folds = np.array_split(indices, k)

    for i in range(k):
        val_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        yield train_idx, val_idx


def kfold_cv_score(model_factory, X, y, k=5, random_state=42):
    """
    model_factory باید هر بار یک مدل جدید بسازد.
    مثال:
        lambda: RidgeRegressionScratch(lambda_=10)
    """

    X = np.asarray(X)
    y = np.asarray(y)

    errors = []

    for train_idx, val_idx in kfold_indices(len(y), k=k, random_state=random_state):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = model_factory()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        errors.append(mean_squared_error(y_val, y_pred))

    return np.mean(errors), np.std(errors)


# ============================================================
# Ordinary Least Squares From Scratch
# ============================================================

class LinearRegressionScratch:
    """
    Ordinary Least Squares Regression from scratch.

    مدل:
        y = beta_0 + beta_1 x_1 + ... + beta_p x_p + error

    ضرایب با pseudo-inverse محاسبه می‌شوند:
        beta = (X^T X)^(-1) X^T y

    ولی برای پایداری، از np.linalg.pinv استفاده می‌کنیم.
    """

    def __init__(self):
        self.intercept_ = None
        self.coef_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        n = X.shape[0]

        X_aug = np.column_stack([np.ones(n), X])

        beta = np.linalg.pinv(X_aug) @ y

        self.intercept_ = beta[0]
        self.coef_ = beta[1:]

        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return self.intercept_ + X @ self.coef_


# ============================================================
# Best Subset Selection From Scratch
# ============================================================

def best_subset_selection(X, y, max_features=None):
    """
    Best Subset Selection.

    برای هر k، همه مدل‌های شامل k ویژگی را بررسی می‌کند.
    سپس بهترین مدل با کمترین RSS را انتخاب می‌کند.

    هشدار:
    این روش برای p بزرگ بسیار سنگین است چون تعداد مدل‌ها 2^p است.
    """

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    n_features = X.shape[1]

    if max_features is None:
        max_features = n_features

    results = []

    for k in range(1, max_features + 1):
        best_rss = np.inf
        best_features = None
        best_model = None
        best_r2 = None

        for features in itertools.combinations(range(n_features), k):
            X_subset = X[:, features]

            model = LinearRegressionScratch()
            model.fit(X_subset, y)

            y_pred = model.predict(X_subset)
            rss = residual_sum_of_squares(y, y_pred)

            if rss < best_rss:
                best_rss = rss
                best_features = features
                best_model = model
                best_r2 = r2_score(y, y_pred)

        results.append({
            "num_features": k,
            "features": best_features,
            "rss": best_rss,
            "r2": best_r2,
            "model": best_model
        })

    return results


# ============================================================
# Forward Stepwise Selection From Scratch
# ============================================================

def forward_stepwise_selection(X, y, max_features=None):
    """
    Forward Stepwise Selection.

    از مدل خالی شروع می‌کند.
    در هر مرحله یک feature اضافه می‌کند.
    featureای اضافه می‌شود که بیشترین کاهش RSS را بدهد.
    """

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    n_features = X.shape[1]

    if max_features is None:
        max_features = n_features

    selected = []
    remaining = list(range(n_features))
    results = []

    for k in range(1, max_features + 1):
        best_rss = np.inf
        best_feature = None
        best_model = None
        best_r2 = None

        for feature in remaining:
            candidate_features = selected + [feature]
            X_subset = X[:, candidate_features]

            model = LinearRegressionScratch()
            model.fit(X_subset, y)

            y_pred = model.predict(X_subset)
            rss = residual_sum_of_squares(y, y_pred)

            if rss < best_rss:
                best_rss = rss
                best_feature = feature
                best_model = model
                best_r2 = r2_score(y, y_pred)

        selected.append(best_feature)
        remaining.remove(best_feature)

        results.append({
            "num_features": k,
            "features": tuple(selected),
            "rss": best_rss,
            "r2": best_r2,
            "model": best_model
        })

    return results


# ============================================================
# Backward Stepwise Selection From Scratch
# ============================================================

def backward_stepwise_selection(X, y, min_features=1):
    """
    Backward Stepwise Selection.

    از مدل کامل شروع می‌کند.
    در هر مرحله یک feature حذف می‌کند.
    featureای حذف می‌شود که بعد از حذف آن، مدل بهترین RSS را داشته باشد.

    توجه:
    این روش وقتی p > n مشکل دارد چون از مدل کامل شروع می‌کند.
    """

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    current_features = list(range(X.shape[1]))
    results = []

    while len(current_features) >= min_features:
        X_subset = X[:, current_features]

        model = LinearRegressionScratch()
        model.fit(X_subset, y)

        y_pred = model.predict(X_subset)
        rss = residual_sum_of_squares(y, y_pred)

        results.append({
            "num_features": len(current_features),
            "features": tuple(current_features),
            "rss": rss,
            "r2": r2_score(y, y_pred),
            "model": model
        })

        if len(current_features) == min_features:
            break

        best_rss = np.inf
        best_features_after_removal = None

        for feature in current_features:
            candidate_features = [f for f in current_features if f != feature]
            X_candidate = X[:, candidate_features]

            candidate_model = LinearRegressionScratch()
            candidate_model.fit(X_candidate, y)

            candidate_pred = candidate_model.predict(X_candidate)
            candidate_rss = residual_sum_of_squares(y, candidate_pred)

            if candidate_rss < best_rss:
                best_rss = candidate_rss
                best_features_after_removal = candidate_features

        current_features = best_features_after_removal

    return results


# ============================================================
# Ridge Regression From Scratch
# ============================================================

class RidgeRegressionScratch:
    """
    Ridge Regression from scratch.

    تابع هدف:
        RSS + lambda * sum(beta_j^2)

    intercept جریمه نمی‌شود.
    """

    def __init__(self, lambda_=1.0):
        self.lambda_ = lambda_
        self.intercept_ = None
        self.coef_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        n, p = X.shape

        X_aug = np.column_stack([np.ones(n), X])

        penalty = np.eye(p + 1)
        penalty[0, 0] = 0.0

        A = X_aug.T @ X_aug + self.lambda_ * penalty
        b = X_aug.T @ y

        beta = np.linalg.pinv(A) @ b

        self.intercept_ = beta[0]
        self.coef_ = beta[1:]

        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return self.intercept_ + X @ self.coef_


# ============================================================
# Standardization Helper
# ============================================================

class Standardizer:
    """
    Standardization from scratch.

    x_scaled = (x - mean) / std
    """

    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)

        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)

        self.std_[self.std_ == 0] = 1.0

        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


# ============================================================
# Lasso Regression From Scratch
# ============================================================

def soft_threshold(value, threshold):
    """
    Soft-thresholding operator.

    اگر value مثبت و بزرگ باشد:
        value - threshold

    اگر value منفی و کوچک باشد:
        value + threshold

    اگر قدرمطلق value کمتر از threshold باشد:
        صفر
    """

    if value > threshold:
        return value - threshold
    elif value < -threshold:
        return value + threshold
    else:
        return 0.0


class LassoRegressionScratch:
    """
    Lasso Regression from scratch using Coordinate Descent.

    تابع هدف کتاب:
        RSS + lambda * sum(|beta_j|)

    نکته:
    قبل از Lasso، X را standardize می‌کنیم.
    intercept جریمه نمی‌شود.
    """

    def __init__(self, lambda_=1.0, max_iter=5000, tol=1e-6):
        self.lambda_ = lambda_
        self.max_iter = max_iter
        self.tol = tol

        self.intercept_ = None
        self.coef_ = None

        self.x_mean_ = None
        self.x_std_ = None
        self.y_mean_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        n, p = X.shape

        self.x_mean_ = X.mean(axis=0)
        self.x_std_ = X.std(axis=0)
        self.x_std_[self.x_std_ == 0] = 1.0

        self.y_mean_ = y.mean()

        Xs = (X - self.x_mean_) / self.x_std_
        yc = y - self.y_mean_

        beta = np.zeros(p)

        for _ in range(self.max_iter):
            beta_old = beta.copy()

            for j in range(p):
                residual_j = yc - Xs @ beta + Xs[:, j] * beta[j]

                rho_j = Xs[:, j] @ residual_j
                z_j = Xs[:, j] @ Xs[:, j]

                beta[j] = soft_threshold(rho_j, self.lambda_ / 2) / z_j

            max_change = np.max(np.abs(beta - beta_old))

            if max_change < self.tol:
                break

        self.coef_ = beta / self.x_std_
        self.intercept_ = self.y_mean_ - self.x_mean_ @ self.coef_

        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return self.intercept_ + X @ self.coef_


# ============================================================
# PCA From Scratch
# ============================================================

class PCAScratch:
    """
    Principal Components Analysis from scratch using SVD.

    PCA فقط روی X انجام می‌شود و Y را نمی‌بیند.
    """

    def __init__(self, n_components=None, standardize=True):
        self.n_components = n_components
        self.standardize = standardize

        self.mean_ = None
        self.std_ = None
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)

        n, p = X.shape

        self.mean_ = X.mean(axis=0)
        X_centered = X - self.mean_

        if self.standardize:
            self.std_ = X_centered.std(axis=0)
            self.std_[self.std_ == 0] = 1.0
            X_processed = X_centered / self.std_
        else:
            self.std_ = np.ones(p)
            X_processed = X_centered

        U, S, Vt = np.linalg.svd(X_processed, full_matrices=False)

        explained_variance = (S ** 2) / (n - 1)
        total_variance = explained_variance.sum()

        self.explained_variance_ = explained_variance
        self.explained_variance_ratio_ = explained_variance / total_variance

        if self.n_components is None:
            self.n_components = min(n, p)

        self.components_ = Vt[:self.n_components]

        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)

        X_centered = X - self.mean_
        X_processed = X_centered / self.std_

        return X_processed @ self.components_.T

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


# ============================================================
# Principal Components Regression From Scratch
# ============================================================

class PCRScratch:
    """
    Principal Components Regression from scratch.

    مراحل:
        1. PCA روی X
        2. گرفتن M principal components
        3. Linear Regression روی componentها
    """

    def __init__(self, n_components=2, standardize=True):
        self.n_components = n_components
        self.standardize = standardize

        self.pca_ = None
        self.regression_ = None

    def fit(self, X, y):
        self.pca_ = PCAScratch(
            n_components=self.n_components,
            standardize=self.standardize
        )

        Z = self.pca_.fit_transform(X)

        self.regression_ = LinearRegressionScratch()
        self.regression_.fit(Z, y)

        return self

    def predict(self, X):
        Z = self.pca_.transform(X)
        return self.regression_.predict(Z)


# ============================================================
# Partial Least Squares From Scratch
# ============================================================

class PLSScratch:
    """
    Partial Least Squares Regression from scratch.

    این نسخه برای یک response یعنی y تک‌بعدی نوشته شده است.

    تفاوت با PCR:
        PCR فقط X را نگاه می‌کند.
        PLS برای ساخت componentها از Y هم استفاده می‌کند.
    """

    def __init__(self, n_components=2, standardize=True):
        self.n_components = n_components
        self.standardize = standardize

        self.x_mean_ = None
        self.x_std_ = None
        self.y_mean_ = None

        self.W_ = None
        self.P_ = None
        self.q_ = None

        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)

        n, p = X.shape

        self.x_mean_ = X.mean(axis=0)
        X_centered = X - self.x_mean_

        if self.standardize:
            self.x_std_ = X_centered.std(axis=0)
            self.x_std_[self.x_std_ == 0] = 1.0
            X_res = X_centered / self.x_std_
        else:
            self.x_std_ = np.ones(p)
            X_res = X_centered

        self.y_mean_ = y.mean()
        y_res = y - self.y_mean_

        W = []
        P = []
        q_values = []

        for _ in range(self.n_components):
            # وزن‌ها براساس ارتباط X با y
            w = X_res.T @ y_res

            norm_w = np.linalg.norm(w)

            if norm_w < 1e-12:
                break

            w = w / norm_w

            # score یا component
            t = X_res @ w

            t_norm_squared = t @ t

            if t_norm_squared < 1e-12:
                break

            # loading برای X
            p_vec = (X_res.T @ t) / t_norm_squared

            # loading برای y
            q = (y_res @ t) / t_norm_squared

            # deflation
            X_res = X_res - np.outer(t, p_vec)
            y_res = y_res - q * t

            W.append(w)
            P.append(p_vec)
            q_values.append(q)

        self.W_ = np.column_stack(W)
        self.P_ = np.column_stack(P)
        self.q_ = np.array(q_values)

        # ضریب نهایی در فضای standardized X
        # beta_scaled = W (P^T W)^(-1) q
        beta_scaled = self.W_ @ np.linalg.pinv(self.P_.T @ self.W_) @ self.q_

        self.coef_ = beta_scaled / self.x_std_
        self.intercept_ = self.y_mean_ - self.x_mean_ @ self.coef_

        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return self.intercept_ + X @ self.coef_


# ============================================================
# Hyperparameter Selection Helpers
# ============================================================

def select_best_ridge_lambda(X, y, lambda_values, k=5, random_state=42):
    results = []

    for lambda_ in lambda_values:
        mean_cv, std_cv = kfold_cv_score(
            model_factory=lambda lambda_=lambda_: RidgeRegressionScratch(lambda_=lambda_),
            X=X,
            y=y,
            k=k,
            random_state=random_state
        )

        results.append({
            "lambda": lambda_,
            "mean_cv_mse": mean_cv,
            "std_cv_mse": std_cv
        })

    best = min(results, key=lambda item: item["mean_cv_mse"])

    return best, results


def select_best_lasso_lambda(X, y, lambda_values, k=5, random_state=42):
    results = []

    for lambda_ in lambda_values:
        mean_cv, std_cv = kfold_cv_score(
            model_factory=lambda lambda_=lambda_: LassoRegressionScratch(lambda_=lambda_),
            X=X,
            y=y,
            k=k,
            random_state=random_state
        )

        results.append({
            "lambda": lambda_,
            "mean_cv_mse": mean_cv,
            "std_cv_mse": std_cv
        })

    best = min(results, key=lambda item: item["mean_cv_mse"])

    return best, results


def select_best_pcr_components(X, y, component_values, k=5, random_state=42):
    results = []

    for m in component_values:
        mean_cv, std_cv = kfold_cv_score(
            model_factory=lambda m=m: PCRScratch(n_components=m),
            X=X,
            y=y,
            k=k,
            random_state=random_state
        )

        results.append({
            "n_components": m,
            "mean_cv_mse": mean_cv,
            "std_cv_mse": std_cv
        })

    best = min(results, key=lambda item: item["mean_cv_mse"])

    return best, results


def select_best_pls_components(X, y, component_values, k=5, random_state=42):
    results = []

    for m in component_values:
        mean_cv, std_cv = kfold_cv_score(
            model_factory=lambda m=m: PLSScratch(n_components=m),
            X=X,
            y=y,
            k=k,
            random_state=random_state
        )

        results.append({
            "n_components": m,
            "mean_cv_mse": mean_cv,
            "std_cv_mse": std_cv
        })

    best = min(results, key=lambda item: item["mean_cv_mse"])

    return best, results


# ============================================================
# Simulated Data Generator
# ============================================================

def generate_sparse_regression_data(
    n_samples=100,
    n_features=30,
    n_signal=5,
    noise_std=1.0,
    random_state=42
):
    """
    دیتاست شبیه‌سازی‌شده برای regression.

    فقط n_signal ویژگی واقعاً با y مرتبط‌اند.
    بقیه ویژگی‌ها noise هستند.
    """

    rng = np.random.default_rng(random_state)

    X = rng.normal(0, 1, size=(n_samples, n_features))

    beta_true = np.zeros(n_features)

    signal_indices = np.arange(n_signal)
    beta_true[signal_indices] = rng.uniform(1.5, 4.0, size=n_signal)

    noise = rng.normal(0, noise_std, size=n_samples)

    y = X @ beta_true + noise

    return X, y, beta_true


# ============================================================
# Demo
# ============================================================

if __name__ == "__main__":

    X, y, beta_true = generate_sparse_regression_data(
        n_samples=120,
        n_features=20,
        n_signal=5,
        noise_std=1.0,
        random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split_scratch(
        X, y, test_size=0.25, random_state=42
    )

    print("=" * 70)
    print("True non-zero coefficients:")
    print(np.where(beta_true != 0)[0])
    print(beta_true)
    print("=" * 70)

    # ------------------------------------------------------------
    # Ordinary Least Squares
    # ------------------------------------------------------------

    ols = LinearRegressionScratch()
    ols.fit(X_train, y_train)

    y_pred_ols = ols.predict(X_test)

    print("\nOLS Test MSE:", mean_squared_error(y_test, y_pred_ols))
    print("OLS Test R2:", r2_score(y_test, y_pred_ols))

    # ------------------------------------------------------------
    # Ridge with CV
    # ------------------------------------------------------------

    lambda_grid = np.logspace(-3, 3, 15)

    best_ridge, ridge_results = select_best_ridge_lambda(
        X_train, y_train, lambda_grid, k=5, random_state=42
    )

    ridge = RidgeRegressionScratch(lambda_=best_ridge["lambda"])
    ridge.fit(X_train, y_train)

    y_pred_ridge = ridge.predict(X_test)

    print("\nBest Ridge Lambda:", best_ridge)
    print("Ridge Test MSE:", mean_squared_error(y_test, y_pred_ridge))
    print("Ridge Test R2:", r2_score(y_test, y_pred_ridge))

    # ------------------------------------------------------------
    # Lasso with CV
    # ------------------------------------------------------------

    lasso_lambda_grid = np.logspace(-2, 2, 15)

    best_lasso, lasso_results = select_best_lasso_lambda(
        X_train, y_train, lasso_lambda_grid, k=5, random_state=42
    )

    lasso = LassoRegressionScratch(lambda_=best_lasso["lambda"])
    lasso.fit(X_train, y_train)

    y_pred_lasso = lasso.predict(X_test)

    print("\nBest Lasso Lambda:", best_lasso)
    print("Lasso Test MSE:", mean_squared_error(y_test, y_pred_lasso))
    print("Lasso Test R2:", r2_score(y_test, y_pred_lasso))
    print("Lasso Non-zero Coefficients:", np.where(np.abs(lasso.coef_) > 1e-6)[0])

    # ------------------------------------------------------------
    # PCR with CV
    # ------------------------------------------------------------

    component_values = list(range(1, min(X_train.shape[1], X_train.shape[0]) + 1))

    best_pcr, pcr_results = select_best_pcr_components(
        X_train, y_train, component_values, k=5, random_state=42
    )

    pcr = PCRScratch(n_components=best_pcr["n_components"])
    pcr.fit(X_train, y_train)

    y_pred_pcr = pcr.predict(X_test)

    print("\nBest PCR Components:", best_pcr)
    print("PCR Test MSE:", mean_squared_error(y_test, y_pred_pcr))
    print("PCR Test R2:", r2_score(y_test, y_pred_pcr))

    # ------------------------------------------------------------
    # PLS with CV
    # ------------------------------------------------------------

    max_pls_components = min(10, X_train.shape[1])
    pls_component_values = list(range(1, max_pls_components + 1))

    best_pls, pls_results = select_best_pls_components(
        X_train, y_train, pls_component_values, k=5, random_state=42
    )

    pls = PLSScratch(n_components=best_pls["n_components"])
    pls.fit(X_train, y_train)

    y_pred_pls = pls.predict(X_test)

    print("\nBest PLS Components:", best_pls)
    print("PLS Test MSE:", mean_squared_error(y_test, y_pred_pls))
    print("PLS Test R2:", r2_score(y_test, y_pred_pls))

    # ------------------------------------------------------------
    # Subset Selection Examples
    # ------------------------------------------------------------

    print("\n" + "=" * 70)
    print("Subset Selection Examples on first 8 features")
    print("=" * 70)

    X_small = X_train[:, :8]

    best_subset_results = best_subset_selection(X_small, y_train, max_features=4)

    print("\nBest Subset Results:")
    for result in best_subset_results:
        print(
            "k:",
            result["num_features"],
            "features:",
            result["features"],
            "RSS:",
            round(result["rss"], 4),
            "R2:",
            round(result["r2"], 4)
        )

    forward_results = forward_stepwise_selection(X_small, y_train, max_features=4)

    print("\nForward Stepwise Results:")
    for result in forward_results:
        print(
            "k:",
            result["num_features"],
            "features:",
            result["features"],
            "RSS:",
            round(result["rss"], 4),
            "R2:",
            round(result["r2"], 4)
        )

    backward_results = backward_stepwise_selection(X_small, y_train, min_features=4)

    print("\nBackward Stepwise Results:")
    for result in backward_results:
        print(
            "k:",
            result["num_features"],
            "features:",
            result["features"],
            "RSS:",
            round(result["rss"], 4),
            "R2:",
            round(result["r2"], 4)
        )