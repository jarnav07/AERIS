"""A small CPU-friendly PyTorch MLP regressor with a scikit-learn-compatible API.

The sklearn ``MLPRegressor`` is a fair baseline but converges slowly on
hundreds-of-features geometry inputs because its per-epoch Python overhead
dominates on CPU. This module provides the same role with vectorised PyTorch
training: a fully converged MLP (100+ epochs) on a few thousand airfoil rows
finishes in tens of seconds instead of many minutes, so it can be compared
honestly against the tree and linear baselines inside a bounded job budget.

The class deliberately mirrors the parts of the sklearn estimator API that the
rest of the pipeline uses: ``fit(X, y)``, ``predict(X)``, and the optional
``loss_curve_`` / ``validation_scores_`` attributes that ``train_from_kulfan_csv``
persists as history JSON. Inputs and targets are expected to be already
standardised by the pipeline's ``FeaturePreprocessor``; this module does no
scaling of its own.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class TorchMLPRegressor:
    """Multi-layer perceptron regressor with Adam, early stopping, and joblib persistence."""

    def __init__(
        self,
        hidden_layers: tuple[int, ...] = (128, 64, 32),
        max_epochs: int = 250,
        batch_size: int = 128,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        patience: int = 20,
        validation_fraction: float = 0.15,
        early_stopping: bool = True,
        seed: int = 42,
        num_threads: int | None = 4,
    ) -> None:
        if not hidden_layers:
            raise ValueError("hidden_layers must not be empty")
        if batch_size < 1 or max_epochs < 1:
            raise ValueError("batch_size and max_epochs must be positive")
        self.hidden_layers = tuple(int(size) for size in hidden_layers)
        self.max_epochs = int(max_epochs)
        self.batch_size = int(batch_size)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.patience = int(patience)
        self.validation_fraction = float(validation_fraction)
        self.early_stopping = bool(early_stopping)
        self.seed = int(seed)
        # On many-core machines torch's default thread pool oversubscribes
        # small CPU matmuls (each batch here is tiny), so a few threads are
        # dramatically faster than all cores. None leaves torch's default.
        self.num_threads = num_threads
        self._model: nn.Sequential | None = None
        self._state_dict: dict[str, np.ndarray] | None = None
        self.n_features_in_: int | None = None
        self.n_outputs_: int | None = None
        self.n_iter_: int = 0
        self.loss_curve_: list[float] = []
        self.validation_scores_: list[float] = []
        self.best_validation_score_: float = -np.inf

    def _build_model(self) -> nn.Sequential:
        layers: list[nn.Module] = []
        in_features = int(self.n_features_in_ or 0)
        for size in self.hidden_layers:
            layers.append(nn.Linear(in_features, int(size)))
            layers.append(nn.ReLU())
            in_features = int(size)
        layers.append(nn.Linear(in_features, int(self.n_outputs_ or 1)))
        return nn.Sequential(*layers)

    def fit(self, x: np.ndarray, y: np.ndarray) -> "TorchMLPRegressor":
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        if x.ndim != 2 or y.ndim != 2 or len(x) != len(y):
            raise ValueError("x and y must be 2-D matrices with the same number of rows")
        rng = np.random.default_rng(self.seed)
        torch.manual_seed(self.seed)
        if self.num_threads is not None:
            torch.set_num_threads(int(self.num_threads))
        self.n_features_in_ = int(x.shape[1])
        self.n_outputs_ = int(y.shape[1])
        self._model = self._build_model()

        train_x, train_y = x, y
        val_x: np.ndarray | None = None
        val_y: np.ndarray | None = None
        if self.early_stopping and self.validation_fraction > 0 and len(x) >= 10:
            n_val = max(1, int(round(len(x) * self.validation_fraction)))
            indices = rng.permutation(len(x))
            val_idx, train_idx = indices[:n_val], indices[n_val:]
            train_x, train_y = x[train_idx], y[train_idx]
            val_x, val_y = x[val_idx], y[val_idx]

        model = self._model
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        loss_fn = nn.MSELoss()
        best_state: dict[str, np.ndarray] | None = None
        best_score = -np.inf
        epochs_without_improvement = 0

        train_tensor = torch.from_numpy(train_x)
        train_targets = torch.from_numpy(train_y)
        val_tensor = torch.from_numpy(val_x) if val_x is not None else None
        val_targets = torch.from_numpy(val_y) if val_y is not None else None

        for epoch in range(self.max_epochs):
            model.train()
            permutation = rng.permutation(len(train_tensor))
            epoch_losses: list[float] = []
            for start in range(0, len(train_tensor), self.batch_size):
                batch_idx = permutation[start : start + self.batch_size]
                batch_x = train_tensor[batch_idx]
                batch_y = train_targets[batch_idx]
                optimizer.zero_grad()
                prediction = model(batch_x)
                loss = loss_fn(prediction, batch_y)
                loss.backward()
                optimizer.step()
                epoch_losses.append(float(loss.detach()))
            self.loss_curve_.append(float(np.mean(epoch_losses)))

            if val_tensor is not None and val_targets is not None:
                model.eval()
                with torch.no_grad():
                    val_pred = model(val_tensor).numpy()
                val_score = self._r2(val_y, val_pred)
                self.validation_scores_.append(float(val_score))
                self.n_iter_ = epoch + 1
                if val_score > best_score:
                    best_score = val_score
                    best_state = {name: tensor.detach().cpu().numpy() for name, tensor in model.state_dict().items()}
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                    if self.early_stopping and epochs_without_improvement >= self.patience:
                        break
            else:
                self.n_iter_ = epoch + 1

        if best_state is not None:
            self._state_dict = best_state
            self.best_validation_score_ = float(best_score)
        return self

    @staticmethod
    def _r2(actual: np.ndarray, predicted: np.ndarray) -> float:
        if actual.shape[0] < 2:
            return 0.0
        ss_res = float(np.sum((actual - predicted) ** 2))
        ss_tot = float(np.sum((actual - np.mean(actual, axis=0)) ** 2))
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("TorchMLPRegressor has not been fitted")
        self._model.eval()
        x = np.asarray(x, dtype=np.float32)
        with torch.no_grad():
            output = self._model(torch.from_numpy(x))
        return output.numpy()

    def _apply_state(self, state: dict[str, np.ndarray] | None) -> None:
        if state is None:
            return
        self._model = self._build_model()
        self._model.load_state_dict({name: torch.from_numpy(value) for name, value in state.items()})

    def __getstate__(self) -> dict[str, object]:
        state = self.__dict__.copy()
        state["_model"] = None
        state["_state_dict"] = self._state_dict
        return state

    def __setstate__(self, state: dict[str, object]) -> None:
        self.__dict__.update(state)
        self._apply_state(self._state_dict)
