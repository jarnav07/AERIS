"""CPU-friendly PyTorch MLP with a small scikit-learn-style interface."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class TorchMLPRegressor:
    """Multi-output MLP with Adam, validation-based early stopping and persistence."""

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
        if not 0 < validation_fraction < 1:
            raise ValueError("validation_fraction must be between 0 and 1")
        self.hidden_layers = tuple(int(size) for size in hidden_layers)
        self.max_epochs = int(max_epochs)
        self.batch_size = int(batch_size)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.patience = int(patience)
        self.validation_fraction = float(validation_fraction)
        self.early_stopping = bool(early_stopping)
        self.seed = int(seed)
        self.num_threads = num_threads
        self._model: nn.Sequential | None = None
        self._state_dict: dict[str, np.ndarray] | None = None
        self.n_features_in_: int | None = None
        self.n_outputs_: int | None = None
        self.n_iter_: int = 0
        self.loss_curve_: list[float] = []
        self.validation_scores_: list[float] = []
        self.best_validation_score_: float = float("-inf")

    def _build_model(self) -> nn.Sequential:
        layers: list[nn.Module] = []
        in_features = int(self.n_features_in_ or 0)
        for width in self.hidden_layers:
            layers.extend([nn.Linear(in_features, width), nn.ReLU()])
            in_features = width
        layers.append(nn.Linear(in_features, int(self.n_outputs_ or 1)))
        return nn.Sequential(*layers)

    def fit(self, x: np.ndarray, y: np.ndarray) -> "TorchMLPRegressor":
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        if x.ndim != 2 or y.ndim != 2 or len(x) != len(y):
            raise ValueError("x and y must be 2-D matrices with the same row count")

        rng = np.random.default_rng(self.seed)
        torch.manual_seed(self.seed)
        if self.num_threads is not None:
            torch.set_num_threads(int(self.num_threads))

        self.n_features_in_ = int(x.shape[1])
        self.n_outputs_ = int(y.shape[1])
        self._model = self._build_model()
        self._state_dict = None
        self.loss_curve_ = []
        self.validation_scores_ = []

        train_x, train_y = x, y
        val_x = val_y = None
        if self.early_stopping and len(x) >= 10:
            n_val = max(1, int(round(len(x) * self.validation_fraction)))
            indices = rng.permutation(len(x))
            val_idx, train_idx = indices[:n_val], indices[n_val:]
            train_x, train_y = x[train_idx], y[train_idx]
            val_x, val_y = x[val_idx], y[val_idx]

        model = self._model
        optimizer = torch.optim.Adam(
            model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        loss_fn = nn.MSELoss()
        best_state: dict[str, np.ndarray] | None = None
        best_score = float("-inf")
        epochs_without_improvement = 0

        train_tensor = torch.from_numpy(train_x)
        train_targets = torch.from_numpy(train_y)
        val_tensor = torch.from_numpy(val_x) if val_x is not None else None
        val_targets = torch.from_numpy(val_y) if val_y is not None else None

        for epoch in range(self.max_epochs):
            model.train()
            order = rng.permutation(len(train_tensor))
            batch_losses: list[float] = []
            for start in range(0, len(train_tensor), self.batch_size):
                batch_idx = order[start : start + self.batch_size]
                optimizer.zero_grad()
                loss = loss_fn(model(train_tensor[batch_idx]), train_targets[batch_idx])
                loss.backward()
                optimizer.step()
                batch_losses.append(float(loss.detach()))
            self.loss_curve_.append(float(np.mean(batch_losses)))
            self.n_iter_ = epoch + 1

            if val_tensor is None or val_targets is None:
                continue
            model.eval()
            with torch.no_grad():
                val_prediction = model(val_tensor).numpy()
            score = self._r2(val_y, val_prediction)
            self.validation_scores_.append(float(score))
            if score > best_score:
                best_score = score
                best_state = {
                    name: tensor.detach().cpu().numpy().copy()
                    for name, tensor in model.state_dict().items()
                }
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if self.early_stopping and epochs_without_improvement >= self.patience:
                    break

        if best_state is not None:
            self._state_dict = best_state
            self.best_validation_score_ = float(best_score)
            self._apply_state(best_state)
        return self

    @staticmethod
    def _r2(actual: np.ndarray, predicted: np.ndarray) -> float:
        actual = np.asarray(actual, dtype=float)
        predicted = np.asarray(predicted, dtype=float)
        if len(actual) < 2:
            return 0.0
        ss_res = float(np.sum((actual - predicted) ** 2))
        ss_tot = float(np.sum((actual - np.mean(actual, axis=0)) ** 2))
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("TorchMLPRegressor has not been fitted")
        x = np.asarray(x, dtype=np.float32)
        self._model.eval()
        with torch.no_grad():
            return self._model(torch.from_numpy(x)).numpy()

    def _apply_state(self, state: dict[str, np.ndarray] | None) -> None:
        if state is None:
            return
        self._model = self._build_model()
        self._model.load_state_dict(
            {name: torch.from_numpy(value) for name, value in state.items()}
        )

    def __getstate__(self) -> dict[str, object]:
        state = self.__dict__.copy()
        state["_model"] = None
        return state

    def __setstate__(self, state: dict[str, object]) -> None:
        self.__dict__.update(state)
        self._apply_state(self._state_dict)
