import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

class PredictionDatasetNCL(Dataset):
    def __init__(self, dyn_path, window_length=48, stride=1, augment_fn=None):
        """
        Dataset pour l'entraînement self-supervised NCL.
        
        Args:
            dyn_path (str): chemin vers dyn.parquet
            window_length (int): longueur des fenêtres temporelles
            stride (int): décalage entre les fenêtres (par défaut 1h)
            augment_fn (callable): fonction retournant deux vues augmentées (x1, x2)
        """
        self.window_length = window_length
        self.stride = stride
        self.augment_fn = augment_fn

        # Lire dyn.parquet
        dyn = pd.read_parquet(dyn_path)
        dyn["time"] = dyn["time"].dt.total_seconds() // 3600  # timedelta → heures entières
        dyn = dyn.sort_values(["stay_id", "time"])

        # Sélection des colonnes dynamiques
        self.feature_cols = [col for col in dyn.columns if col not in ["stay_id", "time"]]

        # Normalisation par feature (z-score)
        self.mean = dyn[self.feature_cols].mean()
        self.std = dyn[self.feature_cols].std()
        dyn[self.feature_cols] = (dyn[self.feature_cols] - self.mean) / self.std

        # Création des fenêtres glissantes
        self.windows = []
        for stay_id, group in dyn.groupby("stay_id"):
            group = group.reset_index(drop=True)
            for i in range(self.window_length, len(group) + 1, self.stride):
                window = group.iloc[i - self.window_length:i]
                if window.shape[0] == self.window_length:
                    x = window[self.feature_cols].values.astype(np.float32)
                    time_index = int(window["time"].iloc[-1])
                    self.windows.append((x, (int(stay_id), time_index)))  # key = (stay_id, t)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        x, key = self.windows[idx]  # x shape: (L, D)
        x = torch.tensor(x)         # conversion en tensor PyTorch

        # Appliquer les augmentations pour produire x1 et x2
        if self.augment_fn:
            x1, x2 = self.augment_fn(x)
        else:
            x1, x2 = x, x  # fallback (pas d’augmentation)

        key_tensor = torch.tensor(key, dtype=torch.int32)  # (2,) : (stay_id, time)
        return x1, x2, key_tensor
