from pathlib import Path

import pandas as pd
import torch
from PIL import Image


class RetinaDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path_or_df, transform=None):
        if isinstance(csv_path_or_df, (str, Path)):
            self.df = pd.read_csv(csv_path_or_df)
        else:
            self.df = csv_path_or_df.copy()
        required = {"id_code", "diagnosis", "image_path"}
        missing = required.difference(self.df.columns)
        if missing:
            raise ValueError(f"Faltan columnas requeridas en dataset: {sorted(missing)}")
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = Path(row["image_path"])
        try:
            with Image.open(image_path) as img:
                image = img.convert("RGB")
                if self.transform:
                    image = self.transform(image)
        except Exception as exc:
            raise RuntimeError(f"No se pudo cargar la imagen {image_path}: {exc}") from exc
        label = int(row["diagnosis"])
        return image, label, str(row["id_code"]), str(image_path)
