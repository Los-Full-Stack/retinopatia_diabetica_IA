import argparse
import json
from pathlib import Path

import pandas as pd

from config.config import CLASS_NAMES, CSV_PATH
from src.inference import RetinaPredictor


def add_ground_truth_if_available(result: dict, image_path: str) -> dict:
    if not CSV_PATH.exists():
        return result
    image_id = Path(image_path).stem
    df = pd.read_csv(CSV_PATH)
    if "id_code" not in df.columns or "diagnosis" not in df.columns:
        return result
    match = df.loc[df["id_code"].astype(str) == image_id]
    if match.empty:
        return result
    true_class_id = int(match.iloc[0]["diagnosis"])
    result["true_class_id"] = true_class_id
    result["true_class_name"] = CLASS_NAMES[true_class_id]
    result["correct"] = result["class_id"] == true_class_id
    return result


def main():
    parser = argparse.ArgumentParser(description="Prediccion individual RetinaAI")
    parser.add_argument("--image", required=True, help="Ruta de la imagen")
    parser.add_argument("--checkpoint", default=None, help="Ruta opcional del checkpoint")
    parser.add_argument(
        "--show-true",
        action="store_true",
        help="Si la imagen pertenece al dataset, muestra la etiqueta real desde train.csv",
    )
    args = parser.parse_args()
    predictor = RetinaPredictor(checkpoint_path=args.checkpoint)
    result = predictor.predict(args.image)
    if args.show_true:
        result = add_ground_truth_if_available(result, args.image)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
