"""
Entry point for Task 2 – Thali detection + Khana classification.

Usage:
    uv run run.py --model path/to/model.pt --images data/images/ --output outputs/

Important:
    --model-name  defaults to convnext_small (your Task 1 model)
    --img-size    defaults to 320 (your Task 1 fine-tune resolution)
    --classes     path to a text file with one class name per line, in the SAME
                  order as your training dataset (class index 0 = line 0)
"""

import argparse
import json
from pathlib import Path

from src.classifier import KhanaClassifier
from src.detector import ThaliDetector
from src.pipeline import ThaliPipeline
from src.visualize import draw_detections


def load_classes(path: str) -> list[str]:
    return [
        line.strip() for line in Path(path).read_text().splitlines() if line.strip()
    ]


def parse_args():
    p = argparse.ArgumentParser(description="Thali food detection – Task 2")
    p.add_argument(
        "--model",
        required=True,
        help="Path to Task 1 ConvNeXt .pt file",
        default="models/best_detection.pt",
    )
    p.add_argument(
        "--classes",
        required=True,
        help="Path to classes.txt (one class per line)",
        default="data/classes.txt",
    )
    p.add_argument(
        "--images",
        default="data/images",
        help="Folder of thali images",
        default="data/images",
    )
    p.add_argument(
        "--output",
        default="outputs",
        help="Output folder for annotated images",
        default="outputs",
    )
    p.add_argument(
        "--model-name",
        default="convnext_small.fb_in22k_ft_in1k",
        help="timm model name matching Task 1 architecture",
    )
    p.add_argument("--img-size", type=int, default=320, help="Classifier input size")
    p.add_argument(
        "--det-conf",
        type=float,
        default=0.25,
        help="YOLO detection confidence threshold",
    )
    p.add_argument(
        "--yolo",
        default="models/thali_detector.pt",
        help="YOLO weights to use for detection",
    )
    return p.parse_args()


def main():
    args = parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    class_names = load_classes(args.classes)
    print(f"Loaded {len(class_names)} Khana classes")

    print(f"Loading classifier: {args.model_name} from {args.model}")
    classifier = KhanaClassifier(
        model_path=args.model,
        class_names=class_names,
        model_name=args.model_name,
        img_size=args.img_size,
    )

    print(f"Loading YOLO detector: {args.yolo}")
    detector = ThaliDetector(weights=args.yolo, conf=args.det_conf)

    pipeline = ThaliPipeline(classifier=classifier, detector=detector)

    image_paths = sorted(Path(args.images).glob("*"))
    image_paths = [
        p for p in image_paths if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]
    print(f"\nProcessing {len(image_paths)} images ...\n")

    all_results = {}
    for img_path in image_paths:
        print(f"[{img_path.name}]")
        detections = pipeline.run(img_path)

        for det in detections:
            print(f"  {det['label']:<30} conf={det['conf']:.3f}  bbox={det['bbox']}")

        out_path = output_dir / img_path.name
        draw_detections(img_path, detections, out_path)

        all_results[img_path.name] = [
            {"label": d["label"], "conf": round(d["conf"], 4), "bbox": d["bbox"]}
            for d in detections
        ]

    # Save JSON for precision/recall evaluation
    results_path = output_dir / "predictions.json"
    results_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nPredictions saved to {results_path}")


if __name__ == "__main__":
    main()
