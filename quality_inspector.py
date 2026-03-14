#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
from ultralytics import YOLO


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

CLASS_METADATA = {
    0: {
        "quality_label": "Minor Defect",
        "issue": "Minor surface anomaly detected",
        "severity_rank": 1,
        "draw_color": (0, 215, 255),
    },
    1: {
        "quality_label": "Major Defect",
        "issue": "Major structural defect detected",
        "severity_rank": 2,
        "draw_color": (0, 0, 255),
    },
    2: {
        "quality_label": "Major Defect",
        "issue": "Severe uncategorized defect detected",
        "severity_rank": 2,
        "draw_color": (0, 0, 255),
    },
}

GOOD_METADATA = {
    "quality_label": "Good Quality",
    "issue": "No visible defects detected",
    "severity_rank": 0,
    "draw_color": (0, 180, 0),
}


@dataclass
class Detection:
    class_id: int
    class_name: str
    raw_model_class_name: str
    mapped_quality: str
    issue: str
    confidence: float
    bbox_xyxy: list[float]


@dataclass
class ImageInspection:
    image_path: str
    inspection_result: str
    quality_label: str
    confidence: float
    issues: list[str]
    detections: list[Detection]
    annotated_image: str


@dataclass
class ProductInspection:
    product_id: str
    inspection_result: str
    quality_label: str
    confidence: float
    issues: list[str]
    report: str
    images: list[ImageInspection]
    generated_at: str
    weights_path: str


class QualityInspector:
    def __init__(self, weights_path: str, conf_threshold: float = 0.25, raw_conf_floor: float = 0.05):
        self.model = YOLO(weights_path)
        self.weights_path = str(Path(weights_path).resolve())
        self.conf_threshold = conf_threshold
        self.raw_conf_floor = min(raw_conf_floor, conf_threshold)
        self._report_generator: Any | None = None

    def inspect(self, image_paths: list[str], product_id: str, output_root: str, use_llm: bool = False) -> ProductInspection:
        resolved_paths = self._resolve_inputs(image_paths)
        if not resolved_paths:
            raise ValueError("No supported image files were found in the provided paths.")

        run_dir = self._build_run_dir(output_root, product_id)
        annotated_dir = run_dir / "annotated"
        annotated_dir.mkdir(parents=True, exist_ok=True)

        predictions = self.model.predict(
            source=resolved_paths,
            conf=self.raw_conf_floor,
            verbose=False,
        )

        image_reports: list[ImageInspection] = []
        for image_path, prediction in zip(resolved_paths, predictions):
            detections = self._extract_detections(prediction)
            accepted = [det for det in detections if det.confidence >= self.conf_threshold]

            if accepted:
                top_rank = max(self._metadata_for(det.class_id)["severity_rank"] for det in accepted)
                quality_label = "Major Defect" if top_rank >= 2 else "Minor Defect"
                inspection_result = "FAIL"
                confidence = max(det.confidence for det in accepted)
                issues = list(dict.fromkeys(det.issue for det in accepted))
            else:
                quality_label = GOOD_METADATA["quality_label"]
                inspection_result = "PASS"
                raw_max_conf = max((det.confidence for det in detections), default=0.0)
                confidence = max(0.55, 1.0 - raw_max_conf) if raw_max_conf else 0.95
                issues = [GOOD_METADATA["issue"]]

            annotated_path = annotated_dir / f"{Path(image_path).stem}_annotated.png"
            self._save_annotated_image(image_path, accepted, annotated_path, quality_label, inspection_result)

            image_reports.append(
                ImageInspection(
                    image_path=str(Path(image_path).resolve()),
                    inspection_result=inspection_result,
                    quality_label=quality_label,
                    confidence=round(confidence, 4),
                    issues=issues,
                    detections=accepted,
                    annotated_image=str(annotated_path.resolve()),
                )
            )

        overall = self._aggregate_product(product_id, image_reports, use_llm)
        report_path = run_dir / "report.json"
        report_path.write_text(json.dumps(asdict(overall), indent=2))
        return overall

    def _resolve_inputs(self, raw_paths: list[str]) -> list[str]:
        resolved: list[str] = []
        for raw_path in raw_paths:
            path = Path(raw_path).expanduser()
            if path.is_dir():
                for child in sorted(path.iterdir()):
                    if child.suffix.lower() in IMAGE_EXTENSIONS:
                        resolved.append(str(child.resolve()))
            elif path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                resolved.append(str(path.resolve()))
        return resolved

    def _build_run_dir(self, output_root: str, product_id: str) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_product_id = product_id.replace(" ", "_")
        run_dir = Path(output_root).expanduser().resolve() / f"{safe_product_id}_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def _metadata_for(self, class_id: int) -> dict[str, Any]:
        return CLASS_METADATA.get(class_id, CLASS_METADATA[2])

    def _extract_detections(self, prediction: Any) -> list[Detection]:
        if prediction.boxes is None:
            return []

        detections: list[Detection] = []
        for box in prediction.boxes:
            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            metadata = self._metadata_for(class_id)
            raw_class_name = str(self.model.names.get(class_id, f"class_{class_id}"))
            class_name = str(metadata["quality_label"]).lower().replace(" ", "_")
            detections.append(
                Detection(
                    class_id=class_id,
                    class_name=class_name,
                    raw_model_class_name=raw_class_name,
                    mapped_quality=str(metadata["quality_label"]),
                    issue=str(metadata["issue"]),
                    confidence=round(confidence, 4),
                    bbox_xyxy=[round(float(value), 2) for value in box.xyxy[0].tolist()],
                )
            )
        return detections

    def _save_annotated_image(
        self,
        image_path: str,
        detections: list[Detection],
        annotated_path: Path,
        quality_label: str,
        inspection_result: str,
    ) -> None:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Unable to read image: {image_path}")

        for detection in detections:
            x1, y1, x2, y2 = [int(value) for value in detection.bbox_xyxy]
            color = self._metadata_for(detection.class_id)["draw_color"]
            label = f"{detection.mapped_quality} {detection.confidence:.2f}"
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.rectangle(image, (x1, max(0, y1 - 28)), (min(x2 + 180, image.shape[1]), y1), color, -1)
            cv2.putText(image, label, (x1 + 6, max(18, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        summary_text = f"{inspection_result} | {quality_label}"
        summary_color = (0, 180, 0) if inspection_result == "PASS" else (0, 0, 255)
        cv2.rectangle(image, (0, 0), (min(520, image.shape[1]), 36), summary_color, -1)
        cv2.putText(image, summary_text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imwrite(str(annotated_path), image)

    def _aggregate_product(self, product_id: str, image_reports: list[ImageInspection], use_llm: bool) -> ProductInspection:
        top_image = max(
            image_reports,
            key=lambda report: (
                2 if report.quality_label == "Major Defect" else 1 if report.quality_label == "Minor Defect" else 0,
                report.confidence,
            ),
        )
        inspection_result = "FAIL" if any(report.inspection_result == "FAIL" for report in image_reports) else "PASS"
        quality_label = top_image.quality_label

        if inspection_result == "FAIL":
            confidence = max(report.confidence for report in image_reports if report.inspection_result == "FAIL")
        else:
            confidence = sum(report.confidence for report in image_reports) / len(image_reports)

        issues = list(dict.fromkeys(issue for report in image_reports for issue in report.issues))
        report_text = self._generate_report(product_id, image_reports, quality_label, inspection_result, confidence, issues, use_llm)

        return ProductInspection(
            product_id=product_id,
            inspection_result=inspection_result,
            quality_label=quality_label,
            confidence=round(confidence, 4),
            issues=issues,
            report=report_text,
            images=image_reports,
            generated_at=datetime.now().isoformat(timespec="seconds"),
            weights_path=self.weights_path,
        )

    def _generate_report(
        self,
        product_id: str,
        image_reports: list[ImageInspection],
        quality_label: str,
        inspection_result: str,
        confidence: float,
        issues: list[str],
        use_llm: bool,
    ) -> str:
        payload = {
            "product_id": product_id,
            "inspection_result": inspection_result,
            "quality_label": quality_label,
            "confidence": round(confidence, 4),
            "issues": issues,
            "image_count": len(image_reports),
        }

        if use_llm:
            llm_report = self._try_generate_llm_report(payload)
            if llm_report:
                return llm_report

        issue_line = issues[0]
        if inspection_result == "PASS":
            return (
                f"Inspection Result: PASS\n"
                f"Issue: {issue_line}\n"
                f"Confidence: {round(confidence * 100)}%"
            )

        return (
            f"Inspection Result: FAIL\n"
            f"Issue: {issue_line}\n"
            f"Confidence: {round(confidence * 100)}%"
        )

    def _try_generate_llm_report(self, payload: dict[str, Any]) -> str | None:
        try:
            generator = self._get_report_generator()
            prompt = (
                "Write a concise factory inspection report in 3 lines.\n"
                "Use exactly this format:\n"
                "Inspection Result: <PASS/FAIL>\n"
                "Issue: <short issue summary>\n"
                "Confidence: <percent>\n\n"
                f"Inspection payload:\n{json.dumps(payload, indent=2)}"
            )
            response = generator(prompt, max_new_tokens=64, do_sample=False)
            text = response[0]["generated_text"].strip()
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            if len(lines) < 3:
                return None
            return "\n".join(lines[:3])
        except Exception:
            return None

    def _get_report_generator(self) -> Any:
        if self._report_generator is not None:
            return self._report_generator

        from transformers import pipeline

        model_name = os.getenv("HF_REPORT_MODEL", "google/flan-t5-base")
        self._report_generator = pipeline(
            "text2text-generation",
            model=model_name,
        )
        return self._report_generator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multi-image quality inspection on product photos.")
    parser.add_argument("inputs", nargs="+", help="One or more image files or directories.")
    parser.add_argument("--weights", default="models/best.pt", help="Path to YOLO weights.")
    parser.add_argument("--conf-threshold", type=float, default=0.25, help="Confidence threshold for accepted detections.")
    parser.add_argument("--output-dir", default="outputs", help="Directory where reports and annotated images are saved.")
    parser.add_argument("--product-id", default="product", help="Identifier used in the output folder and report.")
    parser.add_argument("--use-llm", action="store_true", help="Use a local open-source Hugging Face model to generate the final 3-line report.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inspector = QualityInspector(
        weights_path=args.weights,
        conf_threshold=args.conf_threshold,
    )
    inspection = inspector.inspect(
        image_paths=args.inputs,
        product_id=args.product_id,
        output_root=args.output_dir,
        use_llm=args.use_llm,
    )
    print(inspection.report)
    print(json.dumps(asdict(inspection), indent=2))


if __name__ == "__main__":
    main()
