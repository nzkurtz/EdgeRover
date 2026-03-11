import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import time
import os
import traceback

from ai_edge_litert.interpreter import Interpreter, load_delegate
from qai_hub_models.models.mediapipe_hand_gesture.app import MediaPipeHandGestureApp

def build_delegates(use_qnn: bool):
    if not use_qnn:
        return []

    delegate_path = os.environ.get("QNN_TFLITE_DELEGATE", "libQnnTFLiteDelegate.so")
    print(f"Trying QNN delegate: {delegate_path}")

    try:
        d = load_delegate(
            delegate_path,
            options={
                "backend_type": "htp",
                # Try these only after basic load works:
                # "htp_device_id": "0",
                # "htp_performance_mode": "2",
                # "htp_precision": "1",
            },
        )
        print("QNN delegate loaded successfully")
        return [d]
    except Exception as e:
        print("QNN delegate load failed:")
        traceback.print_exc()
        raise


def quantize_array(arr: np.ndarray, detail: dict) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    scale, zp = detail.get("quantization", (0.0, 0))
    dtype = detail["dtype"]

    if scale and float(scale) != 0.0:
        q = np.rint(arr / float(scale) + int(zp))
        if dtype == np.uint8:
            return np.clip(q, 0, 255).astype(np.uint8)
        if dtype == np.int8:
            return np.clip(q, -128, 127).astype(np.int8)

    return arr.astype(dtype)


def dequantize_array(arr: np.ndarray, detail: dict) -> np.ndarray:
    scale, zp = detail.get("quantization", (0.0, 0))
    arr = np.asarray(arr)

    if scale and float(scale) != 0.0:
        return (arr.astype(np.float32) - float(zp)) * float(scale)

    return arr.astype(np.float32)


def nchw01_to_nhwc_quantized(x: torch.Tensor, detail: dict) -> np.ndarray:
    arr = x.detach().cpu().numpy().astype(np.float32)
    arr = np.transpose(arr, (0, 2, 3, 1))
    return quantize_array(arr, detail)


class PalmDetectorTFLite:
    def __init__(self, model_path: str, use_qnn: bool = False):
        self.interpreter = Interpreter(
            model_path=model_path,
            experimental_delegates=build_delegates(use_qnn),
        )
        self.interpreter.allocate_tensors()
        self.in_detail = self.interpreter.get_input_details()[0]
        self.out_details = self.interpreter.get_output_details()

    def __call__(self, x: torch.Tensor):
        t1 = time.time()

        inp = nchw01_to_nhwc_quantized(x, self.in_detail)
        self.interpreter.set_tensor(self.in_detail["index"], inp)
        self.interpreter.invoke()

        box_coords = dequantize_array(
            self.interpreter.get_tensor(self.out_details[0]["index"]),
            self.out_details[0],
        )
        box_scores = dequantize_array(
            self.interpreter.get_tensor(self.out_details[1]["index"]),
            self.out_details[1],
        )

        t2 = time.time()
        print(f"PalmDetector: {(t2 - t1) * 1000:.2f} ms")

        return torch.from_numpy(box_coords).float(), torch.from_numpy(box_scores).float()


class HandLandmarkDetectorTFLite:
    def __init__(self, model_path: str, use_qnn: bool = False):
        self.interpreter = Interpreter(
            model_path=model_path,
            experimental_delegates=build_delegates(use_qnn),
        )
        self.interpreter.allocate_tensors()
        self.in_detail = self.interpreter.get_input_details()[0]
        self.out_details = self.interpreter.get_output_details()

    def __call__(self, x: torch.Tensor):
        t1 = time.time()

        inp = nchw01_to_nhwc_quantized(x, self.in_detail)
        self.interpreter.set_tensor(self.in_detail["index"], inp)
        self.interpreter.invoke()

        landmarks = dequantize_array(
            self.interpreter.get_tensor(self.out_details[0]["index"]),
            self.out_details[0],
        )
        scores = dequantize_array(
            self.interpreter.get_tensor(self.out_details[1]["index"]),
            self.out_details[1],
        )
        lr = dequantize_array(
            self.interpreter.get_tensor(self.out_details[2]["index"]),
            self.out_details[2],
        )
        world_landmarks = dequantize_array(
            self.interpreter.get_tensor(self.out_details[3]["index"]),
            self.out_details[3],
        )

        t2 = time.time()
        print(f"HandLandmarkDetector: {(t2 - t1) * 1000:.2f} ms")

        return (
            torch.from_numpy(landmarks).float(),
            torch.from_numpy(scores).float(),
            torch.from_numpy(lr).float(),
            torch.from_numpy(world_landmarks).float(),
        )


class CannedGestureClassifierTFLite:
    def __init__(self, model_path: str, use_qnn: bool = False):
        self.interpreter = Interpreter(
            model_path=model_path,
            experimental_delegates=build_delegates(use_qnn),
        )
        self.interpreter.allocate_tensors()
        self.in_details = self.interpreter.get_input_details()
        self.out_detail = self.interpreter.get_output_details()[0]

    def __call__(self, hand: torch.Tensor, mirrored_hand: torch.Tensor):
        t1 = time.time()

        hand_np = hand.detach().cpu().numpy().astype(np.float32)
        mirrored_np = mirrored_hand.detach().cpu().numpy().astype(np.float32)

        q_hand = quantize_array(hand_np, self.in_details[0])
        q_mirror = quantize_array(mirrored_np, self.in_details[1])

        self.interpreter.set_tensor(self.in_details[0]["index"], q_hand)
        self.interpreter.set_tensor(self.in_details[1]["index"], q_mirror)
        self.interpreter.invoke()

        out = dequantize_array(
            self.interpreter.get_tensor(self.out_detail["index"]),
            self.out_detail,
        )

        t2 = time.time()
        print(f"CannedGestureClassifier: {(t2 - t1) * 1000:.2f} ms")

        return torch.from_numpy(out).float()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="images.jpg")
    parser.add_argument("--use-qnn", action="store_true")
    args = parser.parse_args()

    base = Path(".")
    palm_path = str(base / "PalmDetector.tflite")
    landmark_path = str(base / "HandLandmarkDetector.tflite")
    gesture_path = str(base / "CannedGestureClassifier.tflite")

    palm_detector = PalmDetectorTFLite(palm_path, use_qnn=args.use_qnn)
    landmark_detector = HandLandmarkDetectorTFLite(landmark_path, use_qnn=args.use_qnn)
    gesture_classifier = CannedGestureClassifierTFLite(gesture_path, use_qnn=args.use_qnn)

    anchors = torch.empty(0)
    palm_input_spec = {"image": ((1, 3, 256, 256), "float32")}
    landmark_input_spec = {"image": ((1, 3, 224, 224), "float32")}

    app = MediaPipeHandGestureApp(
        palm_detector=palm_detector,
        hand_landmark_detector=landmark_detector,
        anchors=anchors,
        hand_landmark_detector_includes_postprocessing=True,
        palm_detector_input_spec=palm_input_spec,
        landmark_detector_input_spec=landmark_input_spec,
        gesture_classifier=gesture_classifier,
        min_detector_hand_box_score=0.75,
        nms_iou_threshold=0.3,
        min_landmark_score=0.2,
    )

    img = Image.open(args.image).convert("RGB")
    img_np = np.array(img)

    t0 = time.perf_counter()
    raw = app.predict_landmarks_from_image(img_np, raw_output=True)
    (
        _batched_boxes,
        _batched_keypoints,
        _batched_rois,
        _batched_landmarks,
        batched_is_right,
        batched_gestures,
    ) = raw
    t1 = time.perf_counter()
    print(f"Total pipeline: {(t1 - t0) * 1000:.2f} ms")

    gestures = batched_gestures[0] if len(batched_gestures) > 0 else []
    hands = batched_is_right[0] if len(batched_is_right) > 0 else []

    if not gestures:
        print("No hand gesture detected.")
    else:
        for i, (g, right) in enumerate(zip(gestures, hands, strict=False), start=1):
            handedness = "Right" if right else "Left"
            print(f"Hand {i}: {handedness} - {g}")


if __name__ == "__main__":
    main()
