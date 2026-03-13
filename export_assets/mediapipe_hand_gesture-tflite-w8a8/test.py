import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from ai_edge_litert.interpreter import Interpreter, load_delegate
from qai_hub_models.models.mediapipe_hand_gesture.app import MediaPipeHandGestureApp


def build_delegates(use_qnn: bool):
    if not use_qnn:
        return []
    delegate_path = os.environ.get("QNN_TFLITE_DELEGATE", "libQnnTFLiteDelegate.so")
    print(f"Trying QNN delegate: {delegate_path}")
    d = load_delegate(delegate_path, options={"backend_type": "htp"})
    print("QNN delegate loaded successfully")
    return [d]


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

        return torch.from_numpy(out).float()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--use-qnn", action="store_true")
    args = parser.parse_args()

    base = Path(".")
    palm_path = str(base / "PalmDetector.tflite")
    landmark_path = str(base / "HandLandmarkDetector.tflite")
    gesture_path = str(base / "CannedGestureClassifier.tflite")

    palm_detector = PalmDetectorTFLite(palm_path, use_qnn=args.use_qnn)
    landmark_detector = HandLandmarkDetectorTFLite(landmark_path, use_qnn=args.use_qnn)

    # Keep gesture classifier on CPU for stability
    gesture_classifier = CannedGestureClassifierTFLite(gesture_path, use_qnn=False)

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

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Could not open camera")
        sys.exit(1)

    print("Running live gesture recognition. Press Ctrl+C to stop.")

    last_printed = None
    last_print_time = 0.0

    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                print("Failed to read frame")
                continue

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            t0 = time.perf_counter()
            raw = app.predict_landmarks_from_image(frame_rgb, raw_output=True)
            t1 = time.perf_counter()

            (
                _batched_boxes,
                _batched_keypoints,
                _batched_rois,
                _batched_landmarks,
                batched_is_right,
                batched_gestures,
            ) = raw

            gestures = batched_gestures[0] if len(batched_gestures) > 0 else []
            hands = batched_is_right[0] if len(batched_is_right) > 0 else []

            now = time.time()

            if not gestures:
                current = "No hand detected"
            else:
                g = gestures[0]
                right = hands[0] if len(hands) > 0 else False
                handedness = "Right" if right else "Left"
                current = f"{handedness} - {g}"

            # print when changed, or every 1 second
            if current != last_printed or (now - last_print_time) > 1.0:
                print(f"{current}    ({(t1 - t0)*1000:.2f} ms)")
                last_printed = current
                last_print_time = now

    except KeyboardInterrupt:
        print("\nStopping...")

    finally:
        cap.release()
        sys.stdout.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
