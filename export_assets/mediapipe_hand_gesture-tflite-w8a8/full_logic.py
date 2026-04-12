import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import requests
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


class RoverController:
    def __init__(self, rover_ip: str, refresh_interval: float = 0.25):
        self.base_url = f"http://{rover_ip}/js?json="
        self.refresh_interval = refresh_interval
        self.last_sent = (None, None)
        self.last_send_time = 0.0
        self.last_error_time = 0.0

    def send_speed(self, left: float, right: float, force: bool = False):
        left = round(float(left), 3)
        right = round(float(right), 3)
        now = time.perf_counter()

        same_cmd = (left, right) == self.last_sent
        sent_recently = (now - self.last_send_time) < self.refresh_interval

        if not force and same_cmd and sent_recently:
            return

        cmd = {"T": 1, "L": left, "R": right}
        url = self.base_url + json.dumps(cmd, separators=(",", ":"))

        try:
            requests.get(url, timeout=0.25)
        except requests.RequestException as e:
            if now - self.last_error_time > 2.0:
                print(f"Rover send failed: {e}")
                self.last_error_time = now

        self.last_sent = (left, right)
        self.last_send_time = now


def classify_control(gestures, hands):
    """
    Returns:
        control_token:
            STOP_NOW, LATCH_FORWARD, REVERSE, TURN_LEFT, TURN_RIGHT,
            UNKNOWN, AMBIGUOUS, None
        display_text: overlay text
    """
    requested = set()
    display_labels = []

    for i, g in enumerate(gestures):
        if not g or g == "None":
            continue

        is_right = bool(hands[i]) if i < len(hands) else False
        hand_name = "Right" if is_right else "Left"
        display_labels.append(f"{hand_name}: {g}")

        if g == "Closed_Fist":
            requested.add("STOP_NOW")
        elif g == "Thumb_Up":
            requested.add("LATCH_FORWARD")
        elif g == "Thumb_Down":
            requested.add("REVERSE")
        elif g == "Open_Palm":
            requested.add("TURN_RIGHT" if is_right else "TURN_LEFT")
        else:
            requested.add("UNKNOWN")

    if not display_labels:
        return None, "No hand detected"

    if "STOP_NOW" in requested:
        return "STOP_NOW", ", ".join(display_labels)

    non_unknown = {x for x in requested if x != "UNKNOWN"}

    if len(non_unknown) > 1:
        return "AMBIGUOUS", ", ".join(display_labels)

    if len(non_unknown) == 1:
        return next(iter(non_unknown)), ", ".join(display_labels)

    return "UNKNOWN", ", ".join(display_labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--use-qnn", action="store_true")
    parser.add_argument("--rover-ip", type=str, default="192.168.4.1")
    args = parser.parse_args()

    base = Path(".")
    palm_path = str(base / "PalmDetector.tflite")
    landmark_path = str(base / "HandLandmarkDetector.tflite")
    gesture_path = str(base / "CannedGestureClassifier.tflite")

    palm_detector = PalmDetectorTFLite(palm_path, use_qnn=args.use_qnn)
    landmark_detector = HandLandmarkDetectorTFLite(landmark_path, use_qnn=args.use_qnn)
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

    rover = RoverController(args.rover_ip)

    forward_speed = 0.18
    reverse_speed = 0.12
    stationary_turn_speed = 0.30
    moving_turn_inner_scale = 0.10

    forward_latched = False

    candidate_control = None
    candidate_text = "No hand detected"
    candidate_count = 0

    stable_control = None
    stable_text = "No hand detected"

    confirm_frames = 3
    clear_frames = 2

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Could not open camera")
        sys.exit(1)

    rover.send_speed(0.0, 0.0, force=True)

    print("Press q to quit.")

    fps = 0.0
    prev_time = time.perf_counter()

    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
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

            observed_control, observed_text = classify_control(gestures, hands)

            if observed_control == "STOP_NOW":
                forward_latched = False
                stable_control = "STOP_NOW"
                stable_text = observed_text
                left_speed = 0.0
                right_speed = 0.0
                rover_state = "STOPPED (FIST)"
                rover.send_speed(left_speed, right_speed, force=True)
            else:
                if observed_control == candidate_control:
                    candidate_count += 1
                else:
                    candidate_control = observed_control
                    candidate_text = observed_text
                    candidate_count = 1

                if candidate_control is None:
                    if candidate_count >= clear_frames:
                        stable_control = None
                        stable_text = "No hand detected"
                else:
                    if candidate_count >= confirm_frames:
                        stable_control = candidate_control
                        stable_text = candidate_text

                if stable_control == "LATCH_FORWARD":
                    forward_latched = True

                if stable_control == "TURN_LEFT":
                    if forward_latched:
                        left_speed = forward_speed * moving_turn_inner_scale
                        right_speed = forward_speed
                        rover_state = "TURN LEFT (MOVING)"
                    else:
                        left_speed = -stationary_turn_speed
                        right_speed = stationary_turn_speed
                        rover_state = "TURN LEFT (STATIONARY)"

                elif stable_control == "TURN_RIGHT":
                    if forward_latched:
                        left_speed = forward_speed
                        right_speed = forward_speed * moving_turn_inner_scale
                        rover_state = "TURN RIGHT (MOVING)"
                    else:
                        left_speed = stationary_turn_speed
                        right_speed = -stationary_turn_speed
                        rover_state = "TURN RIGHT (STATIONARY)"

                elif stable_control == "REVERSE":
                    left_speed = -reverse_speed
                    right_speed = -reverse_speed
                    rover_state = "REVERSE"

                elif stable_control == "AMBIGUOUS":
                    left_speed = 0.0
                    right_speed = 0.0
                    rover_state = "STOP (AMBIGUOUS)"

                elif stable_control == "UNKNOWN":
                    left_speed = 0.0
                    right_speed = 0.0
                    rover_state = "STOP (UNKNOWN)"

                else:
                    if forward_latched:
                        left_speed = forward_speed
                        right_speed = forward_speed
                        rover_state = "FORWARD (LATCHED)"
                    else:
                        left_speed = 0.0
                        right_speed = 0.0
                        rover_state = "STOPPED"

                rover.send_speed(left_speed, right_speed)

            now = time.perf_counter()
            dt = now - prev_time
            prev_time = now
            if dt > 0:
                inst_fps = 1.0 / dt
                fps = inst_fps if fps == 0.0 else (0.9 * fps + 0.1 * inst_fps)

            latency_ms = (t1 - t0) * 1000.0

            cv2.putText(
                frame_bgr,
                f"Gesture: {stable_text}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame_bgr,
                f"State: {rover_state}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame_bgr,
                f"Forward latch: {'ON' if forward_latched else 'OFF'}",
                (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 220, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame_bgr,
                f"Cmd L/R: {left_speed:.2f}, {right_speed:.2f}",
                (20, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (200, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame_bgr,
                f"Latency: {latency_ms:.1f} ms",
                (20, 175),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame_bgr,
                f"FPS: {fps:.1f}",
                (20, 210),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("Live Hand Gesture Inference", frame_bgr)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        try:
            rover.send_speed(0.0, 0.0, force=True)
            time.sleep(0.1)
        except Exception:
            pass

        cap.release()
        cv2.destroyAllWindows()
        sys.stdout.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
