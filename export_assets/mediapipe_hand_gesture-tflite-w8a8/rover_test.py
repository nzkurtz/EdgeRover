import time
import json
import requests

ROVER_IP = "192.168.4.1"
BASE_URL = f"http://{ROVER_IP}/js?json="

def send_cmd(cmd: dict):
    cmd_str = json.dumps(cmd, separators=(",", ":"))
    url = BASE_URL + cmd_str
    response = requests.get(url, timeout=2)
    print(f"Sent: {cmd_str}")
    print(f"Response: {response.text}")

def move(left_speed: float, right_speed: float, duration: float, interval: float = 0.5):
    """
    Send repeated speed commands for 'duration' seconds.
    Speed range is typically -0.5 to 0.5.
    Positive = forward, negative = backward.
    """
    cmd = {"T": 1, "L": left_speed, "R": right_speed}

    end_time = time.time() + duration
    while time.time() < end_time:
        send_cmd(cmd)
        time.sleep(interval)

def stop():
    send_cmd({"T": 1, "L": 0, "R": 0})

if __name__ == "__main__":
    try:
        print("Starting rover movement test...")

        # Forward
        print("\nForward")
        move(0.25, 0.25, 2.0)
        stop()
        time.sleep(1)

        # Backward
        print("\nBackward")
        move(-0.25, -0.25, 2.0)
        stop()
        time.sleep(1)

        # Turn left in place
        print("\nTurn left")
        move(-0.2, 0.2, 1.5)
        stop()
        time.sleep(1)

        # Turn right in place
        print("\nTurn right")
        move(0.2, -0.2, 1.5)
        stop()
        time.sleep(1)

        print("\nDone.")
        stop()

    except KeyboardInterrupt:
        print("\nInterrupted, stopping rover...")
        stop()
    except Exception as e:
        print(f"Error: {e}")
        try:
            stop()
        except:
            pass
