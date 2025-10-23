"""USB/virtual camera -> ArUco -> WebSocket -> Unity bridge.

Usage (PowerShell):
  cd server
  python -m venv .venv
  . .venv/Scripts/activate
  pip install -r requirements.txt
  set CAM_DEVICE=video=e2eSoft iVCam  # or your virtual camera name / set CAM_INDEX
  python WebSocket2Unity.py

Keys:
  c : click TL,TR,BR,BL to compute homography (writes homography.json)
  Esc : close preview window
"""

import asyncio
import json
import os
import queue
import sys
import threading
import time

import cv2
import numpy as np
import websockets

CAM_DEVICE = os.environ.get("CAM_DEVICE", "").strip()
CAM_INDEX = os.environ.get("CAM_INDEX", "").strip()
COMMON_NAMES = [

    # Other popular virtual cameras
    "video=Camo",
    "video=Camo Studio Virtual Camera",
    "video=EpocCam Camera",
    "video=OBS Virtual Camera",
    # iVCam variants first
    "video=e2eSoft iVCam",
    "video=iVCam",
    "video=e2eSoft iVCam #2",
    # DroidCam variants (least preferred)
    "video=DroidCam Source",
    "video=DroidCam Source 2",
    "video=DroidCam Source 3",
]

ARUCO_DICT = cv2.aruco.DICT_4X4_50
WS_HOST, WS_PORT = "0.0.0.0", 8765

MAP_CFG = "map_config.json"
H_FILE = "homography.json"

cap = None
frame_q = queue.Queue(maxsize=2)
H = None
dst_size = None  # (W, H)
SRC_NAME = None

payload_condition = threading.Condition()
latest_payload_json = None
latest_seq = 0


def load_map_cfg():
    """Load destination plane size from map_config.json."""
    global dst_size
    if os.path.exists(MAP_CFG):
        try:
            with open(MAP_CFG, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            dst = np.array(
                cfg.get("dst_corners", [[0, 0], [1000, 0], [1000, 700], [0, 700]]),
                dtype=np.float32,
            )
            w = float(np.linalg.norm(dst[1] - dst[0]))
            h = float(np.linalg.norm(dst[3] - dst[0]))
            dst_size = (max(1, int(round(w))), max(1, int(round(h))))
        except Exception as exc:  # pragma: no cover - just log
            print("[WARN] map_config load failed:", exc)
    if dst_size is None:
        dst_size = (1000, 700)


def save_h(Hm):
    with open(H_FILE, "w", encoding="utf-8") as f:
        json.dump({"H": Hm.tolist()}, f, indent=2)
    print("[INFO] Homography saved:", H_FILE)


def load_h():
    global H
    if os.path.exists(H_FILE):
        try:
            with open(H_FILE, "r", encoding="utf-8") as f:
                H = np.array(json.load(f)["H"], dtype=np.float64)
            print("[INFO] Homography loaded")
        except Exception as exc:
            print("[WARN] homography load failed:", exc)


def open_capture():
    """Try to open the virtual camera using various DirectShow backends."""
    global cap, SRC_NAME
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF]

    def try_open(source, backend):
        cam = cv2.VideoCapture(source, backend)
        return cam if cam.isOpened() else None

    if CAM_DEVICE:
        for backend in backends:
            print(f"[INFO] Try {CAM_DEVICE} backend={backend}")
            cam = try_open(CAM_DEVICE, backend)
            if cam:
                cap = cam
                SRC_NAME = CAM_DEVICE
                return True

    if CAM_INDEX:
        try:
            idx = int(CAM_INDEX)
            for backend in backends:
                print(f"[INFO] Try index {idx} backend={backend}")
                cam = try_open(idx, backend)
                if cam:
                    cap = cam
                    SRC_NAME = f"index:{idx}"
                    return True
        except ValueError:
            print("[WARN] Invalid CAM_INDEX, must be integer")

    for name in COMMON_NAMES:
        for backend in backends:
            print(f"[INFO] Try {name} backend={backend}")
            cam = try_open(name, backend)
            if cam:
                cap = cam
                SRC_NAME = name
                return True

    for idx in range(0, 5):
        for backend in backends:
            print(f"[INFO] Try index {idx} backend={backend}")
            cam = try_open(idx, backend)
            if cam:
                cap = cam
                SRC_NAME = f"index:{idx}"
                return True

    return False


def grab():
    if not open_capture():
        print("[ERR] No camera opened. Set CAM_DEVICE or CAM_INDEX.")
        sys.exit(1)
    print(f"[INFO] Camera opened: {SRC_NAME}")
    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.02)
            continue
        if frame_q.full():
            try:
                frame_q.get_nowait()
            except queue.Empty:
                pass
        frame_q.put(frame)


def compute_homography_interactive(frame):
    pts = []
    clone = frame.copy()
    window = "Click TL,TR,BR,BL then ENTER"
    cv2.imshow(window, clone)

    def on_mouse(event, x, y, _flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
            pts.append((x, y))
            cv2.circle(param, (x, y), 6, (0, 255, 0), -1)
            cv2.imshow(window, param)

    cv2.setMouseCallback(window, on_mouse, clone)
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key in (13, 10):  # Enter
            break
        if key == 27:  # Esc cancels
            pts.clear()
            break

    cv2.destroyWindow(window)
    if len(pts) != 4:
        print("[WARN] Need exactly 4 points (TL,TR,BR,BL)")
        return None

    TL, TR, BR, BL = pts
    src = np.array([TL, TR, BR, BL], dtype=np.float32)
    if dst_size is None:
        W, Hout = 1000, 700
    else:
        W, Hout = dst_size
    dst = np.array([[0, 0], [W, 0], [W, Hout], [0, Hout]], dtype=np.float32)
    Hm, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    return Hm, (W, Hout)


def processing_loop():
    global H, dst_size, latest_payload_json, latest_seq
    print("[PROC] Processing loop started")
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    try:
        params = cv2.aruco.DetectorParameters()
    except AttributeError:
        params = cv2.aruco.DetectorParameters_create()
    try:
        detector = cv2.aruco.ArucoDetector(aruco_dict, params)
    except AttributeError:
        detector = None

    while True:
        frame = frame_q.get()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if detector is not None:
            corners, ids, _ = detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=params)

        display = frame.copy()
        markers = []
        if ids is not None:
            for i, cid in enumerate(ids.flatten()):
                pts = corners[i].reshape(-1, 2)
                cx, cy = pts.mean(axis=0)
                uv = None
                if H is not None:
                    p = cv2.perspectiveTransform(np.array([[[cx, cy]]], dtype=np.float32), H)[0][0]
                    W, Hh = dst_size
                    if W and Hh:
                        uv = [float(p[0] / W), float(p[1] / Hh)]
                markers.append({"id": int(cid), "px": [float(cx), float(cy)], "uv": uv})
                cv2.polylines(display, [pts.astype(int)], True, (0, 255, 0), 2)
                cv2.putText(
                    display,
                    f"ID {int(cid)}",
                    (int(cx) + 6, int(cy) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                )

        W, Hh = dst_size if dst_size else (1000, 700)
        payload = {
            "markers": markers,
            "ts": time.time(),
            "map": {"size": [W, Hh]},
            "hasH": H is not None,
        }

        info = f"H:{'Y' if H is not None else 'N'}  Markers:{len(markers)}  Map:{dst_size}"
        cv2.putText(display, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow("Cam", display)

        with payload_condition:
            latest_payload_json = json.dumps(payload)
            latest_seq += 1
            payload_condition.notify_all()

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            print("[UI] ESC pressed, stopping processing loop")
            break
        if key == ord("c"):
            result = compute_homography_interactive(frame)
            if result:
                Hm, size = result
                H = Hm
                dst_size = size
                save_h(H)


async def serve(ws):
    print("[WS] connected")
    loop = asyncio.get_running_loop()
    last_seq = 0

    def wait_for_payload():
        nonlocal last_seq
        with payload_condition:
            payload_condition.wait_for(
                lambda: latest_payload_json is not None and latest_seq != last_seq
            )
            last_seq = latest_seq
            return latest_payload_json

    try:
        while True:
            data = await loop.run_in_executor(None, wait_for_payload)
            await ws.send(data)
    except websockets.ConnectionClosed:
        print("[WS] client disconnected")
    except Exception as exc:
        print("[WS] error:", exc)


async def main():
    load_map_cfg()
    load_h()
    threading.Thread(target=grab, daemon=True).start()
    threading.Thread(target=processing_loop, daemon=True).start()
    async with websockets.serve(serve, WS_HOST, WS_PORT, max_size=2 ** 23):
        await asyncio.Future()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    finally:
        try:
            if cap is not None:
                cap.release()
        except Exception:
            pass
        cv2.destroyAllWindows()

