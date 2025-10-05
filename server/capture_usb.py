
import os, sys, time, json, threading, queue, asyncio
import numpy as np
import cv2
import websockets

CAM_DEVICE = os.environ.get("CAM_DEVICE", "").strip()
CAM_INDEX  = os.environ.get("CAM_INDEX", "").strip()
WS_HOST, WS_PORT = "0.0.0.0", 8765
ARUCO_DICT = cv2.aruco.DICT_4X4_50

MAP_CFG = "map_config.json"
H_FILE  = "homography.json"

COMMON_NAMES = [
    "video=Camo Studio Virtual Camera",
    "video=e2eSoft iVCam",
    "video=EpocCam Camera",
    "video=USB Video",
    "video=OBS Virtual Camera"
]

cap = None
frame_q = queue.Queue(maxsize=2)
H = None
dst_size = None

def load_map_cfg():
    global dst_size
    if os.path.exists(MAP_CFG):
        try:
            cfg = json.load(open(MAP_CFG, "r", encoding="utf-8"))
            dst = np.array(cfg.get("dst_corners", [[0,0],[1000,0],[1000,700],[0,700]]), dtype=np.float32)
            w = np.linalg.norm(dst[1]-dst[0]); h = np.linalg.norm(dst[3]-dst[0])
            dst_size = (max(1,int(round(w))), max(1,int(round(h))))
        except Exception as e:
            print("[WARN] map_config load failed:", e)

def save_homography(Hm):
    json.dump({"H": Hm.tolist()}, open(H_FILE, "w", encoding="utf-8"), indent=2)
    print("[INFO] Homography saved:", H_FILE)

def load_homography():
    global H
    if os.path.exists(H_FILE):
        try:
            H = np.array(json.load(open(H_FILE, "r", encoding="utf-8"))["H"], dtype=np.float64)
            print("[INFO] Homography loaded")
        except Exception as e:
            print("[WARN] homography load failed:", e)

def open_capture():
    global cap
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF]
    if CAM_DEVICE:
        for be in backends:
            print(f"[INFO] Try {CAM_DEVICE} backend={be}")
            c = cv2.VideoCapture(CAM_DEVICE, be)
            if c.isOpened(): cap = c; return True
    if CAM_INDEX:
        idx = int(CAM_INDEX)
        for be in backends:
            print(f"[INFO] Try index {idx} backend={be}")
            c = cv2.VideoCapture(idx, be)
            if c.isOpened(): cap = c; return True
    for name in COMMON_NAMES:
        for be in backends:
            print(f"[INFO] Try {name} backend={be}")
            c = cv2.VideoCapture(name, be)
            if c.isOpened(): cap = c; return True
    for idx in range(0,5):
        for be in backends:
            print(f"[INFO] Try index {idx} backend={be}")
            c = cv2.VideoCapture(idx, be)
            if c.isOpened(): cap = c; return True
    return False

def start_capture():
    global cap
    if not open_capture():
        print("[ERR] No camera opened. Set CAM_DEVICE or CAM_INDEX."); sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    print("[INFO] Camera opened")
    while True:
        ok, frame = cap.read()
        if not ok: time.sleep(0.02); continue
        if frame_q.full():
            try: frame_q.get_nowait()
            except: pass
        frame_q.put(frame)

def compute_homography_interactive(frame):
    pts = []
    clone = frame.copy()
    win = "Click TL,TR,BR,BL then ENTER"
    cv2.imshow(win, clone)
    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
            pts.append((x,y))
            cv2.circle(param, (x,y), 6, (0,255,0), -1)
            cv2.imshow(win, param)
    cv2.setMouseCallback(win, on_mouse, clone)
    while True:
        k = cv2.waitKey(1) & 0xFF
        if k in (13,10): break
        if k == 27: pts.clear(); break
    cv2.destroyWindow(win)
    if len(pts) != 4:
        print("[WARN] need 4 points"); return None
    TL,TR,BR,BL = pts
    src = np.array([TL,TR,BR,BL], dtype=np.float32)
    if dst_size is None: W,Hout = 1000,700
    else: W,Hout = dst_size
    dst = np.array([[0,0],[W,0],[W,Hout],[0,Hout]], dtype=np.float32)
    Hm,_ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    return Hm, (W,Hout)

def preview_loop():
    global H, dst_size
    print("[UI] Preview loop started")
    while True:
        frame = frame_q.get()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
        try:
            params = cv2.aruco.DetectorParameters()
        except AttributeError:
            params = cv2.aruco.DetectorParameters_create()
        try:
            det = cv2.aruco.ArucoDetector(aruco_dict, params)
            corners, ids, _ = det.detectMarkers(gray)
        except AttributeError:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=params)

        payload_markers = []
        if ids is not None:
            for i, cid in enumerate(ids.flatten()):
                pts = corners[i].reshape(-1,2)
                cx, cy = pts.mean(axis=0)
                uv = None
                if H is not None:
                    p = np.array([[[cx, cy]]], dtype=np.float32)
                    p = cv2.perspectiveTransform(p, H)[0][0]
                    if dst_size:
                        W,Hout = dst_size
                        uv = [float(p[0]/W), float(p[1]/Hout)]
                    else:
                        uv = [float(p[0]), float(p[1])]
                payload_markers.append({"id": int(cid), "px": [float(cx),float(cy)], "uv": uv})
                cv2.polylines(frame, [pts.astype(int)], True, (0,255,0), 2)
                cv2.putText(frame, f"ID {cid}", (int(cx)+6,int(cy)-6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

        info = f"H:{'Y' if H is not None else 'N'} Markers:{len(payload_markers)}"
        cv2.putText(frame, info, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.imshow("USB Camera", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        if key == ord('c'):
            res = compute_homography_interactive(frame)
            if res:
                Hm, size = res
                H, dst_size = Hm, size
                save_homography(H)

async def ws_handler(websocket):
    global H, dst_size
    print("[WS] client connected")
    try:
        while True:
            frame = frame_q.get()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
            try:
                params = cv2.aruco.DetectorParameters()
            except AttributeError:
                params = cv2.aruco.DetectorParameters_create()
            try:
                det = cv2.aruco.ArucoDetector(aruco_dict, params)
                corners, ids, _ = det.detectMarkers(gray)
            except AttributeError:
                corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=params)

            payload_markers = []
            if ids is not None:
                for i, cid in enumerate(ids.flatten()):
                    pts = corners[i].reshape(-1,2)
                    cx, cy = pts.mean(axis=0)
                    uv = None
                    if H is not None:
                        p = np.array([[[cx, cy]]], dtype=np.float32)
                        p = cv2.perspectiveTransform(p, H)[0][0]
                        if dst_size:
                            W,Hout = dst_size
                            uv = [float(p[0]/W), float(p[1]/Hout)]
                        else:
                            uv = [float(p[0]), float(p[1])]
                    payload_markers.append({"id": int(cid), "px": [float(cx),float(cy)], "uv": uv})
                    cv2.polylines(frame, [pts.astype(int)], True, (0,255,0), 2)
                    cv2.putText(frame, f"ID {cid}", (int(cx)+6,int(cy)-6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

            await websocket.send(json.dumps({"markers": payload_markers, "ts": time.time()}))

            info = f"H:{'Y' if H is not None else 'N'} Markers:{len(payload_markers)}"
            cv2.putText(frame, info, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            cv2.imshow("USB Camera", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27: break
            if key == ord('c'):
                res = compute_homography_interactive(frame)
                if res:
                    Hm, size = res
                    H, dst_size = Hm, size
                    save_homography(H)
    except websockets.ConnectionClosed:
        print("[WS] client disconnected")

async def main():
    load_map_cfg(); load_homography()
    t = threading.Thread(target=start_capture, daemon=True); t.start()
    ui = threading.Thread(target=preview_loop, daemon=True); ui.start()
    print(f"[INFO] WebSocket ws://{WS_HOST}:{WS_PORT}")
    async with websockets.serve(ws_handler, WS_HOST, WS_PORT, max_size=2**23):
        await asyncio.Future()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    finally:
        if cap is not None: cap.release()
        cv2.destroyAllWindows()
