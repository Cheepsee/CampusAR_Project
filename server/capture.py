
import asyncio, json, time, threading, queue, os, sys
import numpy as np
import cv2
import websockets

STREAM_URL = os.environ.get("CAM_STREAM_URL", "http://192.168.1.123:8080/video")
WS_HOST, WS_PORT = "0.0.0.0", 8765
ARUCO_DICT = cv2.aruco.DICT_4X4_50

MAP_CFG = "map_config.json"
H_FILE  = "homography.json"

cap = None
frame_q = queue.Queue(maxsize=2)
H = None
dst_size = None

def load_map_cfg():
    global dst_size
    if os.path.exists(MAP_CFG):
        cfg = json.load(open(MAP_CFG, "r", encoding="utf-8"))
        dst = np.array(cfg.get("dst_corners", [[0,0],[1000,0],[1000,700],[0,700]]), dtype=np.float32)
        w = np.linalg.norm(dst[1]-dst[0]); h = np.linalg.norm(dst[3]-dst[0])
        dst_size = (max(1,int(round(w))), max(1,int(round(h))))

def save_homography(Hm):
    json.dump({"H": Hm.tolist()}, open(H_FILE, "w", encoding="utf-8"), indent=2)
    print("[INFO] Homography saved:", H_FILE)

def load_homography():
    global H
    if os.path.exists(H_FILE):
        H = np.array(json.load(open(H_FILE, "r", encoding="utf-8"))["H"], dtype=np.float64)
        print("[INFO] Homography loaded")

def start_capture():
    global cap
    cap = cv2.VideoCapture(STREAM_URL)
    if not cap.isOpened():
        print("[ERR] Cannot open stream:", STREAM_URL); sys.exit(1)
    print("[INFO] Stream:", STREAM_URL)
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
    if dst_size is None:
        W,Hout = 1000,700
    else:
        W,Hout = dst_size
    dst = np.array([[0,0],[W,0],[W,Hout],[0,Hout]], dtype=np.float32)
    Hm,_ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    return Hm, (W,Hout)

async def ws_handler(websocket):
    print("[WS] client connected")
    try:
        while True:
            frame = frame_q.get()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
            params = cv2.aruco.DetectorParameters()
            det = cv2.aruco.ArucoDetector(dict, params)
            corners, ids, _ = det.detectMarkers(gray)

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
            msg = {"markers": payload_markers, "ts": time.time()}
            await websocket.send(json.dumps(msg))

            info = f"H:{'Y' if H is not None else 'N'} Markers:{len(payload_markers)}"
            cv2.putText(frame, info, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            cv2.imshow("Camera", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27: break
            if key == ord('c'):
                res = compute_homography_interactive(frame)
                if res:
                    Hm, size = res
                    global H, dst_size
                    H, dst_size = Hm, size
                    save_homography(H)
    except websockets.ConnectionClosed:
        print("[WS] client disconnected")

async def main():
    load_map_cfg(); load_homography()
    t = threading.Thread(target=start_capture, daemon=True); t.start()
    print(f"[INFO] WebSocket ws://{WS_HOST}:{WS_PORT}")
    async with websockets.serve(ws_handler, WS_HOST, WS_PORT, max_size=2**23):
        await asyncio.Future()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    finally:
        if cap is not None: cap.release()
        cv2.destroyAllWindows()
