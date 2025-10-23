"""USB/虚拟摄像头 → ArUco → 标定 → WebSocket → Unity
使用方法（PowerShell）：
  cd server
  python -m venv .venv
  . .venv/Scripts/activate
  pip install -r requirements.txt
  set CAM_DEVICE=video=Camo   # 或 video=Camo Studio Virtual Camera / 或 set CAM_INDEX=0
  python WebSocket2Unity.py
按键：
  c = 采集一帧并交互点击 TL,TR,BR,BL 完成标定（保存到 homography.json）
  Esc = 退出
"""

# pip install opencv-contrib-python websockets numpy
import asyncio, json, time, threading, queue
import numpy as np, cv2, websockets, os, sys

# 环境变量与默认设备名（会自动尝试）
CAM_DEVICE = os.environ.get("CAM_DEVICE", "").strip()
CAM_INDEX  = os.environ.get("CAM_INDEX", "").strip()
COMMON_NAMES = [
    # iVCam variants (try these first)
    "video=e2eSoft iVCam",
    "video=iVCam",
    "video=e2eSoft iVCam #2",
    # Other common virtual cameras
    "video=Camo",
    "video=Camo Studio Virtual Camera",
    "video=EpocCam Camera",
    "video=OBS Virtual Camera",
    # DroidCam variants (least preferred)
    "video=DroidCam Source",
    "video=DroidCam Source 2",
    "video=DroidCam Source 3",
]

ARUCO_DICT = cv2.aruco.DICT_4X4_50
WS_HOST, WS_PORT = "0.0.0.0", 8765

MAP_CFG = "map_config.json"
H_FILE  = "homography.json"

cap = None
frame_q = queue.Queue(maxsize=2)
H = None
dst_size = None  # (W,H)
SRC_NAME = None  # record which source opened

def load_map_cfg():
    """读取地图配置，得到目标面的宽高（用于 uv 归一化）。"""
    global dst_size
    if os.path.exists(MAP_CFG):
        try:
            cfg = json.load(open(MAP_CFG, "r", encoding="utf-8"))
            dst = np.array(cfg.get("dst_corners", [[0,0],[1000,0],[1000,700],[0,700]]), dtype=np.float32)
            w = np.linalg.norm(dst[1]-dst[0]); h = np.linalg.norm(dst[3]-dst[0])
            dst_size = (max(1,int(round(w))), max(1,int(round(h))))
        except Exception as e:
            print("[WARN] map_config load failed:", e)
    if dst_size is None:
        dst_size = (1000,700)

def save_h(Hm):
    with open(H_FILE, "w", encoding="utf-8") as f:
        json.dump({"H": Hm.tolist()}, f, indent=2)
    print("[INFO] Homography saved:", H_FILE)

def load_h():
    global H
    if os.path.exists(H_FILE):
        try:
            H = np.array(json.load(open(H_FILE, "r", encoding="utf-8"))["H"], dtype=np.float64)
            print("[INFO] Homography loaded")
        except Exception as e:
            print("[WARN] homography load failed:", e)

def open_capture():
    """尽力按名称/索引 + 多后端打开虚拟摄像头。"""
    global cap, SRC_NAME
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF]
    if CAM_DEVICE:
        for be in backends:
            print(f"[INFO] Try {CAM_DEVICE} backend={be}")
            c = cv2.VideoCapture(CAM_DEVICE, be)
            if c.isOpened():
                cap = c; SRC_NAME = CAM_DEVICE; return True
    if CAM_INDEX:
        try:
            idx = int(CAM_INDEX)
            for be in backends:
                print(f"[INFO] Try index {idx} backend={be}")
                c = cv2.VideoCapture(idx, be)
                if c.isOpened():
                    cap = c; SRC_NAME = f"index:{idx}"; return True
        except: pass
    for name in COMMON_NAMES:
        for be in backends:
            print(f"[INFO] Try {name} backend={be}")
            c = cv2.VideoCapture(name, be)
            if c.isOpened():
                cap = c; SRC_NAME = name; return True
    for idx in range(0,5):
        for be in backends:
            print(f"[INFO] Try index {idx} backend={be}")
            c = cv2.VideoCapture(idx, be)
            if c.isOpened():
                cap = c; SRC_NAME = f"index:{idx}"; return True
    return False

def grab():
    if not open_capture():
        print("[ERR] No camera opened. Set CAM_DEVICE or CAM_INDEX.")
        sys.exit(1)
    print(f"[INFO] Camera opened: {SRC_NAME}")
    while True:
        ok, f = cap.read()
        if not ok:
            time.sleep(0.02)
            continue
        if frame_q.full():
            try: frame_q.get_nowait()
            except: pass
        frame_q.put(f)

def calibrate(frame):
    # 交互点击：TL,TR,BR,BL
    pts = []
    img = frame.copy()
    win = "Click TL,TR,BR,BL and press ENTER"
    cv2.imshow(win, img)
    def on_mouse(e,x,y,_,p):
        if e == cv2.EVENT_LBUTTONDOWN and len(pts)<4:
            pts.append((x,y)); cv2.circle(p,(x,y),6,(0,255,0),-1); cv2.imshow(win,p)
    cv2.setMouseCallback(win,on_mouse,img)
    while True:
        k = cv2.waitKey(1)&0xFF
        if k in (13,10): break
        if k==27: pts.clear(); break
    cv2.destroyWindow(win)
    if len(pts)!=4: return None
    TL,TR,BR,BL = pts
    src = np.array([TL,TR,BR,BL], dtype=np.float32)
    W,H = dst_size
    dst = np.array([[0,0],[W,0],[W,H],[0,H]], dtype=np.float32)
    Hm,_ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    return Hm

async def serve(ws):
    global H
    print("[WS] connected")
    # Throttle sending to a fixed interval (seconds). Default 1.0s
    try:
        send_interval = float(os.environ.get("SEND_INTERVAL", "1.0"))
    except Exception:
        send_interval = 1.0
    last_sent = 0.0
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    # 兼容新旧 API
    try:
        params = cv2.aruco.DetectorParameters()
    except AttributeError:
        params = cv2.aruco.DetectorParameters_create()
    try:
        det = cv2.aruco.ArucoDetector(aruco_dict, params)
    except AttributeError:
        det = None

    start_time = time.time()
    interval = 0.02
    while True:
        f = frame_q.get()
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        if det is not None:
            corners, ids, _ = det.detectMarkers(g)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(g, aruco_dict, parameters=params)

        W,Hh = dst_size
        payload = {"markers": [], "ts": time.time(), "map": {"size": [W, Hh]}, "hasH": (H is not None)}
        if ids is not None:
            for i, cid in enumerate(ids.flatten()):
                pts = corners[i].reshape(-1,2)
                cx, cy = pts.mean(axis=0)
                uv = None
                if H is not None:
                    p = cv2.perspectiveTransform(np.array([[[cx,cy]]],np.float32), H)[0][0]
                    u, v = float(p[0]/W), float(p[1]/Hh)
                    uv = [u, v]
                payload["markers"].append({"id": int(cid), "px": [float(cx),float(cy)], "uv": uv})
                # 可视化调试
                cv2.polylines(f, [pts.astype(int)], True, (0,255,0), 2)
                cv2.putText(f, f"ID {cid}", (int(cx)+6,int(cy)-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
                
        end_time = time.time()
        if end_time - start_time > interval:
            print(end_time - start_time)
        now = time.time()
        if now - last_sent >= send_interval:
            await ws.send(json.dumps(payload))
            last_sent = now
            start_time = time.time()
        
        info = f"H:{'Y' if H is not None else 'N'}  Markers:{len(payload['markers'])}  Map:{dst_size}"
        cv2.putText(f, info, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.imshow("Cam", f)
        k = cv2.waitKey(1)&0xFF
        if k==27: break
        if k==ord('c'):
            Hm = calibrate(f)
            if Hm is not None:
                H = Hm; save_h(H)

def preview_loop():
    global H, dst_size
    print("[UI] Preview loop started")
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    try:
        params = cv2.aruco.DetectorParameters()
    except AttributeError:
        params = cv2.aruco.DetectorParameters_create()
    try:
        det = cv2.aruco.ArucoDetector(aruco_dict, params)
    except AttributeError:
        det = None
    while True:
        f = frame_q.get()
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        if det is not None:
            corners, ids, _ = det.detectMarkers(g)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(g, aruco_dict, parameters=params)
        payload_markers = []
        if ids is not None:
            for i, cid in enumerate(ids.flatten()):
                pts = corners[i].reshape(-1,2)
                cx, cy = pts.mean(axis=0)
                uv = None
                if H is not None:
                    p = cv2.perspectiveTransform(np.array([[[cx,cy]]],np.float32), H)[0][0]
                    W,Hh = dst_size
                    uv = [float(p[0]/W), float(p[1]/Hh)]
                payload_markers.append({"id": int(cid), "px": [float(cx),float(cy)], "uv": uv})
                cv2.polylines(f, [pts.astype(int)], True, (0,255,0), 2)
                cv2.putText(f, f"ID {cid}", (int(cx)+6,int(cy)-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
        info = f"H:{'Y' if H is not None else 'N'}  Markers:{len(payload_markers)}  Map:{dst_size}"
        cv2.putText(f, info, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.imshow("Cam", f)
        k = cv2.waitKey(1)&0xFF
        if k==27: break
        if k==ord('c'):
            Hm = calibrate(f)
            if Hm is not None:
                H = Hm; save_h(H)

async def main():
    load_map_cfg(); load_h()
    threading.Thread(target=grab, daemon=True).start()
    threading.Thread(target=preview_loop, daemon=True).start()
    async with websockets.serve(serve, WS_HOST, WS_PORT, max_size=2**23):
        await asyncio.Future()

if __name__ == "__main__":
    import threading, websockets, asyncio
    try:
        asyncio.run(main())
    finally:
        try:
            if cap is not None: cap.release()
        except: pass
        cv2.destroyAllWindows()
