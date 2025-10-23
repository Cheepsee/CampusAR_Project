import subprocess, sys

COMMON_NAMES = [
    # iVCam
    "video=e2eSoft iVCam",
    "video=iVCam",
    "video=e2eSoft iVCam #2",
    # Camo
    "video=Camo",
    "video=Camo Studio Virtual Camera",
    # Others
    "video=EpocCam Camera",
    "video=OBS Virtual Camera",
    # DroidCam
    "video=DroidCam Source",
    "video=DroidCam Source 2",
    "video=DroidCam Source 3",
]

def list_with_ffmpeg():
    try:
        # ffmpeg prints device list to stderr
        proc = subprocess.run(
            ["ffmpeg", "-hide_banner", "-list_devices", "true", "-f", "dshow", "-i", "dummy"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False
        )
        out = proc.stderr
        print("=== DirectShow devices via ffmpeg ===")
        for line in out.splitlines():
            if "DirectShow video devices" in line or "Alternative name" in line or "]  " in line:
                print(line)
        print("=====================================")
        return True
    except FileNotFoundError:
        return False

def probe_common_with_opencv():
    import cv2
    print("=== Probing common names with OpenCV ===")
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF]
    for name in COMMON_NAMES:
        ok_any = False
        for be in backends:
            cap = cv2.VideoCapture(name, be)
            if cap.isOpened():
                w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                fps = cap.get(cv2.CAP_PROP_FPS)
                print(f"OK: {name} (backend={be}) {int(w)}x{int(h)} @{fps:.1f}fps")
                cap.release()
                ok_any = True
                break
        if not ok_any:
            print(f"--: {name}")
    print("======================================")

if __name__ == "__main__":
    used_ffmpeg = list_with_ffmpeg()
    if not used_ffmpeg:
        print("ffmpeg not found, falling back to OpenCV probing...\n")
        probe_common_with_opencv()

