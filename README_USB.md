# CampusAR — USB 有线摄像头 Add‑on（路线 2）

将 iPhone(USB) 作为虚拟摄像头输入到 Windows，OpenCV 读取，其他流程不变。

## 使用步骤
1) 安装 Camo/iVCam/EpocCam（iPhone 与 Windows 端）。USB 连接并“信任此电脑”。
2) Windows 端能看到对应的虚拟摄像头（例如 “Camo Studio Virtual Camera”）。
3) 在本目录运行：
```bash
cd server
python -m venv .venv
. .venv/Scripts/activate   # PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
set CAM_DEVICE=video=Camo Studio Virtual Camera   # 或 set CAM_INDEX=0
python capture_usb.py
```
按 **c** 依次点击地图四角（TL,TR,BR,BL），脚本会生成/覆盖 `homography.json`。

Unity 端继续使用你现有的 `WSClient.cs` 接收 JSON 并做可视化。
