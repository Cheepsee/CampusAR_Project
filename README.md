# CampusAR

## 使用步骤
1) 使用DroidCam或其他ip camera。
2) Windows 端能看到对应的虚拟摄像头（例如 “Camo Studio Virtual Camera”）。
3) 在本目录运行：
```bash
cd server
python -m venv .venv
#若使用conda，请在conda内创建环境
. .venv/Scripts/activate   # PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
set CAM_DEVICE=video=Camo
# 或 set CAM_INDEX=0,需要获取你的电脑虚拟摄像头索引或名称如果下载了ffmpeg可通过
ffmpeg -hide_banner -f dshow -list_devices true -i dummy
#查看你的虚拟相机名称
python python WebSocket2Unity.py
```
按 **c** 依次点击地图四角（TL,TR,BR,BL），脚本会生成/覆盖 `homography.json`。

Unity 端继续使用你现有的 `WSClient.cs` 接收 JSON 并做可视化。
