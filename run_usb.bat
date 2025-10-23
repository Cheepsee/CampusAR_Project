@echo off
setlocal
cd /d %~dp0
cd server
if not exist .venv (
  py -3 -m venv .venv
)
call .venv\Scripts\activate
pip install -r requirements.txt
REM Set one of the following:
REM set CAM_DEVICE=video=Camo Studio Virtual Camera
REM set CAM_DEVICE=video=e2eSoft iVCam
REM set CAM_DEVICE=video=EpocCam Camera
REM set CAM_INDEX=0
python capture_usb.py
pause
