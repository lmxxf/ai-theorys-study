@echo off
set "HTTPS_PROXY=http://127.0.0.1:17899"
set "HTTP_PROXY=http://127.0.0.1:17899"
set "NO_PROXY=127.0.0.1,localhost"
cd /d C:\Strata-Lab\Strata
C:\Strata-Lab\Python\python.exe -m venv .rocm-win
.rocm-win\Scripts\python.exe -m pip install --disable-pip-version-check --index-url https://nightly.repo.amd.com/rocm/whl-next/ "rocm[libraries,devel,device-gfx1201]==10.2.0a20260930"
if errorlevel 1 exit /b 1
.rocm-win\Scripts\rocm-sdk.exe init
if errorlevel 1 exit /b 1
> .rocm-win\strata-rocm.txt echo 10.2.0a20260930 libraries,devel,device-gfx1201
