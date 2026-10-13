@echo off
set "PATH=C:\Strata-Lab\Python;C:\Strata-Lab\Strata\.venv\Scripts;%PATH%"
set "HTTPS_PROXY="
set "HTTP_PROXY="
set "NO_PROXY=127.0.0.1,localhost"
set "STRATA_HIP_ARCHS=gfx1201"
set "STRATA_GGML_DIR=C:\Strata-Lab\Strata\third_party\llama.cpp"
set "CMAKE_BUILD_PARALLEL_LEVEL=8"
set "ROCM_VENV=C:\Strata-Lab\Strata\.rocm-win"
set "BUILD_DIR=C:\Strata-Lab\build-speed"
set "DIST_DIR=C:\Strata-Lab\dist-speed"
cd /d C:\Strata-Lab\Strata-speed
call tools\hip\build_windows.bat
set "RC=%ERRORLEVEL%"
echo %RC% > C:\Strata-Lab\build-speed-exit.txt
exit /b %RC%
