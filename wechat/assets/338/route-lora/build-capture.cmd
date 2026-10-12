@echo off
set "PATH=C:\Strata-Lab\Python;C:\Strata-Lab\Strata\.venv\Scripts;%PATH%"
set "HTTPS_PROXY=http://127.0.0.1:17899"
set "HTTP_PROXY=http://127.0.0.1:17899"
set "NO_PROXY=127.0.0.1,localhost"
set "STRATA_HIP_ARCHS=gfx1201"
set "STRATA_GGML_DIR=C:\Strata-Lab\Strata\third_party\llama.cpp"
set "CMAKE_BUILD_PARALLEL_LEVEL=8"
set "ROCM_VENV=C:\Strata-Lab\Strata\.rocm-win"
set "BUILD_DIR=C:\Strata-Lab\build-capture"
set "DIST_DIR=C:\Strata-Lab\dist-capture"
cd /d C:\Strata-Lab\Strata-capture
call tools\hip\build_windows.bat
set "RC=%ERRORLEVEL%"
echo %RC% > C:\Strata-Lab\build-capture-exit.txt
exit /b %RC%
