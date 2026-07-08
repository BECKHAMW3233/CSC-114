@echo off
:: ============================================================================
:: 01_install_cuda.bat
:: CUDA + cuDNN Installation Script for RTX 4080 + TensorFlow 2.16+
:: Run this FIRST before anything else — as Administrator
:: Right-click this file → "Run as administrator"
:: ============================================================================

echo ============================================================
echo  CUDA + cuDNN Installer for RTX 4080 / TensorFlow
echo ============================================================
echo.
echo This script will:
echo   1. Check your current NVIDIA driver version
echo   2. Open the correct CUDA 12.3 download page
echo   3. Open the correct cuDNN 8.9 download page
echo   4. Add CUDA to your system PATH automatically
echo   5. Verify the install when done
echo.
echo YOU WILL NEED:
echo   - A free NVIDIA developer account (for cuDNN download)
echo   - ~4 GB of disk space
echo   - Internet connection
echo.
pause

:: ----------------------------------------------------------------------------
:: Step 1 — Check current NVIDIA driver (must be 525.60+ for CUDA 12.x)
:: ----------------------------------------------------------------------------
echo.
echo [Step 1] Checking NVIDIA driver version...
nvidia-smi
echo.
echo If you see your RTX 4080 listed above, your driver is installed.
echo The "Driver Version" shown must be 525.60 or higher.
echo If it is lower, visit https://www.nvidia.com/drivers and update first.
echo.
pause

:: ----------------------------------------------------------------------------
:: Step 2 — Download CUDA 12.3
:: TensorFlow 2.16 requires CUDA 12.x. We use 12.3 — stable and well-tested.
:: ----------------------------------------------------------------------------
echo [Step 2] Opening CUDA 12.3 download page...
echo.
echo On the page that opens, select:
echo   Operating System : Windows
echo   Architecture     : x86_64
echo   Version          : 11  (or your Windows version)
echo   Installer Type   : exe (local)
echo.
echo Download and run the installer. Choose "Custom" install and select:
echo   CUDA Toolkit (required)
echo   CUDA Documentation (optional)
echo   CUDA Samples (optional)
echo   DO NOT install the bundled display driver if yours is already up to date.
echo.
start https://developer.nvidia.com/cuda-12-3-0-download-archive
pause

:: ----------------------------------------------------------------------------
:: Step 3 — Download cuDNN 8.9 for CUDA 12.x
:: cuDNN gives TensorFlow access to GPU-accelerated deep learning primitives.
:: You need a free NVIDIA developer account to download it.
:: ----------------------------------------------------------------------------
echo [Step 3] Opening cuDNN download page...
echo.
echo On the page that opens:
echo   1. Log in or create a free NVIDIA developer account
echo   2. Select: cuDNN v8.9.x for CUDA 12.x
echo   3. Download: "Local Installer for Windows (Zip)"
echo.
echo After downloading, extract the zip. You will find three folders:
echo   bin\       cuda\bin\
echo   include\   cuda\include\
echo   lib\       cuda\lib\x64\
echo.
echo Copy the CONTENTS of each folder into the matching CUDA install folder:
echo   bin\     contents → C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3\bin\
echo   include\ contents → C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3\include\
echo   lib\     contents → C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3\lib\x64\
echo.
start https://developer.nvidia.com/rdp/cudnn-download
pause

:: ----------------------------------------------------------------------------
:: Step 4 — Add CUDA to system PATH
:: This lets Python / TensorFlow find the CUDA libraries at runtime.
:: ----------------------------------------------------------------------------
echo [Step 4] Adding CUDA 12.3 to system PATH...
echo.

setx PATH "%PATH%;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3\libnvvp" /M

echo PATH updated. You must CLOSE and REOPEN any terminal windows for this to take effect.
echo.
pause

:: ----------------------------------------------------------------------------
:: Step 5 — Verify CUDA installation
:: ----------------------------------------------------------------------------
echo [Step 5] Verifying CUDA install...
echo.
nvcc --version
echo.
echo If you see "release 12.3" above, CUDA is installed correctly.
echo If you see "'nvcc' is not recognized", close this window and reopen as Admin.
echo.

:: Check nvidia-smi again — should now show CUDA Version: 12.3
nvidia-smi
echo.
echo The "CUDA Version" in the top-right corner of the table above should show 12.3
echo.
pause

:: ----------------------------------------------------------------------------
:: Done
:: ----------------------------------------------------------------------------
echo ============================================================
echo  CUDA setup complete.
echo  Next step: Run  02_install_python_packages.bat
echo ============================================================
pause
