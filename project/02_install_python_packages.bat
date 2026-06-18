@echo off
:: ============================================================================
:: 02_install_python_packages.bat
:: Python environment + all pip packages for the OCR training pipeline
:: Run AFTER 01_install_cuda.bat has completed successfully
:: Does NOT need to be run as Administrator
:: ============================================================================

echo ============================================================
echo  Python Package Installer — EMNIST OCR Pipeline
echo  Target: E:\CSC-114\emnist-model
echo ============================================================
echo.

:: ----------------------------------------------------------------------------
:: Step 0 — Confirm Python is available and check version
:: Requires Python 3.10 or 3.11. TensorFlow 2.16 does NOT support Python 3.12+
:: Download Python 3.11 from: https://www.python.org/downloads/release/python-3119/
:: ----------------------------------------------------------------------------
echo [Step 0] Checking Python version...
python --version
echo.
echo Python 3.10 or 3.11 is required.
echo TensorFlow 2.16 does NOT support Python 3.12 or higher.
echo If your version is 3.12+, download 3.11 from https://www.python.org/downloads/
echo.
pause

:: ----------------------------------------------------------------------------
:: Step 1 — Create the project directory
:: ----------------------------------------------------------------------------
echo [Step 1] Creating project directory...
if not exist "E:\CSC-114\emnist-model" (
    mkdir "E:\CSC-114\emnist-model"
    echo Created: E:\CSC-114\emnist-model
) else (
    echo Already exists: E:\CSC-114\emnist-model
)

if not exist "E:\CSC-114\emnist-model\datasets" (
    mkdir "E:\CSC-114\emnist-model\datasets"
    echo Created: E:\CSC-114\emnist-model\datasets
)

echo.

:: ----------------------------------------------------------------------------
:: Step 2 — Create a virtual environment inside the project folder
:: This keeps all packages isolated from your system Python install.
:: ----------------------------------------------------------------------------
echo [Step 2] Creating virtual environment at E:\CSC-114\emnist-model\venv ...
python -m venv "E:\CSC-114\emnist-model\venv"
echo Virtual environment created.
echo.

:: ----------------------------------------------------------------------------
:: Step 3 — Activate the virtual environment
:: ----------------------------------------------------------------------------
echo [Step 3] Activating virtual environment...
call "E:\CSC-114\emnist-model\venv\Scripts\activate.bat"
echo Virtual environment active.
echo.

:: ----------------------------------------------------------------------------
:: Step 4 — Upgrade pip itself first
:: Old pip versions sometimes fail to resolve TensorFlow's CUDA dependencies.
:: ----------------------------------------------------------------------------
echo [Step 4] Upgrading pip...
python -m pip install --upgrade pip
echo.

:: ----------------------------------------------------------------------------
:: Step 5 — Install core packages in dependency order
::
:: Order matters:
::   1. numpy          — everything depends on this
::   2. tensorflow     — installs TF + bundled CUDA libs (tensorflow[and-cuda])
::   3. keras          — Keras 3 multi-backend frontend
::   4. keras-hub      — pretrained models (Xception backbone)
::   5. tensorflow-datasets — EMNIST download and caching
::   6. pillow         — image loading in predict_image()
::   7. matplotlib     — training curve plots
:: ----------------------------------------------------------------------------
echo [Step 5] Installing Python packages (this will take 5-15 minutes)...
echo.

echo Installing numpy...
pip install "numpy>=1.24,<2.0"
echo.

echo Installing TensorFlow with bundled CUDA support...
echo (tensorflow[and-cuda] bundles CUDA runtime so TF finds the GPU automatically)
pip install "tensorflow[and-cuda]>=2.16"
echo.

echo Installing Keras 3...
pip install "keras>=3.0"
echo.

echo Installing KerasHub (pretrained Xception backbone)...
pip install keras-hub
echo.

echo Installing TensorFlow Datasets (EMNIST loader)...
pip install tensorflow-datasets
echo.

echo Installing Pillow (image I/O)...
pip install pillow
echo.

echo Installing Matplotlib (training plots)...
pip install matplotlib

echo.
echo Installing PyTorch with CUDA 12.1 support (RTX 4080)...
echo (This is a large download — ~2.5 GB, may take 10-20 minutes)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo.
echo Installing torchmetrics (accuracy tracking for PyTorch training loop)...
pip install torchmetrics
echo.

:: ----------------------------------------------------------------------------
:: Step 6 — Copy the training script into the project folder
:: ----------------------------------------------------------------------------
echo [Step 6] Copying training script...
if exist "%~dp0ocr_handwriting_model.py" (
    copy "%~dp0ocr_handwriting_model.py" "E:\CSC-114\emnist-model\ocr_handwriting_model.py"
    echo Copied ocr_handwriting_model.py to E:\CSC-114\emnist-model\
) else (
    echo WARNING: ocr_handwriting_model.py not found next to this script.
    echo Manually copy it to E:\CSC-114\emnist-model\ocr_handwriting_model.py
)
echo.

:: ----------------------------------------------------------------------------
:: Step 7 — Copy and run the GPU verification script
:: ----------------------------------------------------------------------------
echo [Step 7] Running GPU verification...
if exist "%~dp003_verify_gpu.py" (
    python "%~dp003_verify_gpu.py"
) else (
    echo Skipping GPU check — 03_verify_gpu.py not found next to this script.
    echo Run it manually after setup.
)
echo.

:: ----------------------------------------------------------------------------
:: Step 8 — Print final package list for your records
:: ----------------------------------------------------------------------------
echo [Step 8] Installed packages:
pip list
echo.

:: ----------------------------------------------------------------------------
:: Done
:: ----------------------------------------------------------------------------
echo ============================================================
echo  Installation complete.
echo.
echo  To train the model:
echo    1. Open a terminal
echo    2. Run: E:\CSC-114\emnist-model\venv\Scripts\activate
echo    3. Run: cd E:\CSC-114\emnist-model
echo    4. Run: python ocr_handwriting_model.py
echo.
echo  All output files will appear in E:\CSC-114\emnist-model\
echo ============================================================
pause
