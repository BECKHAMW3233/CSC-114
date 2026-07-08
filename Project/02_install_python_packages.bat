@echo off
:: ============================================================================
:: 02_install_python_packages.bat
:: Python environment + all pip packages for the OCR training pipeline
:: Run AFTER 01_install_cuda.bat has completed successfully
:: Does NOT need to be run as Administrator
:: ============================================================================

echo ============================================================
echo  Python Package Installer — EMNIST OCR Pipeline
echo  Target: E:\CSC-114\project
echo ============================================================
echo.

:: ----------------------------------------------------------------------------
:: Step 0 — Confirm Python is available and check version
:: ----------------------------------------------------------------------------
echo [Step 0] Checking Python version...
python --version
echo.
echo Python 3.12 is required for this pipeline.
echo PyTorch 2.5.1+cu121 supports Python 3.12.
echo.
pause

:: ----------------------------------------------------------------------------
:: Step 1 — Create the project directory
:: ----------------------------------------------------------------------------
echo [Step 1] Creating project directory...
if not exist "E:\CSC-114\project" (
    mkdir "E:\CSC-114\project"
    echo Created: E:\CSC-114\project
) else (
    echo Already exists: E:\CSC-114\project
)

if not exist "E:\CSC-114\project\datasets" (
    mkdir "E:\CSC-114\project\datasets"
    echo Created: E:\CSC-114\project\datasets
)

if not exist "E:\CSC-114\project\datasets\kaggle" (
    mkdir "E:\CSC-114\project\datasets\kaggle"
    echo Created: E:\CSC-114\project\datasets\kaggle
)

echo.

:: ----------------------------------------------------------------------------
:: Step 2 — Create a virtual environment inside the project folder
:: ----------------------------------------------------------------------------
echo [Step 2] Creating virtual environment at E:\CSC-114\project\venv ...
python -m venv "E:\CSC-114\project\venv"
echo Virtual environment created.
echo.

:: ----------------------------------------------------------------------------
:: Step 3 — Activate the virtual environment
:: ----------------------------------------------------------------------------
echo [Step 3] Activating virtual environment...
call "E:\CSC-114\project\venv\Scripts\activate.bat"
echo Virtual environment active.
echo.

:: ----------------------------------------------------------------------------
:: Step 4 — Upgrade pip
:: ----------------------------------------------------------------------------
echo [Step 4] Upgrading pip...
python -m pip install --upgrade pip
echo.

:: ----------------------------------------------------------------------------
:: Step 5 — Install core packages
:: ----------------------------------------------------------------------------
echo [Step 5] Installing Python packages (this will take 5-15 minutes)...
echo.

echo Installing numpy...
pip install "numpy>=1.24,<2.0"
echo.

echo Installing Pillow...
pip install pillow
echo.

echo Installing Matplotlib...
pip install matplotlib
echo.

echo Installing PyTorch with CUDA 12.1 support (RTX 4080)...
echo (Large download — ~2.5 GB, may take 10-20 minutes)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
echo.

echo Installing torchmetrics...
pip install torchmetrics
echo.

echo Installing onnx (model export)...
pip install onnx
echo.

echo Installing onnxruntime-gpu (CUDA 12.1 compatible — required for pipeline inference)...
echo Version pinned to 1.19.2 — matches CUDA 12.1. Do NOT upgrade without checking CUDA compatibility.
pip install onnxruntime-gpu==1.19.2
echo.

echo Installing optuna (hyperparameter search)...
pip install optuna
echo.

echo Installing Keras 3 (course assignments)...
pip install "keras>=3.0"
echo.

echo Installing KerasHub (course assignments)...
pip install keras-hub
echo.

echo Installing TensorFlow Datasets (EMNIST loader)...
pip install tensorflow-datasets
echo.

:: ----------------------------------------------------------------------------
:: Step 6 — Install supplementary dataset dependencies
:: ----------------------------------------------------------------------------
echo [Step 6] Installing supplementary dataset packages...
echo.

echo Installing Kaggle API (supplementary dataset download)...
pip install kaggle
echo.
echo [IMPORTANT] To use the Kaggle dataset downloader:
echo   1. Go to https://www.kaggle.com/settings
echo   2. Click API section -^> "Create New Token"
echo   3. This downloads kaggle.json
echo   4. Place kaggle.json at: C:\Users\Will\.kaggle\kaggle.json
echo   5. Then run: python download_datasets.py
echo.

echo Installing pandas (required for Kaggle CSV preprocessing)...
pip install pandas
echo.

echo Installing scipy (required for SVHN .mat file loading)...
pip install scipy
echo.

echo Installing certifi (SSL certificate fix for USPS download)...
pip install certifi
echo.

:: ----------------------------------------------------------------------------
:: Step 7 — Install advanced optimizers for ensemble training
:: ----------------------------------------------------------------------------
echo [Step 7] Installing advanced optimizer packages...
echo.

echo Installing lion-pytorch (Lion optimizer — Model 1)...
echo Discovered via symbolic search, superior generalization on vision CNNs
pip install lion-pytorch
echo.

echo Installing schedulefree (Schedule-Free AdamW — Model 2)...
echo MLCommons 2024 AlgoPerf challenge winner, no LR scheduler required
pip install schedulefree
echo.

echo Installing pytorch_optimizer (SOAP, AdaHessian, 100+ modern optimizers)...
echo Required for experimental second-order optimizer training files
pip install pytorch_optimizer
echo.

echo Installing psutil (hardware monitoring — CPU percent and system RAM per epoch)...
pip install psutil
echo.

:: ----------------------------------------------------------------------------
:: Step 8 — Copy scripts
:: ----------------------------------------------------------------------------
echo [Step 8] Copying training scripts...
if exist "%~dp0ocr_pytorch_model.py" (
    copy "%~dp0ocr_pytorch_model.py" "E:\CSC-114\project\ocr_pytorch_model.py"
    echo Copied ocr_pytorch_model.py
)
if exist "%~dp0ocr_pytorch_model2.py" (
    copy "%~dp0ocr_pytorch_model2.py" "E:\CSC-114\project\ocr_pytorch_model2.py"
    echo Copied ocr_pytorch_model2.py
)
if exist "%~dp0ocr_pytorch_model3.py" (
    copy "%~dp0ocr_pytorch_model3.py" "E:\CSC-114\project\ocr_pytorch_model3.py"
    echo Copied ocr_pytorch_model3.py
)
if exist "%~dp0supplementary_data.py" (
    copy "%~dp0supplementary_data.py" "E:\CSC-114\project\supplementary_data.py"
    echo Copied supplementary_data.py
)
if exist "%~dp0download_datasets.py" (
    copy "%~dp0download_datasets.py" "E:\CSC-114\project\download_datasets.py"
    echo Copied download_datasets.py
)
if exist "%~dp0ocr_pipeline.py" (
    copy "%~dp0ocr_pipeline.py" "E:\CSC-114\project\ocr_pipeline.py"
    echo Copied ocr_pipeline.py
)
echo.

:: ----------------------------------------------------------------------------
:: Step 9 — GPU verification
:: ----------------------------------------------------------------------------
echo [Step 9] Running GPU verification...
if exist "%~dp003_verify_gpu.py" (
    python "%~dp003_verify_gpu.py"
) else (
    echo Skipping GPU check — 03_verify_gpu.py not found.
)
echo.

:: ----------------------------------------------------------------------------
:: Step 10 — Package list
:: ----------------------------------------------------------------------------
echo [Step 10] Installed packages:
pip list
echo.

:: ----------------------------------------------------------------------------
:: Done
:: ----------------------------------------------------------------------------
echo ============================================================
echo  Installation complete.
echo.
echo  NEXT STEPS:
echo    1. Place kaggle.json at C:\Users\Will\.kaggle\kaggle.json
echo    2. Run: python download_datasets.py             (downloads all datasets)
echo    3. Run: python ocr_pytorch_model.py             (train Model 1 — Lion, auto batch)
echo    4. Run: python ocr_pytorch_model2.py            (train Model 2 — SF-AdamW, auto batch)
echo    5. Run: python ocr_pytorch_model3.py            (train Model 3 — SGD, auto batch)
echo.
echo  DISTILLATION STEPS (run after all 3 base models complete):
echo    6. Run: python ocr_distillation.py --phase 1             (generate soft labels)
echo    7. Run: python ocr_distillation.py --phase 2 --model 1   (distill Model 1)
echo    8. Run: python ocr_distillation.py --phase 2 --model 2   (distill Model 2)
echo    9. Run: python ocr_distillation.py --phase 2 --model 3   (distill Model 3)
echo   10. Run: python ocr_distillation.py --phase 3             (ONNX validation)
echo.
echo  EXPERIMENTAL OPTIMIZER RUNS (standalone, independent of main ensemble):
echo   11. Run: python ocr_adahessian_64.py             (AdaHessian OCRConvNetTriple 64x64)
echo   12. Run: python ocr_adahessian_128.py            (AdaHessian OCRConvNetTriple 128x128)
echo   13. Run: python ocr_soap_64.py                   (SOAP OCRConvNetTriple 64x64)
echo   14. Run: python ocr_soap_128.py                  (SOAP OCRConvNetTriple 128x128)
echo.
echo  All base output:       E:\CSC-114\project\pytorch\
echo  All distilled output:  E:\CSC-114\project\pytorch_distill1\  (2\  3\)
echo  AdaHessian 64x64:      E:\CSC-114\project\pytorch_adahessian_64\
echo  AdaHessian 128x128:    E:\CSC-114\project\pytorch_adahessian_128\
echo  SOAP 64x64:            E:\CSC-114\project\pytorch_soap_64\
echo  SOAP 128x128:          E:\CSC-114\project\pytorch_soap_128\
echo ============================================================
pause