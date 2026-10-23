@echo off
setlocal
set "ROOT=%~dp0"

rem ---------------------------------------------------------------------------
rem  IndexTTS-2.5 low-VRAM launcher (4-8 GB cards)
rem    INDEXTTS_DTYPE    auto | off | fp32 | fp16 | bf16
rem    INDEXTTS_LOWVRAM  auto | off
rem  "auto" resolves both from the GPU: pre-Turing cards fall back to fp32,
rem  cards below 8 GB switch to stage swapping instead of loading every weight
rem  onto the card at once.  Run tools\vram_bench.py to see your own numbers.
rem ---------------------------------------------------------------------------
set "INDEXTTS_DTYPE=auto"
set "INDEXTTS_LOWVRAM=auto"

echo ==========================================================
echo   IndexTTS-2.5 WebUI   [low-VRAM launcher]
echo ==========================================================
echo.

if not exist "%ROOT%python_embeded\python.exe" (
    echo [ERROR] python_embeded\python.exe not found.
    echo         Put this file in the root of an IndexTTS-2.5 package.
    pause
    exit /b 1
)
if not exist "%ROOT%indextts\vram.py" (
    echo [ERROR] indextts\vram.py not found - run apply.py first.
    pause
    exit /b 1
)

cd /d "%ROOT%"

echo [1/2] Checking runtime environment...
"%ROOT%python_embeded\python.exe" -u -c "import torch; print('    PyTorch:', torch.__version__, '| CUDA:', torch.cuda.is_available()); print('    GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'not detected'); print('    native BF16:', torch.cuda.is_bf16_supported(including_emulation=False) if torch.cuda.is_available() else False)"
if %errorlevel% neq 0 (
    echo [ERROR] runtime environment problem
    pause
    exit /b 1
)

echo.
echo [2/2] Loading models and starting the server...
echo     - compute type picked from the GPU
echo     - stage swapping keeps peak VRAM low (disable: --lowvram off)
echo     - long text is split on punctuation on small cards
echo     - first start takes a while; the console stays quiet until it is ready
echo.

echo ==========================================================
echo.
echo      Open this URL in your browser:
echo.
echo          http://127.0.0.1:7860
echo.
echo      Wait for "Running on local URL" = server is ready.
echo      Keep this window open; closing it stops the server.
echo.
echo ==========================================================
echo.

"%ROOT%python_embeded\python.exe" -u "%ROOT%webui.py" --version 2.5 --model_dir "%ROOT%checkpoints" --fp16 --cuda_kernel --gui_seg_tokens 60

echo.
echo [INFO] WebUI exited.
pause
