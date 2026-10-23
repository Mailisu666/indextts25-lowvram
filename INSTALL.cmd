@echo off
setlocal
cd /d "%~dp0"
title IndexTTS-2.5 low-VRAM patch - installer

echo ==========================================================
echo   IndexTTS-2.5 low-VRAM patch  -  installer
echo ==========================================================
echo.
echo You need an ALREADY WORKING IndexTTS-2.5 install first.
echo That is the folder which contains:
echo     webui.py
echo     indextts\
echo     checkpoints\        (with gpt.pth, s2mel.pth, codec.pth ...)
echo.
echo Drag that folder into this window and press Enter,
echo or type its full path.
echo.

:ask
set "TARGET="
set /p TARGET=IndexTTS folder: 
if not defined TARGET goto ask_empty
for %%A in ("%TARGET%") do set "TARGET=%%~fA"
if not exist "%TARGET%\webui.py" goto bad_folder
if not exist "%TARGET%\indextts\infer_v2_5.py" goto bad_version
goto pick_python

:ask_empty
echo [ERROR] nothing entered, try again.
echo.
goto ask

:bad_folder
echo.
echo [ERROR] "%TARGET%\webui.py" not found.
echo         Pick the folder that DIRECTLY contains webui.py.
echo.
goto ask

:bad_version
echo.
echo [ERROR] "%TARGET%\indextts\infer_v2_5.py" not found.
echo         This patch only supports IndexTTS-2.5, not 1.x or 2.0.
echo.
pause
exit /b 1

rem ---- locate a python interpreter (the installer itself needs no torch) --
:pick_python
set "PYEXE="
set "PYARGS="
set "PYLIST=%TEMP%\indextts_python_path.txt"
if exist "%TARGET%\python_embeded\python.exe" set "PYEXE=%TARGET%\python_embeded\python.exe"
if defined PYEXE goto have_python
where python > "%PYLIST%" 2>nul
if errorlevel 1 goto try_py
set /p PYEXE=<"%PYLIST%"
goto have_python

:try_py
where py > "%PYLIST%" 2>nul
if errorlevel 1 goto no_python
set /p PYEXE=<"%PYLIST%"

:have_python
if not defined PYEXE goto no_python
if /i "%PYEXE:~-6%"=="py.exe" set "PYARGS=-3"
echo.
echo target : %TARGET%
echo python : "%PYEXE%" %PYARGS%
echo.
echo Patching...
echo.
"%PYEXE%" %PYARGS% "%~dp0apply.py" --root "%TARGET%"
if errorlevel 1 goto failed
goto done

:no_python
echo.
echo [ERROR] no python found.
echo         Use a bundle that ships python_embeded, or install Python 3.8+.
echo.
pause
exit /b 1

:failed
echo.
echo ==========================================================
echo   FAILED - see the message above.
echo   If it says "anchor not found", your package version is
echo   different from the one this patch was written for, and
echo   NOTHING was modified.
echo ==========================================================
echo.
pause
exit /b 1

:done
echo.
echo ==========================================================
echo   Done.
echo.
echo   Next step: double-click
echo       %TARGET%\webui_6g.bat
echo.
echo   Then open http://127.0.0.1:7860 in your browser.
echo   The first start takes 1-3 minutes; the console stays
echo   quiet until it prints "Running on local URL".
echo.
echo   To undo: copy the *.bak_pre_lowvram files back
echo   (see README.md, section "rollback").
echo ==========================================================
echo.
pause
exit /b 0
