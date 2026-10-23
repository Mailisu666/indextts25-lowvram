@echo off
setlocal
cd /d "%~dp0"

echo ==========================================================
echo   Publish this repository to GitHub
echo ==========================================================
echo.

where git >nul 2>nul
if errorlevel 1 (
    echo [ERROR] git is not installed.
    echo.
    echo     winget install --id Git.Git -e --source winget
    echo.
    echo   After it finishes, close this window and run publish.cmd again.
    echo.
    pause
    exit /b 1
)

rem ---- preflight: no model weights, license present, syntax clean ----------
set "PY="
if exist "D:\2Z\AI\python_embeded\python.exe" set "PY=D:\2Z\AI\python_embeded\python.exe"
if not defined PY (
    where python >nul 2>nul
    if not errorlevel 1 set "PY=python"
)
if defined PY (
    "%PY%" tools\preflight.py
    if errorlevel 1 (
        echo.
        echo [ABORT] preflight failed - fix the issues above before publishing.
        echo.
        pause
        exit /b 1
    )
) else (
    echo [WARN] python not found, preflight skipped.
)

echo.
echo Create an EMPTY repository at https://github.com/new  (no README, no .gitignore)
echo then paste its URL below.
echo.
set "REMOTE="
set /p REMOTE=GitHub repo URL: 
if "%REMOTE%"=="" (
    echo [ERROR] no URL given.
    pause
    exit /b 1
)

if not exist ".git" (
    git init
    git branch -M main
)

rem ---- author identity (loop until both are non-empty) ---------------------
set "GITNAME="
for /f "delims=" %%i in ('git config user.name 2^>nul') do set "GITNAME=%%i"
if not defined GITNAME goto ask_name
goto have_name

:ask_name
set "GITNAME="
set /p GITNAME=Your name for git commits (anything, e.g. your GitHub username): 
if not defined GITNAME (
    echo [ERROR] name cannot be empty.
    goto ask_name
)
git config user.name "%GITNAME%"

:have_name
set "GITEMAIL="
for /f "delims=" %%i in ('git config user.email 2^>nul') do set "GITEMAIL=%%i"
if not defined GITEMAIL goto ask_email
goto have_email

:ask_email
set "GITEMAIL="
set /p GITEMAIL=Your email for git commits (GitHub noreply address is fine): 
if not defined GITEMAIL (
    echo [ERROR] email cannot be empty.
    goto ask_email
)
git config user.email "%GITEMAIL%"

:have_email
git add .
git commit -m "IndexTTS-2.5 low-VRAM patch: capability-based dtype + stage swapping"
if errorlevel 1 echo [INFO] nothing new to commit.

rem ---- never push an unborn branch (that is what "src refspec does not match" means)
git rev-parse --verify HEAD >nul 2>nul
if errorlevel 1 (
    echo.
    echo [ERROR] no commit exists yet - nothing to push.
    echo         Check the messages above ^(usually a missing git author identity^).
    echo.
    pause
    exit /b 1
)

git remote get-url origin >nul 2>nul
if errorlevel 1 (
    git remote add origin "%REMOTE%"
) else (
    git remote set-url origin "%REMOTE%"
)

echo.
echo Pushing... (a browser window may open for GitHub login)
git push -u origin main
if errorlevel 1 (
    echo.
    echo [ERROR] push failed. Common causes:
    echo   - the repository was created WITH a README/LICENSE ^(must be empty^)
    echo   - you are not logged in, or the URL/branch is wrong
    echo   Fix and run this file again.
    echo.
    pause
    exit /b 1
)

echo.
echo Done. Repository published.
echo.
pause
