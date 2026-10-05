@echo off
rem One scheduled tick for Windows Task Scheduler. It runs the same due check
rem the console loop uses; when a weekday slot has passed the pipeline
rem refreshes every ticker and exits, so Task Scheduler restarts it cleanly
rem next time instead of leaving a long-lived process behind. Pass --force to
rem run immediately, which is what manual repair uses.
setlocal
for %%I in ("%~dp0..") do set "PROG5_DIR=%%~fI"
cd /d "%PROG5_DIR%"
if not exist "data" mkdir "data"
set "LOG=%PROG5_DIR%\data\scheduler.log"
set "ARGS=%*"
if "%ARGS%"=="" set "ARGS=--once"
where python >nul 2>&1 && (set "PYTHON=python") || (set "PYTHON=py -3")
echo [%date% %time%] %PYTHON% -m prog5.cli schedule %ARGS% >> "%LOG%"
call %PYTHON% -m prog5.cli schedule %ARGS% >> "%LOG%" 2>&1
echo [%date% %time%] exit=%errorlevel% >> "%LOG%"
endlocal
