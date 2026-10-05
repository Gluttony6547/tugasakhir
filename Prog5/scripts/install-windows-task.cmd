@echo off
rem Register the weekday refresh with Windows Task Scheduler. No Docker and no
rem administrator rights: the task runs as the signed-in user and only while
rem that user is logged on. Re-running replaces the task in place.
setlocal
set "TASK_NAME=Prog5 daily refresh"
for %%I in ("%~dp0..") do set "PROG5_DIR=%%~fI"
set "RUNNER=%PROG5_DIR%\scripts\run-scheduled-refresh.cmd"

if not exist "%RUNNER%" (
  echo Missing %RUNNER%
  exit /b 1
)

echo Registering "%TASK_NAME%" ...
schtasks /create /f /tn "%TASK_NAME%" /sc weekly /d MON,TUE,WED,THU,FRI /st 17:35 /tr "\"%RUNNER%\""
if errorlevel 1 (
  echo.
  echo Could not register the task. Run this script from a normal user terminal.
  exit /b 1
)

echo.
echo Registered "%TASK_NAME%": weekdays at 17:35 local time, after the 17:30
echo pipeline slot so the due check cannot race the trigger.
echo   Log:     %PROG5_DIR%\data\scheduler.log
echo   Run now: schtasks /run /tn "%TASK_NAME%"
echo   Remove:  schtasks /delete /f /tn "%TASK_NAME%"
echo.
echo The task refreshes whichever database the app is configured for. To write
echo to the deployed Neon database from this host, set the connection string
echo once and sign out and back in so Task Scheduler inherits it:
echo   setx PROG5_DATABASE_URL "postgresql://..."
echo.
echo Without that variable the task keeps the local SQLite database current.
endlocal
