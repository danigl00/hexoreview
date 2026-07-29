@echo off
rem Stops the review dashboard. Safe to run even if it is not running.
title Stop seizure review
echo Closing the seizure review dashboard...
set FOUND=0
for /f "tokens=5" %%p in ('netstat -ano ^| findstr ":5006" ^| findstr "LISTENING"') do (
    taskkill /F /PID %%p >nul 2>&1
    set FOUND=1
)
if "%FOUND%"=="1" (echo Stopped.) else (echo It was not running.)
timeout /t 3 >nul