@echo off
chcp 65001 > nul
cd /d "%~dp0"

echo.
echo =====================================
echo         SL_tool 総合実験メニュー
echo =====================================
echo.

set /p SUBJECT_ID=被験者IDを入力してください: 
set /p SUBJECT_NAME=被験者名を入力してください (省略可): 

:MENU
cls
echo.
echo =====================================
echo   被験者ID: %SUBJECT_ID%  名前: %SUBJECT_NAME%
echo =====================================
echo   1. Naming (命名課題)
echo   2. ActionDescription (行動描写課題)
echo   3. Description (描写課題)
echo   4. Repetition (復唱課題)
echo   5. ReadAloud (音読課題)
echo   6. WordFluency (語想起課題)
echo   0. 終了する
echo =====================================
echo.

set /p CHOICE=実行する課題の番号を選んでください (1〜6, 0): 

if "%CHOICE%"=="1" goto TASK_NAMING
if "%CHOICE%"=="2" goto TASK_ACTION
if "%CHOICE%"=="3" goto TASK_DESC
if "%CHOICE%"=="4" goto TASK_REP
if "%CHOICE%"=="5" goto TASK_READ
if "%CHOICE%"=="6" goto TASK_FLUENCY
if "%CHOICE%"=="0" exit

goto MENU

:TASK_NAMING
echo.
echo [起動] Naming タスクを実行します...
cd /d "%~dp0Naming"
start "" /wait "Naming.exe" "%SUBJECT_ID%" "%SUBJECT_NAME%"
cd /d "%~dp0"
pause
goto MENU

:TASK_ACTION
echo.
echo [起動] ActionDescription タスクを実行します...
cd /d "%~dp0ActionDescription"
start "" /wait "ActionDescription.exe" "%SUBJECT_ID%" "%SUBJECT_NAME%"
cd /d "%~dp0"
pause
goto MENU

:TASK_DESC
echo.
echo [起動] Description タスクを実行します...
cd /d "%~dp0Description"
start "" /wait "Description.exe" "%SUBJECT_ID%" "%SUBJECT_NAME%"
cd /d "%~dp0"
pause
goto MENU

:TASK_REP
echo.
echo [起動] Repetition タスクを実行します...
cd /d "%~dp0Repetition"
start "" /wait "Repetition.exe" "%SUBJECT_ID%" "%SUBJECT_NAME%"
cd /d "%~dp0"
pause
goto MENU

:TASK_READ
echo.
echo [起動] ReadAloud タスクを実行します...
cd /d "%~dp0ReadAloud"
start "" /wait "ReadAloud.exe" "%SUBJECT_ID%" "%SUBJECT_NAME%"
cd /d "%~dp0"
pause
goto MENU

:TASK_FLUENCY
echo.
echo [起動] WordFluency タスクを実行します...
cd /d "%~dp0WordFluency"
start "" /wait "WordFluency.exe" "%SUBJECT_ID%" "%SUBJECT_NAME%"
cd /d "%~dp0"
pause
goto MENU