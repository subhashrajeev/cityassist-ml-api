@echo off
echo ============================================================
echo CityAssist DS - Push to GitHub Helper
echo ============================================================
echo.
echo This script will help you push your code to GitHub.
echo.
echo BEFORE running this script:
echo 1. Create a new repository on GitHub.com
echo 2. Get the repository URL (example: https://github.com/username/repo.git)
echo.
echo ============================================================
echo.
set /p REPO_URL="Enter your GitHub repository URL: "

if "%REPO_URL%"=="" (
    echo ERROR: Repository URL cannot be empty!
    pause
    exit /b 1
)

echo.
echo Adding remote repository...
git remote add origin %REPO_URL%

echo.
echo Pushing to GitHub...
git branch -M main
git push -u origin main

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ============================================================
    echo SUCCESS! Your code has been pushed to GitHub!
    echo ============================================================
    echo.
    echo Repository URL: %REPO_URL%
    echo.
    echo Share this URL with your DevOps team member.
    echo They can clone and deploy with:
    echo   git clone %REPO_URL%
    echo   cd cityassist-ds
    echo   docker-compose up --build -d
    echo.
) else (
    echo.
    echo ============================================================
    echo ERROR: Failed to push to GitHub!
    echo ============================================================
    echo.
    echo Possible reasons:
    echo - Repository URL is incorrect
    echo - Authentication failed (use Personal Access Token as password)
    echo - Remote 'origin' already exists
    echo.
    echo To fix 'remote already exists' error:
    echo   git remote remove origin
    echo   Then run this script again
    echo.
)

pause
