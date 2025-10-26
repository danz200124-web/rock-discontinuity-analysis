@echo off
REM GitHub上传脚本
REM 请先修改下面的用户信息

echo ================================================================================
echo GitHub项目上传脚本
echo ================================================================================
echo.

REM ===== 步骤1: 配置Git用户信息 (首次使用需要设置) =====
echo [步骤1] 配置Git用户信息...
set /p GIT_USERNAME="请输入您的GitHub用户名: "
set /p GIT_EMAIL="请输入您的GitHub邮箱: "

git config user.name "%GIT_USERNAME%"
git config user.email "%GIT_EMAIL%"
echo Git配置完成！
echo.

REM ===== 步骤2: 查看将要提交的文件 =====
echo [步骤2] 查看将要提交的文件...
git status
echo.

pause
echo.

REM ===== 步骤3: 添加所有文件到Git =====
echo [步骤3] 添加文件到Git...
git add .
echo 文件添加完成！
echo.

REM ===== 步骤4: 创建初始提交 =====
echo [步骤4] 创建初始提交...
git commit -m "Initial commit: Rock discontinuity analysis system with HDBSCAN

Features:
- HDBSCAN clustering for adaptive discontinuity detection
- MAGSAC++ plane fitting
- Mean-Shift density analysis
- Optimized configurations for high coverage (70-90%%)
- Multiple preset configurations
- Comprehensive documentation

Generated with Claude Code (https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"

echo 提交完成！
echo.

REM ===== 步骤5: 检查GitHub CLI是否安装 =====
echo [步骤5] 检查GitHub CLI...
where gh >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ❌ GitHub CLI未安装！
    echo.
    echo 请选择以下方式之一：
    echo.
    echo 方式1: 安装GitHub CLI ^(推荐^)
    echo   1. 访问: https://cli.github.com/
    echo   2. 下载并安装GitHub CLI
    echo   3. 运行: gh auth login
    echo   4. 重新运行本脚本
    echo.
    echo 方式2: 手动在GitHub网页创建仓库
    echo   1. 访问: https://github.com/new
    echo   2. 创建新仓库 ^(例如: rock-discontinuity-analysis^)
    echo   3. 运行以下命令:
    echo      git remote add origin https://github.com/您的用户名/仓库名.git
    echo      git branch -M main
    echo      git push -u origin main
    echo.
    pause
    exit /b 1
)

echo GitHub CLI已安装！
echo.

REM ===== 步骤6: 检查GitHub登录状态 =====
echo [步骤6] 检查GitHub登录状态...
gh auth status
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ❌ 未登录GitHub！
    echo 请运行: gh auth login
    echo.
    pause
    exit /b 1
)

echo GitHub登录成功！
echo.

REM ===== 步骤7: 创建GitHub仓库 =====
echo [步骤7] 创建GitHub仓库...
echo.
set /p REPO_NAME="请输入仓库名称 [rock-discontinuity-analysis]: "
if "%REPO_NAME%"=="" set REPO_NAME=rock-discontinuity-analysis

set /p REPO_DESC="请输入仓库描述 [Rock discontinuity analysis system with advanced algorithms]: "
if "%REPO_DESC%"=="" set REPO_DESC=Rock discontinuity analysis system with advanced algorithms

set /p REPO_VISIBILITY="仓库可见性 (public/private) [public]: "
if "%REPO_VISIBILITY%"=="" set REPO_VISIBILITY=public

echo.
echo 即将创建仓库:
echo   名称: %REPO_NAME%
echo   描述: %REPO_DESC%
echo   可见性: %REPO_VISIBILITY%
echo.
pause

gh repo create %REPO_NAME% --description "%REPO_DESC%" --%REPO_VISIBILITY% --source=. --remote=origin --push

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ================================================================================
    echo ✅ 成功上传到GitHub！
    echo ================================================================================
    echo.
    echo 仓库地址: https://github.com/%GIT_USERNAME%/%REPO_NAME%
    echo.
    echo 后续推送代码:
    echo   git add .
    echo   git commit -m "更新说明"
    echo   git push
    echo.
) else (
    echo.
    echo ❌ 上传失败！请检查错误信息。
    echo.
)

pause
