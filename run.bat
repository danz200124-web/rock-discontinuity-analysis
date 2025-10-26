@echo off
REM 岩体不连续面分析系统 - 高性能运行脚本
REM 自动使用所有CPU核心进行并行计算

echo ============================================================
echo 岩体不连续面自动检测与表征系统
echo 高性能多核并行计算模式
echo ============================================================
echo.

REM 检查Python是否安装
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到Python，请先安装Python 3.7+
    pause
    exit /b 1
)

echo [信息] 系统CPU核心数: 64核
echo [信息] 将使用全部CPU核心进行并行计算
echo.

REM 检查输入参数
if "%~1"=="" (
    echo [错误] 请提供输入点云文件路径
    echo.
    echo 使用方法:
    echo   run.bat ^<点云文件^> [输出目录] [配置文件]
    echo.
    echo 示例:
    echo   run.bat data-files/sample_data/zzm.ply
    echo   run.bat data-files/sample_data/zzm.ply output/
    echo   run.bat data-files/sample_data/zzm.ply output/ config.json
    echo.
    pause
    exit /b 1
)

REM 设置参数
set INPUT_FILE=%~1
set OUTPUT_DIR=%~2
set CONFIG_FILE=%~3

REM 默认值
if "%OUTPUT_DIR%"=="" set OUTPUT_DIR=output
if "%CONFIG_FILE%"=="" set CONFIG_FILE=config.json

REM 检查输入文件
if not exist "%INPUT_FILE%" (
    echo [错误] 输入文件不存在: %INPUT_FILE%
    pause
    exit /b 1
)

REM 显示配置
echo [配置信息]
echo   输入文件: %INPUT_FILE%
echo   输出目录: %OUTPUT_DIR%
echo   配置文件: %CONFIG_FILE%
echo   并行模式: 启用 (64核心)
echo.

REM 创建输出目录
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

REM 记录开始时间
echo [%date% %time%] 开始分析...
echo.

REM 运行程序
python src/main.py "%INPUT_FILE%" -o "%OUTPUT_DIR%" -c "%CONFIG_FILE%"

REM 检查运行结果
if errorlevel 1 (
    echo.
    echo [错误] 程序运行失败！
    pause
    exit /b 1
) else (
    echo.
    echo ============================================================
    echo [成功] 分析完成！
    echo ============================================================
    echo.
    echo 结果已保存至: %OUTPUT_DIR%
    echo   - parameters.json: 不连续面参数
    echo   - discontinuity_set_*.ply: 各组点云数据
    echo.
)

pause
