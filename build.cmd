@echo off
rem Build sa3.cpp for one backend into its own build dir (backends coexist for A/B testing).
rem
rem Usage: build.cmd [cpu^|cuda^|vulkan^|all]   (default: cpu)
rem   cpu     -> build\         portable, no GPU
rem   cuda    -> build-cuda\    NVIDIA (needs CUDA Toolkit; if the VS CUDA targets complain,
rem                             set CUDA_PATH_V12_8 to your CUDA dir, see docs/DISTRIBUTION.md)
rem   vulkan  -> build-vulkan\  any GPU (needs the Vulkan SDK; open a fresh shell after install)
rem   all     -> build-all\     one binary, CUDA+Vulkan loaded at runtime (GGML_BACKEND_DL)
setlocal EnableDelayedExpansion

set "BACKEND=%~1"
if "%BACKEND%"=="" set "BACKEND=cpu"

rem locate cmake (PATH, else the copy bundled with Visual Studio 2022)
where cmake >nul 2>nul
if errorlevel 1 (
    set "VSCMAKE=C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin"
    if exist "!VSCMAKE!\cmake.exe" set "PATH=!VSCMAKE!;%PATH%"
)

set "GEN=-G "Visual Studio 17 2022" -A x64"

if /i "%BACKEND%"=="cpu" (
    set "DIR=build" & set "FLAGS=-DGGML_CUDA=OFF"
) else if /i "%BACKEND%"=="cuda" (
    set "DIR=build-cuda" & set "FLAGS=-DSA3_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native"
) else if /i "%BACKEND%"=="vulkan" (
    set "DIR=build-vulkan" & set "FLAGS=-DSA3_VULKAN=ON"
    if "%VULKAN_SDK%"=="" echo [sa3] WARNING: VULKAN_SDK not set - install the Vulkan SDK and open a fresh shell.
) else if /i "%BACKEND%"=="all" (
    set "DIR=build-all" & set "FLAGS=-DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON -DSA3_CUDA=ON -DSA3_VULKAN=ON"
) else (
    echo unknown backend: %BACKEND%  ^(cpu^|cuda^|vulkan^|all^)& exit /b 1
)

echo [sa3] configuring %BACKEND% -^> %DIR%\
cmake -S . -B %DIR% %GEN% -DCMAKE_BUILD_TYPE=Release %FLAGS%
if errorlevel 1 exit /b 1
echo [sa3] building ...
cmake --build %DIR% --config Release
if errorlevel 1 exit /b 1
echo [sa3] done -^> %DIR%\bin\Release\

rem Write env files so the tools run from any dir without the long path.
set "BIN=%CD%\%DIR%\bin\Release"
> env.cmd echo @set "PATH=%BIN%;%%PATH%%"
>> env.cmd echo @set "SA3_MODELS_DIR=%CD%\models"
> env.ps1 echo $env:Path = "%BIN%;$env:Path"
>> env.ps1 echo $env:SA3_MODELS_DIR = "%CD%\models"
echo [sa3] to run the tools from any dir:
echo [sa3]   cmd:         env.cmd
echo [sa3]   powershell:  . .\env.ps1
echo [sa3]   then, e.g.:  sa3-generate --model medium --prompt "..." --out song.wav
