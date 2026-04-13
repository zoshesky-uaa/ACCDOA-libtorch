$externalDir = Join-Path $PSScriptRoot 'external'
if (-not (Test-Path -LiteralPath $externalDir)) {
    New-Item -ItemType Directory -Path $externalDir -Force | Out-Null
}
# libtorch, precompiled for Windows
$current_libtorch = "https://download.pytorch.org/libtorch/cu130/libtorch-win-shared-with-deps-debug-2.11.0%2Bcu130.zip"
$torchZip = Join-Path $externalDir 'libtorch.zip'
$torchDest = Join-Path $externalDir 'libtorch'
if (-not (Test-Path -LiteralPath $torchDest)) {
    Write-Host "Downloading libtorch..."
    curl.exe -L -o $torchZip $current_libtorch
    Write-Host "Extracting LibTorch..."
    Expand-Archive -Path $torchZip -DestinationPath $externalDir -Force
    Write-Host "Cleaning up ZIP file..."
    Remove-Item -Path $torchZip -Force
} else {
    Write-Host "LibTorch already exists, skipping download."
}
