$externalDir = Join-Path $PSScriptRoot 'external'
if (-not (Test-Path -LiteralPath $externalDir)) {
    New-Item -ItemType Directory -Path $externalDir -Force | Out-Null
}
# Latest version to be compatible with Visual Studio 2026
$current_libtorch = "https://download.pytorch.org/libtorch/nightly/cu132/libtorch-win-shared-with-deps-debug-latest.zip"
$torchZip = Join-Path $externalDir 'libtorch.zip'
$torchDest = Join-Path $externalDir 'libtorch'
if (-not (Test-Path -LiteralPath $torchDest)) {
    Write-Host "Downloading LibTorch from $current_libtorch"
    curl.exe -L -o $torchZip $current_libtorch
    Write-Host "Extracting LibTorch..."
    Expand-Archive -Path $torchZip -DestinationPath $externalDir -Force
    Remove-Item -Path $torchZip -Force
     Write-Host "LibTorch Setup Complete."
} else {
    Write-Host "LibTorch already exists, skipping download."
}