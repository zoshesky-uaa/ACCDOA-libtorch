curl.exe -L -o "libtorch.zip" "https://download.pytorch.org/libtorch/cu130/libtorch-win-shared-with-deps-debug-2.11.0%2Bcu130.zip"
Expand-Archive -LiteralPath "libtorch.zip" -DestinationPath $PSScriptRoot
if (Test-Path -LiteralPath "libtorch.zip") {
    Remove-Item -LiteralPath "libtorch.zip" -Force -ErrorAction Stop
}
