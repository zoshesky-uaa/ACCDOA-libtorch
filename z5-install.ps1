curl.exe -L -o "z5.zip" "https://github.com/constantinpape/z5/archive/refs/heads/master.zip"
Expand-Archive -LiteralPath "z5.zip" -DestinationPath $PSScriptRoot -Force
if (Test-Path -LiteralPath "z5.zip") {
    Remove-Item -LiteralPath "z5.zip" -Force -ErrorAction Stop
}

$src  = Join-Path $PSScriptRoot 'z5-master\include\z5'
$dest = Join-Path $PSScriptRoot 'z5'
if (Test-Path -LiteralPath $src) {
    Move-Item -LiteralPath $src -Destination $dest -ErrorAction Stop
    Remove-Item -LiteralPath (Join-Path $PSScriptRoot 'z5-master') -Recurse -Force -ErrorAction Stop
}

$files = Join-Path $PSScriptRoot 'zarr-files'
if (-not (Test-Path -LiteralPath $files)) {
    New-Item -ItemType Directory -Path $files -Force | Out-Null
}