# Creates a desktop shortcut that launches the review dashboard.
#
# The shortcut runs uv.exe directly, with no .vbs or .bat in between, because
# hospital policy commonly blocks scripts from starting other programs.
#
# Run once per reviewer machine, from the repo root:
#     powershell -ExecutionPolicy Bypass -File launch\create_shortcut.ps1

$repo = Split-Path -Parent $PSScriptRoot
$icon = Join-Path $PSScriptRoot 'seizure_review.ico'
$desktop = [Environment]::GetFolderPath('Desktop')
$link = Join-Path $desktop 'Seizure Review.lnk'

$uv = Get-Command uv -ErrorAction SilentlyContinue
if (-not $uv) {
    Write-Error "uv is not on PATH for this user. Install uv, or sign in as the reviewer and run this again."
    exit 1
}

$shell = New-Object -ComObject WScript.Shell
$sc = $shell.CreateShortcut($link)
$sc.TargetPath = $uv.Source
$sc.Arguments = 'run hexoreview run'
$sc.WorkingDirectory = $repo
$sc.Description = 'Overnight seizure review'
$sc.WindowStyle = 7          # start minimised
if (Test-Path $icon) { $sc.IconLocation = $icon }
$sc.Save()

Write-Host "Desktop shortcut created: $link"
Write-Host "Target : $($uv.Source) run hexoreview run"
Write-Host "Folder : $repo"
Write-Host ""
Write-Host "Test it now by double-clicking the icon on the desktop."