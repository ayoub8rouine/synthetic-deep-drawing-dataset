[CmdletBinding()]
param(
    [string]$ShortcutPath = 'C:\ProgramData\Microsoft\Windows\Start Menu\Programs\LS-DYNA Suite R16.1 Student\LS-Run.lnk',
    [switch]$SkipRun
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$reportDir = Join-Path $projectRoot 'simulations\smoke_test'
$runDir = Join-Path $reportDir 'run'
$reportPath = Join-Path $reportDir 'diagnostic.json'
$deckPath = Join-Path $projectRoot 'templates\smoke_test.k'
New-Item -ItemType Directory -Force -Path $runDir | Out-Null

$shortcutTarget = $null
if (Test-Path -LiteralPath $ShortcutPath) {
    $shell = New-Object -ComObject WScript.Shell
    $shortcutTarget = $shell.CreateShortcut($ShortcutPath).TargetPath
}

$roots = @(
    'C:\Program Files\LS-DYNA Suite R16.1 Student',
    'C:\Program Files\LSTC',
    'C:\LSTC',
    'C:\Program Files\ANSYS Inc'
) | Where-Object { Test-Path -LiteralPath $_ }

$executables = foreach ($root in $roots) {
    Get-ChildItem -LiteralPath $root -Recurse -File -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -match '^(ls-dyna|lsdyna).*\.exe$' }
}
$solver = $executables |
    Where-Object { $_.Name -match 'studentversion' } |
    Sort-Object FullName -Descending |
    Select-Object -First 1
if (-not $solver) { $solver = $executables | Sort-Object FullName -Descending | Select-Object -First 1 }

$lsPrePost = @(foreach ($root in $roots) {
    Get-ChildItem -LiteralPath $root -Recurse -File -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -match '^lsprepost.*\.exe$' }
}) | Select-Object -First 1

$exitCode = $null
$termination = $false
$fatalErrors = @()
$commandLineWorks = $false
if (-not $SkipRun -and $solver) {
    Copy-Item -LiteralPath $deckPath -Destination (Join-Path $runDir 'smoke_test.k') -Force
    # The Ansys-distributed Student executable requires this license mode. Setting
    # it here affects this script and its solver child only; no persistent user or
    # machine environment variable is created.
    $env:LSTC_LICENSE = 'ANSYS'
    $process = Start-Process -FilePath $solver.FullName -ArgumentList 'i=smoke_test.k','memory=20m','ncpu=1' -WorkingDirectory $runDir -Wait -PassThru -NoNewWindow
    $exitCode = $process.ExitCode
    $messagePath = Join-Path $runDir 'messag'
    if (Test-Path -LiteralPath $messagePath) {
        $message = Get-Content -LiteralPath $messagePath -Raw
        $termination = $message -match 'N o r m a l\s+t e r m i n a t i o n|normal termination'
        $fatalErrors = @([regex]::Matches($message, '(?im)^.*(?:fatal|error\s+termination|E\s+r\s+r\s+o\s+r\s+t\s+e\s+r\s+m\s+i\s+n\s+a\s+t\s+i\s+o\s+n).*$') | ForEach-Object Value)
    }
    $commandLineWorks = ($exitCode -eq 0 -and $termination -and $fatalErrors.Count -eq 0)
}

$solverVersion = if ($solver) { $solver.VersionInfo.ProductVersion } else { $null }
if (-not $solverVersion -and $solver) {
    $solverVersion = [regex]::Match($solver.Name, 'R\d+(?:\.\d+)?(?:_\d+)?').Value
}
$result = [ordered]@{
    timestamp = (Get-Date).ToString('o')
    shortcut_path = $ShortcutPath
    ls_run_path = $shortcutTarget
    solver_path = if ($solver) { $solver.FullName } else { $null }
    ls_prepost_path = if ($lsPrePost) { $lsPrePost.FullName } else { $null }
    solver_version = $solverVersion
    student_limit_nodes = 128000
    student_limit_elements = 128000
    smoke_test_attempted = -not $SkipRun
    smoke_test_exit_code = $exitCode
    normal_termination = $termination
    fatal_errors = $fatalErrors
    command_line_run_works = $commandLineWorks
    exact_command = if ($solver) { '& "{0}" i=smoke_test.k memory=20m ncpu=1' -f $solver.FullName } else { $null }
}
$result | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $reportPath -Encoding utf8
$result | Format-List
Write-Output "Diagnostic report: $reportPath"
if (-not $solver) { exit 2 }
if (-not $SkipRun -and -not $commandLineWorks) { exit 3 }
