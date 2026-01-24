# Полное обновление репозитория: в remote будет ровно то, что в проекте.
# Запускать из корня проекта (где main.py).
# Параметр -Force: без вопроса "Commit? (y/n)".

param([switch]$Force)

$ErrorActionPreference = "Stop"

# Переход в корень репозитория (каталог скрипта)
$root = $PSScriptRoot
if (-not $root) { $root = (Get-Location).Path }
Set-Location $root

Write-Host "Project root: $root" -ForegroundColor Cyan
Write-Host ""

# 0) Перестать отслеживать каталоги/файлы из .gitignore, если они уже в репо
@('outputs', 'outputsTest', 'Resulst') | ForEach-Object { git rm -r --cached $_ 2>$null }
@('comparison_results.json', 'my_dhfs_data.json') | ForEach-Object { git rm --cached $_ 2>$null }

# 1) Учесть все изменения: новые, изменённые, удалённые (в т.ч. docs, notebooks, если их нет на диске)
git add -A

# 2) Показать, что попадёт в коммит
Write-Host "=== Staged changes ===" -ForegroundColor Yellow
git status

Write-Host ""
if (-not $Force) {
    $ans = Read-Host "Commit? (y/n)"
    if ($ans -ne 'y' -and $ans -ne 'Y') {
        Write-Host "Aborted. Run 'git reset' to unstage if needed."
        exit 0
    }
}

git commit -m "Sync repo with project: pipeline (step2, GA+HHO), initial_weights, robust predecessor lambdas; remove obsolete docs/notebooks; update .gitignore, README"
if ($LASTEXITCODE -ne 0) {
    Write-Host "Commit failed (maybe nothing to commit)." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Done. To upload: git push origin main" -ForegroundColor Green
