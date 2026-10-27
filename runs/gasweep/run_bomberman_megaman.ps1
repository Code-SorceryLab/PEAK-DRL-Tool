# GA sweep for Bomberman then Mega Man: same settings as the finished Mario / Meat Boy sweep.
# Stop with Ctrl+C in this window; finished game x persona sets are kept.
Set-Location "C:\Users\Kevin\Desktop\GITHUB\PEAK-DRL-Tool"
$log = "runs\gasweep\sweep_bomberman_megaman.log"
$personas = "experienced", "clumsy", "collector", "killer", "novice", "speedrunner"
foreach ($game in "bomberman", "megaman") {
    foreach ($persona in $personas) {
        # resume: skip sets an earlier launch finished (an interrupted set restarts, so delete its folder first)
        if ((Test-Path $log) -and (Select-String -Path $log -SimpleMatch "END $game $persona (exit 0)" -Quiet)) { continue }
        "[$(Get-Date -Format 'MM-dd HH:mm')] START $game $persona" | Tee-Object -FilePath $log -Append
        & .\.venv\Scripts\python.exe -m code.neuro.gasweep --game $game --persona $persona --gens 50 `
            --seeds 101 202 303 404 505 --sensors rays --workers 10 --confirm 2>&1 |
            Tee-Object -FilePath $log -Append
        "[$(Get-Date -Format 'MM-dd HH:mm')] END $game $persona (exit $LASTEXITCODE)" | Tee-Object -FilePath $log -Append
    }
}
"[$(Get-Date -Format 'MM-dd HH:mm')] ALL DONE" | Tee-Object -FilePath $log -Append
