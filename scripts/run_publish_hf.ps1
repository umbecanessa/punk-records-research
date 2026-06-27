# Publish kernel weights to Hugging Face Hub
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

if (-not $env:HF_REPO_ID) {
    Write-Host "Set HF_REPO_ID, e.g. YOUR_HF_USER/punk-records-research-kernel-v0.1" -ForegroundColor Yellow
    exit 1
}

pip install -q -e ".[hub]" 2>$null
python -u scripts/publish_hf.py --repo-id $env:HF_REPO_ID @args
