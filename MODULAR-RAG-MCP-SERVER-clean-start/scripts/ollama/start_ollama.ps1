param(
    [string]$BaseUrl = "http://localhost:11434",
    [string]$Model = "bge-m3",
    [int]$TimeoutSeconds = 30
)

$ErrorActionPreference = "Stop"

if ($TimeoutSeconds -lt 1) {
    throw "TimeoutSeconds must be greater than zero."
}

$ollamaCommand = Get-Command ollama -ErrorAction SilentlyContinue
if (-not $ollamaCommand) {
    throw "Ollama is not installed or is not available in PATH."
}

function Get-OllamaTags {
    try {
        return Invoke-RestMethod `
            -Uri "$BaseUrl/api/tags" `
            -Method Get `
            -TimeoutSec 2
    }
    catch {
        return $null
    }
}

$tags = Get-OllamaTags
if (-not $tags) {
    Write-Host "Starting Ollama service at $BaseUrl ..."
    Start-Process `
        -FilePath $ollamaCommand.Source `
        -ArgumentList "serve" `
        -WindowStyle Hidden

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    do {
        Start-Sleep -Seconds 1
        $tags = Get-OllamaTags
    } while (-not $tags -and (Get-Date) -lt $deadline)
}

if (-not $tags) {
    throw "Ollama did not become ready within $TimeoutSeconds seconds."
}

Write-Host "Ollama service is ready at $BaseUrl."

$modelInstalled = @($tags.models | ForEach-Object { $_.name }) | Where-Object {
    $_ -eq $Model -or $_ -like "${Model}:*"
}

if ($modelInstalled) {
    Write-Host "Embedding model '$Model' is installed."
}
else {
    Write-Warning "Embedding model '$Model' is not installed. Run: ollama pull $Model"
}
