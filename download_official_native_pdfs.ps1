$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$dest = Join-Path $repoRoot "gold_dataset\source_pdfs"
New-Item -ItemType Directory -Force -Path $dest | Out-Null

$files = @(
    @{Name="01_coherent_ftlx2672d3xx_10g_bidi_30km_sfpplus.pdf"; Url="https://www.coherent.com/resources/datasheet/networking/optical-transceivers/sfp/ftlx2672d3xx-transceiver-ds.pdf"},
    @{Name="02_coherent_ftlc1154sdpl_100g_lr4_10km_qsfp28.pdf"; Url="https://www.coherent.com/resources/datasheet/networking/optical-transceivers/qsfp/ftlc1154sdpl-transceiver-ds.pdf"},
    @{Name="03_coherent_ftlx2471dc0xx_10g_cwdm_lr_sfpplus.pdf"; Url="https://www.coherent.com/resources/datasheet/networking/optical-transceivers/sfp/ftlx2471dc0xx-transceiver-ds.pdf"},
    @{Name="04_coherent_ftlf8536p5xyz_25g_sr_multirate_sfp28.pdf"; Url="https://www.coherent.com/resources/datasheet/networking/optical-transceivers/sfp/ftlf8536p5xyz-transceiver-ds.pdf"},
    @{Name="05_coherent_ftlf8536w4btv_25g_sr_ecpri_sfp28.pdf"; Url="https://www.coherent.com/resources/datasheet/networking/optical-transceivers/sfp/ftlf8536w4btv-transceiver-ds.pdf"},
    @{Name="06_coherent_ftcd4523e3pcm_400g_dr4_qsfpdd.pdf"; Url="https://www.coherent.com/content/dam/coherent/site/en/resources/datasheet/networking/optical-transceivers/qsfp/ftcd4523e3pcm-transceiver-ds.pdf"},
    @{Name="07_coherent_ftcd8613e2pcm_400g_sr8_qsfpdd.pdf"; Url="https://www.coherent.com/resources/datasheet/networking/optical-transceivers/qsfp/ftcd8613e2pcm-transceiver-ds.pdf"},
    @{Name="08_coherent_ftcd4323e2pcl_400g_lr4_10km_qsfpdd.pdf"; Url="https://www.coherent.com/resources/datasheet/networking/optical-transceivers/qsfp/ftcd4323e2pcl-transceiver-ds.pdf"},
    @{Name="09_coherent_ftl410qe4c_40g_sr4_qsfpplus.pdf"; Url="https://www.coherent.com/resources/datasheet/networking/optical-transceivers/qsfp/ftl410qe4c-transceiver-ds.pdf"},
    @{Name="10_coherent_ftl410qd4n_40g_sr4_extended_temp_qsfpplus.pdf"; Url="https://www.coherent.com/content/dam/coherent/site/en/resources/datasheet/networking/optical-transceivers/qsfp/ftl410qd4n-transceiver-ds.pdf"}
)

Write-Host "Downloading official native-text manufacturer datasheets..." -ForegroundColor Cyan
foreach ($f in $files) {
    $out = Join-Path $dest $f.Name
    Write-Host "  -> $($f.Name)"
    Invoke-WebRequest -Uri $f.Url -OutFile $out -UseBasicParsing
    if ((Get-Item $out).Length -lt 10000) {
        throw "Downloaded file looks too small: $out"
    }
}

Write-Host "`nDownload complete." -ForegroundColor Green
Write-Host "Development PDFs: $((Get-ChildItem $dest -Filter *.pdf).Count)"
Write-Host "Now run: python gold_pipeline\preflight_corpus.py"
