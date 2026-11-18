# Update the table of contents / page fields in Word and export the report to PDF.
# Run from the CourseWork folder:  powershell -ExecutionPolicy Bypass -File report\build\export_pdf.ps1
$ErrorActionPreference = "Stop"
$docx = (Resolve-Path "report\FCVG101_Coursework_Report.docx").Path
$pdf = [System.IO.Path]::ChangeExtension($docx, ".pdf")
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
try {
    $doc = $word.Documents.Open($docx, $false, $false)
    $doc.Repaginate()
    foreach ($toc in $doc.TablesOfContents) { $toc.Update() }
    $doc.Fields.Update() | Out-Null
    foreach ($toc in $doc.TablesOfContents) { $toc.UpdatePageNumbers() }
    $doc.Save()
    # 17 = wdExportFormatPDF; OptimizeFor 1 = minimum size (CMS limit 2 MB); CreateBookmarks 1 = from headings
    $doc.ExportAsFixedFormat($pdf, 17, $false, 1, 0, 0, 0, 0, $true, $true, 1, $true, $true, $false)
    "pages: " + $doc.ComputeStatistics(2)
    $doc.Close($false)
} finally {
    $word.Quit()
}
"PDF: $pdf  size: {0:N2} MB" -f ((Get-Item $pdf).Length / 1MB)
