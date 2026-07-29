' Launches the seizure review dashboard with no visible terminal.
' If the dashboard is already running, it just opens the browser on it, so
' double-clicking the icon twice is always safe.
Option Explicit

Const PORT = 5006
Const WAIT_SECONDS = 120

Dim fso, shell, here, url, logfile, cmd, waited

Set fso = CreateObject("Scripting.FileSystemObject")
Set shell = CreateObject("WScript.Shell")

' repo root = parent of the folder holding this script
here = fso.GetParentFolderName(fso.GetParentFolderName(WScript.ScriptFullName))
shell.CurrentDirectory = here
url = "http://localhost:" & PORT & "/dashboard"
logfile = fso.BuildPath(fso.GetParentFolderName(WScript.ScriptFullName), "last_run.log")

If Not fso.FolderExists(fso.BuildPath(here, "hexoreview")) Then
    MsgBox "Cannot find the review program in:" & vbCrLf & here & _
           vbCrLf & vbCrLf & "Please contact the study coordinator.", _
           16, "Seizure review"
    WScript.Quit 1
End If

' Already running from an earlier session? Just show it.
If ServerUp(url) Then
    shell.Run url, 1, False
    WScript.Quit 0
End If

' Start it hidden, keeping the output so failures can be diagnosed.
cmd = "cmd /c uv run hexoreview run --no-browser --port " & PORT & _
      " > """ & logfile & """ 2>&1"
shell.Run cmd, 0, False

waited = 0
Do While waited < WAIT_SECONDS
    WScript.Sleep 2000
    waited = waited + 2
    If ServerUp(url) Then
        shell.Run url, 1, False
        WScript.Quit 0
    End If
Loop

MsgBox "The seizure review dashboard did not start." & vbCrLf & vbCrLf & _
       "Details were saved to:" & vbCrLf & logfile & vbCrLf & vbCrLf & _
       "Please send that file to the study coordinator.", 16, "Seizure review"
WScript.Quit 1


Function ServerUp(address)
    Dim http
    ServerUp = False
    On Error Resume Next
    Set http = CreateObject("MSXML2.ServerXMLHTTP.6.0")
    If Err.Number <> 0 Then Exit Function
    http.setTimeouts 2000, 2000, 3000, 3000
    http.Open "GET", address, False
    http.Send
    If Err.Number = 0 Then
        If http.Status = 200 Then ServerUp = True
    End If
    Err.Clear
    On Error GoTo 0
End Function