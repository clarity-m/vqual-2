# init after new vm session
powershell -NoProfile -ExecutionPolicy Bypass -File Z:\claude-drop\runner.ps1

# flight sim exe paths — TWO builds ship side by side
#   VQ2 (no pose telemetry): AIGP_3391FlightSim.exe
#   VQ1 (streams ATTITUDE/LOCAL_POSITION_NED/ODOMETRY): AIGP_VQ1_3391FlightSim.exe
C:\Users\clarity\AppData\Local\aigp\AIGP_3391\FlightSim.exe
 - physics is realtime keep 30 fps


# teleop record to local disk
python Z:\vqual-2\pilot\teleop.py --sessions C:\Users\clarity\AppData\Local\aigp\sessions
 - remember default is record pass --no-frames for frames --no-record if necessary

# zips up local session (session name printed after teleop F8)
tar -cf "$env:LOCALAPPDATA\aigp\session.tar" -C "$env:LOCALAPPDATA\aigp\sessions" <session-name>

---

## new-VM bootstrap (Horizon hands you a different machine each time)

Order matters. Cloudpaging virtualizes session-wide, but only into shells started
after activation.

1. AppsAnywhere -> launch **Anaconda 2024.10** (and Unreal Engine 5.8.0 if needed).
2. Open the **Anaconda PowerShell**
3. Start the runner (drive letter varies per session; use what Explorer shows):
   powershell -NoProfile -ExecutionPolicy Bypass -File Z:\claude-drop\runner.ps1
4. Copy the sim to local disk (~11 s, 4.62 GB): 
   robocopy "$env:USERPROFILE\Documents\AIGP_3391" "$env:LOCALAPPDATA\aigp\AIGP_3391" /E /R:3 /W:5 /MT:8
5. In the sim's video settings: **turn VSync OFF** 

## what persists, what does not

* Roams with the CAEN profile (N:) -- survives a new VM:
  pip --user packages (pymavlink, opencv-python 4.10, numpy 1.26.4, keyboard),
  Documents\AIGP_3391, the pilot repo on the laptop side.
* Local to each VM -- redo every session:
  the `%LOCALAPPDATA%\aigp` sim copy, cloudpaged Anaconda, `GameUserSettings.ini`
  (so vsync + scalability reset to Cinematic defaults on a fresh machine).

## paths and processes

* sim (local copy)  %LOCALAPPDATA%\aigp\AIGP_3391\FlightSim.exe
* sim settings ini  %LOCALAPPDATA%\FlightSim\Saved\Config\WindowsNoEditor\GameUserSettings.ini
* real sim process  DCGame-Win64-Shipping   (FlightSim.exe is only a 1-thread launcher)
* kill a hung sim   Stop-Process -Name DCGame-Win64-Shipping -Force

## graphics

# apply Low scalability -- sim must be CLOSED (it rewrites the ini on exit)
python Z:\vqual-2\pilot\gfx_low.py
python Z:\vqual-2\pilot\gfx_low.py --show

## moving data back to the laptop

NEVER robocopy recordings across Z: -- it is the Blast client redirect at ~89 ms per
file. vqual-2\pilot\sessions is 212k files; that copy takes hours. Bulk transfer is
fast (~400 MB/s), per-file is not. So archive on the VM into ONE file.

Use tar, not Compress-Archive: the latter is very slow in PS 5.1 for many files, and
the frames are JPEGs, so compressing them again buys nothing. tar.exe ships with
Windows (C:\Windows\System32\tar.exe).

    # ON THE VM -- writes straight to Z:, no separate copy step.
    # -C means "cd there first", so the archive contains <session>/... with no
    # parent path baked in.
    tar -cf "Z:\vqual-2\pilot\sessions\<session>.tar" -C "$env:LOCALAPPDATA\aigp\sessions" <session>

    # ON THE LAPTOP -- -C sets the destination explicitly. Without it, tar extracts
    # relative to the current directory and you get sessions\sessions\<session>.
    tar -xf <session>.tar -C "C:\Users\USER\Projects\vqual-2\pilot\sessions"

If you only need telemetry and not the JPEGs, skip the archive entirely -- the CSVs are
~1 MB across 7 files and copy in ~4 s:

    robocopy "$env:LOCALAPPDATA\aigp\sessions\<session>" "Z:\vqual-2\pilot\sessions\<session>" *.csv *.jsonl /R:1 /W:1

Code is fine to run directly off Z: -- only ~40 files of imports, and edits made on
the laptop are visible to the VM immediately with no staging step.

## gotchas

* UDP 14550 is exclusive -- teleop and the pilot cannot both run.
* `python` in a non-Anaconda shell is a 0-byte Store stub, not an error you can read.
* C:\ root is not writable by a standard domain user; use %LOCALAPPDATA%.
* Do not `pip install -r requirements.txt` into the conda base: it is unpinned, pulls
  opencv 5 -> numpy 2, and that shadows conda's numpy 1.26.4 and breaks matplotlib,
  scipy and numba. Pin instead:
  pip install --user "numpy==1.26.4" "opencv-python==4.10.0.84" pymavlink keyboard
