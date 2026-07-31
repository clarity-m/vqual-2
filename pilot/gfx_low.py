"""
Force every UE4 scalability group to Low in the simulator's GameUserSettings.ini.

Why this exists: the sim's GUI exposes only graphics and sound, and of the scalability
groups only sg.ShadingQuality was ever set - everything else was running at engine
defaults, which on UE4 means high shadows, post-processing, effects and view distance.
Those dominate frame time far more than resolution does, and the sim was showing
LOW FRAMERATE at 1280x720.

Re-runnable and idempotent, because **the sim rewrites this file when it exits**. A
hand edit made while the sim is running is silently lost on close. Run this with the
sim CLOSED, then launch.

    python3 pilot/gfx_low.py            apply
    python3 pilot/gfx_low.py --show     print the current settings, change nothing
    python3 pilot/gfx_low.py --restore  put back the most recent backup

NOT set here: sg.ResolutionQuality, which is a 0-100 percentage rather than a 0-3
level. Dropping it renders below native and upscales - a large win, but the FPV camera
stream is probably rendered by the same pipeline, and blurring the images we intend to
build a detector on is a bad trade. Left alone deliberately.

CAVEAT worth remembering: post-processing is what produces the bloom flare on gate 1.
Lowering it changes how gates look in the camera stream - probably for the better, but
it means the detector must be tuned at whatever settings are actually raced with. Pick
a configuration and keep it fixed.
"""

import os
import shutil
import sys
import time

INI = os.path.expandvars(
    r"%LOCALAPPDATA%\FlightSim\Saved\Config\WindowsNoEditor\GameUserSettings.ini")

SECTION = "[ScalabilityGroups]"
LOW = {
    "sg.ViewDistanceQuality": "0",
    "sg.AntiAliasingQuality": "0",
    "sg.ShadowQuality": "0",
    "sg.PostProcessQuality": "0",
    "sg.TextureQuality": "0",
    "sg.EffectsQuality": "0",
    "sg.FoliageQuality": "0",
    "sg.ShadingQuality": "0",
}


def read_lines(path):
    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        return f.read().splitlines()


def write_lines(path, lines):
    # UE writes CRLF; keep it that way so the file looks untouched to the engine.
    with open(path, "w", encoding="utf-8", newline="\r\n") as f:
        f.write("\n".join(lines) + "\n")


def current(lines):
    out, in_sec = {}, False
    for ln in lines:
        s = ln.strip()
        if s.startswith("["):
            in_sec = (s == SECTION)
            continue
        if in_sec and "=" in s:
            k, v = s.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def apply(lines):
    """Replace the keys we manage inside [ScalabilityGroups], preserving all else."""
    managed = set(LOW)
    out, in_sec, wrote, seen_section = [], False, False, False

    for ln in lines:
        s = ln.strip()
        if s.startswith("["):
            if in_sec and not wrote:
                out.extend("%s=%s" % kv for kv in sorted(LOW.items()))
                wrote = True
            in_sec = (s == SECTION)
            if in_sec:
                seen_section = True
            out.append(ln)
            continue
        if in_sec:
            key = s.split("=", 1)[0].strip() if "=" in s else ""
            if key in managed:
                continue                      # drop; rewritten in one block below
            if s == "" and not wrote:
                out.extend("%s=%s" % kv for kv in sorted(LOW.items()))
                wrote = True
            out.append(ln)
            continue
        out.append(ln)

    if in_sec and not wrote:
        out.extend("%s=%s" % kv for kv in sorted(LOW.items()))
        wrote = True
    if not seen_section:
        out = [SECTION] + ["%s=%s" % kv for kv in sorted(LOW.items())] + [""] + out
    return out


def main():
    if not os.path.exists(INI):
        sys.exit("not found: %s" % INI)

    if "--restore" in sys.argv:
        backups = sorted(b for b in os.listdir(os.path.dirname(INI))
                         if b.startswith("GameUserSettings.ini.bak"))
        if not backups:
            sys.exit("no backup to restore")
        src = os.path.join(os.path.dirname(INI), backups[-1])
        shutil.copy2(src, INI)
        print("restored %s" % src)
        return

    lines = read_lines(INI)
    if "--show" in sys.argv:
        cur = current(lines)
        print("[ScalabilityGroups] in %s" % INI)
        for k in sorted(set(list(LOW) + list(cur))):
            print("  %-24s %s" % (k, cur.get(k, "(unset -> engine default)")))
        return

    backup = "%s.bak-%s" % (INI, time.strftime("%Y%m%d-%H%M%S"))
    shutil.copy2(INI, backup)
    before = current(lines)
    write_lines(INI, apply(lines))
    after = current(read_lines(INI))

    print("backup: %s\n" % backup)
    print("  %-24s %-28s %s" % ("setting", "before", "after"))
    for k in sorted(LOW):
        print("  %-24s %-28s %s"
              % (k, before.get(k, "(unset -> default)"), after.get(k)))
    print("\nRestart the simulator for these to take effect.")
    print("The sim rewrites this file on exit, so if the setting reverts, close the "
          "sim and run this again before launching.")


if __name__ == "__main__":
    main()
