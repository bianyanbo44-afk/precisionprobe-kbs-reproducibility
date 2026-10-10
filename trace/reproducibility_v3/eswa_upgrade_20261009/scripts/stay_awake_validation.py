"""Keep Windows awake only while this validation pipeline can make progress.

This creates a thread-scoped system execution request; it does not edit power
plans, keep the display on, or prevent explicit sleep/lid actions. Windows
removes the request when the helper exits, including if its process is killed.
"""
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

BASE = Path(__file__).resolve().parents[1]
RUN = BASE / "runs/livecodebench_v6_increment"


def main():
    if sys.platform != "win32":
        raise RuntimeError("this operational helper is Windows-only")
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.SetThreadExecutionState.argtypes = [wintypes.DWORD]
    kernel.SetThreadExecutionState.restype = wintypes.DWORD
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    attempts = [json.loads(x) for x in (RUN / "logs/generation_attempts.jsonl").read_text(encoding="utf-8").splitlines()]
    pid = next(r["pid"] for r in reversed(attempts) if r["status"] == "started")
    handle = kernel.OpenProcess(0x100000, False, pid)
    if not handle:
        raise OSError(ctypes.get_last_error(), "cannot monitor generation process")
    status_file = RUN / "keep_awake_status.json"

    def status(stage):
        status_file.write_text(json.dumps({"updated_at_utc": datetime.now(timezone.utc).isoformat(),
                                          "generation_pid": pid, "stage": stage,
                                          "display_unchanged": True, "power_plan_unchanged": True}, indent=2), encoding="utf-8")

    try:
        if not kernel.SetThreadExecutionState(0x80000001):
            raise OSError(ctypes.get_last_error(), "cannot request temporary system wakefulness")
        deadline = time.monotonic() + 24*60*60
        while time.monotonic() < deadline:
            alive = kernel.WaitForSingleObject(handle, 0) == 258
            if (RUN / "evidence_ready.json").exists():
                break
            if not alive:
                pipeline = json.loads((RUN / "pipeline_status.json").read_text(encoding="utf-8"))
                complete_generations = len(list(RUN.glob("*_generation_complete.json"))) == 6
                if pipeline.get("stage") not in ("evaluating", "analyzing") and not complete_generations:
                    break
                if pipeline.get("stage") == "error":
                    break
            status("temporary_system_request_active")
            time.sleep(30)
    finally:
        kernel.SetThreadExecutionState(0x80000000)
        kernel.CloseHandle(handle)
        status("temporary_system_request_released")


if __name__ == "__main__":
    main()
