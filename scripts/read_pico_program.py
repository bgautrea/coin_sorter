#!/usr/bin/env python3
"""Dump the MicroPython source files running on the Pico belt controller.

The Pico enumerates as a MicroPython USB-CDC device (vid:pid 2e8a:0005), so
the "firmware" is just .py files in its on-board filesystem. This script
opens the same serial port the runtime uses, drops the running program into
the raw REPL, and reads the files back over the wire.

Stop the main sorter loop first — the port is exclusive.

Quick CLI alternative if you'd rather not run this:

    pip install mpremote
    mpremote ls
    mpremote cat :main.py

Usage:
    python scripts/read_pico_program.py                 # dump every .py
    python scripts/read_pico_program.py --file main.py  # dump one file
    python scripts/read_pico_program.py --port /dev/ttyACM1
"""
from __future__ import annotations

import argparse
import sys
import time

import serial

CTRL_A = b"\x01"  # enter raw REPL
CTRL_B = b"\x02"  # exit raw REPL → friendly REPL
CTRL_C = b"\x03"  # interrupt
CTRL_D = b"\x04"  # soft reset (in friendly REPL) / end-of-input (in raw REPL)

RAW_PROMPT = b"raw REPL; CTRL-B to exit\r\n>"
OK = b"OK"


def _read_until(ser: serial.Serial, needle: bytes, timeout_s: float = 3.0) -> bytes:
    deadline = time.monotonic() + timeout_s
    buf = bytearray()
    while time.monotonic() < deadline:
        chunk = ser.read(256)
        if chunk:
            buf.extend(chunk)
            if needle in buf:
                return bytes(buf)
        else:
            time.sleep(0.01)
    raise TimeoutError(f"timed out waiting for {needle!r}; got {bytes(buf)!r}")


def enter_raw_repl(ser: serial.Serial) -> None:
    ser.reset_input_buffer()
    # Two Ctrl-Cs guarantees we break out of any blocking input/loop.
    ser.write(CTRL_C + CTRL_C)
    time.sleep(0.1)
    ser.reset_input_buffer()
    ser.write(CTRL_A)
    try:
        _read_until(ser, RAW_PROMPT, timeout_s=2.0)
    except TimeoutError as e:
        raise RuntimeError(
            "Could not enter raw REPL. The running firmware may be trapping "
            "KeyboardInterrupt. Recovery: hold BOOTSEL while plugging the "
            "Pico in, reflash a stock MicroPython UF2, then re-upload."
        ) from e


def exit_raw_repl(ser: serial.Serial) -> None:
    ser.write(CTRL_B)
    time.sleep(0.05)
    ser.reset_input_buffer()


def restart_firmware(ser: serial.Serial, entry: str = "sorter.py") -> None:
    """Re-exec the firmware entry point in raw REPL, then disconnect.

    The Pico has no main.py, so soft reset would just drop to the REPL.
    Running ``exec(open(entry).read())`` via raw REPL + Ctrl-D launches the
    script; closing the port afterwards leaves it running (the mpremote
    ``run`` pattern). If the entry file isn't present this is a no-op.
    """
    try:
        listing = exec_on_pico(ser, "import os\nprint('\\n'.join(os.listdir()))")
    except Exception:
        return
    if entry not in listing.splitlines():
        return
    ser.write(f"exec(open({entry!r}).read())".encode("utf-8") + CTRL_D)
    ser.read(2)  # consume "OK" ack; the script then runs until disconnect


def exec_on_pico(ser: serial.Serial, code: str, timeout_s: float = 5.0) -> str:
    """Run `code` in the Pico raw REPL and return its stdout. Raises on stderr."""
    ser.reset_input_buffer()
    ser.write(code.encode("utf-8") + CTRL_D)
    # Pico replies with "OK" once it has accepted the code.
    ack = ser.read(2)
    if ack != OK:
        raise RuntimeError(f"raw REPL did not ack code (got {ack!r})")
    # Output frame: <stdout>\x04<stderr>\x04>
    framed = _read_until(ser, b"\x04>", timeout_s=timeout_s)
    body = framed[: -len(b"\x04>")]
    stdout_b, _, stderr_b = body.partition(CTRL_D)
    if stderr_b.strip():
        raise RuntimeError(stderr_b.decode("utf-8", errors="replace").rstrip())
    return stdout_b.decode("utf-8", errors="replace")


def list_files(ser: serial.Serial) -> list[str]:
    raw = exec_on_pico(ser, "import os\nprint('\\n'.join(os.listdir()))")
    return [name for name in raw.splitlines() if name]


def read_file(ser: serial.Serial, name: str) -> str:
    # Use repr() so embedded quotes in filenames can't break the snippet.
    snippet = f"print(open({name!r}).read())"
    return exec_on_pico(ser, snippet, timeout_s=10.0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--file", help="dump just this filename; default = all .py")
    args = ap.parse_args()

    with serial.Serial(args.port, args.baud, timeout=0.5) as ser:
        # USB-CDC reset on DTR — let it settle before banging on the port.
        time.sleep(0.3)
        enter_raw_repl(ser)
        try:
            if args.file:
                targets = [args.file]
            else:
                files = list_files(ser)
                print(f"# Files on {args.port}:", file=sys.stderr)
                for f in files:
                    print(f"#   {f}", file=sys.stderr)
                targets = [f for f in files if f.endswith(".py")]

            for name in targets:
                print(f"\n# ===== {name} =====")
                try:
                    sys.stdout.write(read_file(ser, name))
                except RuntimeError as e:
                    print(f"# <read failed: {e}>", file=sys.stderr)
        finally:
            exit_raw_repl(ser)
    return 0


if __name__ == "__main__":
    sys.exit(main())
