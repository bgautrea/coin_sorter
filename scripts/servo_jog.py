#!/usr/bin/env python3
"""Jog the diverter servo over the Pico's raw REPL and record bin positions.

The dish diverter is an MG996R on GP6: a fixed-tilt plate whose one downhill
direction is aimed by rotating it. There is nothing to home against, so the
three bin headings are just three calibrated pulse widths that have to be
found by eye once and written into the firmware.

MG996R clones vary a lot in centre and us/degree, so don't compute these from
an angle - nudge until the spout points at the cup, press the bin key, repeat.

Interrupts whatever the Pico is running (the port is exclusive) and offers a
soft reset on the way out to relaunch main.py.

    python scripts/servo_jog.py
    python scripts/servo_jog.py --pin 6 --min 800 --max 2200

Keys:
    left/right  or  , .     jog -/+ 10 us
    [ ]                     jog -/+ 50 us
    1 2 3                   record current position as keep / common / check
    c                       back to 1500 us
    t                       toggle attach (detach = pulses off, servo limp)
    p                       print the table so far
    q                       quit
"""
from __future__ import annotations

import argparse
import sys
import termios
import time
import tty

import serial

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from read_pico_program import (  # noqa: E402
    CTRL_D,
    enter_raw_repl,
    exec_on_pico,
    exit_raw_repl,
)

SERIAL_BY_ID = "/dev/serial/by-id"

CENTRE_US = 1500
FINE_US = 10
COARSE_US = 50
# A jump larger than this is walked in steps so the dish never slams a rail.
RAMP_THRESHOLD_US = 60
RAMP_STEP_US = 10

BINS = {"1": "keep", "2": "common", "3": "check"}


def find_port() -> str:
    """Locate the Pico by its USB serial number, not by ttyACM number.

    A Pico that resets re-enumerates and can come back as a different ttyACM,
    leaving anything holding the old node talking to a dead handle. The
    by-id symlink follows the board across resets.
    """
    import pathlib

    by_id = pathlib.Path(SERIAL_BY_ID)
    if by_id.is_dir():
        picos = sorted(p for p in by_id.iterdir() if "MicroPython" in p.name)
        if len(picos) == 1:
            return str(picos[0].resolve())
        if len(picos) > 1:
            names = ", ".join(p.name for p in picos)
            raise SystemExit(f"Several MicroPython boards attached; pass --port.\n  {names}")
    fallback = sorted(pathlib.Path("/dev").glob("ttyACM*"))
    if len(fallback) == 1:
        return str(fallback[0])
    raise SystemExit("Could not find the Pico. Is it plugged in? Pass --port explicitly.")


def _setup(ser: serial.Serial, pin: int) -> None:
    exec_on_pico(
        ser,
        "from machine import Pin, PWM\n"
        f"_srv = PWM(Pin({pin}))\n"
        "_srv.freq(50)\n"
        f"_srv.duty_ns({CENTRE_US} * 1000)\n",
    )


def _write_us(ser: serial.Serial, us: int) -> None:
    exec_on_pico(ser, f"_srv.duty_ns({us} * 1000)")


def _detach(ser: serial.Serial, pin: int) -> None:
    exec_on_pico(ser, f"_srv.deinit()\nPin({pin}, Pin.OUT, 0)")


def _attach(ser: serial.Serial, pin: int, us: int) -> None:
    exec_on_pico(
        ser,
        f"_srv = PWM(Pin({pin}))\n_srv.freq(50)\n_srv.duty_ns({us} * 1000)\n",
    )


def _goto(ser: serial.Serial, frm: int, to: int) -> None:
    """Walk to `to` in small steps if it is a long way, else jump."""
    if abs(to - frm) <= RAMP_THRESHOLD_US:
        _write_us(ser, to)
        return
    step = RAMP_STEP_US if to > frm else -RAMP_STEP_US
    for us in range(frm + step, to, step):
        _write_us(ser, us)
        time.sleep(0.015)
    _write_us(ser, to)


def selftest(ser: serial.Serial, pin: int) -> None:
    """Prove the Pico is executing our code and really emitting the waveform.

    Nothing here is taken on trust: the nonce round-trip rules out a stale
    buffer, the LED gives a signal you can see at the bench, and the pulse
    width is recomputed from the PWM slice's own DIV/TOP/CC registers rather
    than from what MicroPython says it asked for.
    """
    nonce = int(time.time()) % 100000
    print(f"1. round-trip     ", end="", flush=True)
    got = exec_on_pico(ser, f"print({nonce} * 7)").strip()
    assert got == str(nonce * 7), f"expected {nonce * 7}, got {got!r}"
    print(f"ok ({nonce} * 7 = {got})")

    print(f"2. board id       ", end="", flush=True)
    out = exec_on_pico(
        ser,
        "import machine, ubinascii\n"
        "print(ubinascii.hexlify(machine.unique_id()).decode(), machine.freq())\n",
    ).split()
    uid, sysclk = out[0], int(out[1])
    print(f"ok (RP2040 {uid}, sysclk {sysclk / 1e6:.0f} MHz)")

    print(f"3. onboard LED    ", end="", flush=True)
    exec_on_pico(
        ser,
        "from machine import Pin\n"
        "import time\n"
        "_led = Pin(25, Pin.OUT)\n"
        "for _ in range(6):\n"
        "    _led.value(1); time.sleep_ms(120)\n"
        "    _led.value(0); time.sleep_ms(120)\n",
        timeout_s=8.0,
    )
    print("blinked 6x - did you see it?")

    _setup(ser, pin)

    print(f"4. GP{pin} function  ", end="", flush=True)
    # IO_BANK0 GPIOx_CTRL: FUNCSEL 4 = PWM
    ctrl = int(
        exec_on_pico(
            ser, f"from machine import mem32\nprint(mem32[0x40014000 + {pin} * 8 + 4])"
        ).strip()
    )
    funcsel = ctrl & 0x1F
    print(f"{'ok' if funcsel == 4 else 'WRONG'} (FUNCSEL={funcsel}, 4=PWM)")

    print(f"5. slice registers ", end="", flush=True)
    slice_no = (pin // 2) % 8
    base = 0x40050000 + slice_no * 0x14
    regs = exec_on_pico(
        ser,
        "from machine import mem32\n"
        f"print(mem32[{base}], mem32[{base + 4}], mem32[{base + 0xC}], mem32[{base + 0x10}])\n",
    ).split()
    csr, div, cc, top = (int(v) for v in regs)
    divisor = ((div >> 4) & 0xFF) + ((div & 0xF) / 16.0)
    count_hz = sysclk / divisor
    period_us = (top + 1) / count_hz * 1e6
    # GP6 is channel A of its slice -> low 16 bits of CC.
    pulse_us = (cc & 0xFFFF) / count_hz * 1e6
    enabled = bool(csr & 1)
    print(f"slice {slice_no}, DIV {divisor:.2f}, TOP {top}, CC {cc & 0xFFFF}")
    print(f"   -> enabled={enabled}  period={period_us:.0f} us "
          f"({1e6 / period_us:.1f} Hz)  pulse={pulse_us:.0f} us")

    ok = enabled and funcsel == 4 and 19000 < period_us < 21000
    print("\n" + ("PASS - GP6 really is emitting that waveform."
                  if ok else "FAIL - the pin is not outputting what we asked for."))
    if ok:
        print("If the servo still jitters, the fault is downstream of the Pico:\n"
              "power, ground, the lead, or the servo itself.")


def _read_key() -> str:
    """One keypress, arrow keys collapsed to 'LEFT'/'RIGHT'."""
    ch = sys.stdin.read(1)
    if ch != "\x1b":
        return ch
    if sys.stdin.read(1) != "[":
        return ch
    return {"D": "LEFT", "C": "RIGHT"}.get(sys.stdin.read(1), "")


def _status(us: int, attached: bool, table: dict[str, int]) -> None:
    recorded = " ".join(f"{k}={v}" for k, v in table.items()) or "-"
    state = "on " if attached else "OFF"
    print(f"\r  {us:4d} us   pulses {state}   recorded: {recorded}    ", end="")
    sys.stdout.flush()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", default=None, help="default: auto-detect the Pico by USB id")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--pin", type=int, default=6, help="Pico GPIO for the servo signal")
    ap.add_argument("--min", type=int, default=900, help="lower pulse-width clamp (us)")
    ap.add_argument("--max", type=int, default=2100, help="upper pulse-width clamp (us)")
    ap.add_argument(
        "--selftest",
        action="store_true",
        help="prove the Pico is executing our code and emitting the waveform, then exit",
    )
    args = ap.parse_args()
    args.port = args.port or find_port()

    if args.selftest:
        with serial.Serial(args.port, args.baud, timeout=0.5) as ser:
            time.sleep(0.3)
            enter_raw_repl(ser)
            try:
                selftest(ser, args.pin)
            finally:
                exit_raw_repl(ser)
        return 0

    if not args.min < CENTRE_US < args.max:
        print(f"--min/--max must straddle {CENTRE_US} us", file=sys.stderr)
        return 2

    us = CENTRE_US
    attached = True
    table: dict[str, int] = {}

    print(__doc__.split("Keys:")[1].rstrip())
    print(f"\nGP{args.pin} on {args.port}, clamped to {args.min}-{args.max} us.")
    print("Keep hands clear - the dish snaps to centre on connect.\n")

    with serial.Serial(args.port, args.baud, timeout=0.5) as ser:
        time.sleep(0.3)  # USB-CDC resets on DTR
        enter_raw_repl(ser)
        _setup(ser, args.pin)
        fd = sys.stdin.fileno()
        saved = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            _status(us, attached, table)
            while True:
                key = _read_key()
                if key in ("q", "\x03"):
                    break
                delta = {
                    "LEFT": -FINE_US, ",": -FINE_US,
                    "RIGHT": FINE_US, ".": FINE_US,
                    "[": -COARSE_US, "]": COARSE_US,
                }.get(key, 0)
                if delta:
                    target = max(args.min, min(args.max, us + delta))
                    if attached and target != us:
                        _goto(ser, us, target)
                    us = target
                elif key == "c":
                    if attached:
                        _goto(ser, us, CENTRE_US)
                    us = CENTRE_US
                elif key in BINS:
                    table[BINS[key]] = us
                elif key == "t":
                    attached = not attached
                    if attached:
                        _attach(ser, args.pin, us)
                    else:
                        _detach(ser, args.pin)
                elif key == "p":
                    print("\r" + " " * 70 + "\r", end="")
                    print("  " + repr(table) + "\r\n", end="")
                _status(us, attached, table)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, saved)
            print()
            try:
                _detach(ser, args.pin)
            except Exception:
                pass
            exit_raw_repl(ser)

        missing = [b for b in ("keep", "common", "check") if b not in table]
        if missing:
            print(f"Not recorded: {', '.join(missing)}")
        if table:
            print("\nPaste into firmware/main.py:\n")
            print("BIN_POSITIONS_US = {")
            for name in ("keep", "common", "check"):
                if name in table:
                    print(f'    "{name:<7}": {table[name]},')
            print("}")

        if input("\nSoft reset the Pico to relaunch main.py? [y/N] ").lower() == "y":
            ser.write(CTRL_D)
            time.sleep(0.2)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
