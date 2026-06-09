"""Serial wrapper for the Raspberry Pi Pico belt controller.

Protocol (already implemented in Pico firmware):
    - 115200 baud, ``\n``-terminated lines
    - Commands: PING, MOVE <steps>, SPEED <hz>, ENABLE, DISABLE, STATUS,
      SORT <bin_name>, HOME
    - Responses: ``OK`` | ``OK <data>`` | ``ERR <reason>``

Used as a context manager from the main sorter loop::

    with Pico() as pico:
        pico.enable()
        if not pico.ping():
            raise RuntimeError("Pico not responding")
        pico.sort("penny")
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

import serial
from serial import SerialException

log = logging.getLogger("coin_sorter.pico")


class PicoError(RuntimeError):
    """Raised when the Pico returns ``ERR <reason>`` or the link is dead."""


class Pico:
    """Thin serial wrapper around the Pico belt controller.

    Parameters
    ----------
    port:
        Serial device, e.g. ``/dev/ttyACM0``.
    baud:
        Baud rate; must match the firmware (115200).
    timeout_s:
        Per-read timeout in seconds.
    reconnect_delay_s:
        Sleep between reconnect attempts after a disconnect.
    """

    def __init__(
        self,
        port: str = "/dev/ttyACM0",
        baud: int = 115200,
        timeout_s: float = 1.0,
        reconnect_delay_s: float = 1.0,
    ) -> None:
        self.port = port
        self.baud = baud
        self.timeout_s = timeout_s
        self.reconnect_delay_s = reconnect_delay_s
        self._ser: serial.Serial | None = None
        self._lock = threading.Lock()

    # ----- lifecycle -----

    def open(self) -> None:
        """Open the serial port. Blocks briefly while the Pico resets on DTR."""
        if self._ser is not None and self._ser.is_open:
            return
        log.info("Opening serial port %s @ %d", self.port, self.baud)
        self._ser = serial.Serial(self.port, self.baud, timeout=self.timeout_s)
        # Pico USB-CDC typically resets when DTR toggles — give it a moment.
        time.sleep(0.5)
        self._ser.reset_input_buffer()
        self._ser.reset_output_buffer()

    def close(self) -> None:
        """Close the serial port, swallowing errors."""
        if self._ser is not None:
            try:
                self._ser.close()
            except Exception:  # pragma: no cover
                pass
            self._ser = None

    def __enter__(self) -> "Pico":
        self.open()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ----- low-level IO -----

    def _send(self, cmd: str) -> str:
        """Send `cmd` and return the response line (without trailing newline).

        Auto-reconnects once on transient disconnects (e.g. USB unplug/replug).
        Raises :class:`PicoError` if the Pico responds with ``ERR ...``.
        """
        line = (cmd.strip() + "\n").encode("ascii")
        with self._lock:
            for attempt in (1, 2):
                try:
                    if self._ser is None or not self._ser.is_open:
                        self.open()
                    assert self._ser is not None
                    log.debug("-> %s", cmd.strip())
                    self._ser.write(line)
                    self._ser.flush()
                    raw = self._ser.readline()
                    if not raw:
                        raise PicoError(f"Timed out waiting for response to {cmd!r}")
                    response = raw.decode("ascii", errors="replace").strip()
                    log.debug("<- %s", response)
                    if response.startswith("ERR"):
                        raise PicoError(response)
                    return response
                except SerialException as e:
                    log.warning("Serial error on attempt %d: %s", attempt, e)
                    self.close()
                    if attempt == 1:
                        time.sleep(self.reconnect_delay_s)
                        continue
                    raise PicoError(f"Serial link dead: {e}") from e
        raise PicoError("unreachable")  # pragma: no cover

    # ----- protocol -----

    def ping(self) -> bool:
        """Return True if the Pico responds OK to PING."""
        try:
            return self._send("PING").startswith("OK")
        except PicoError as e:
            log.error("PING failed: %s", e)
            return False

    def move(self, steps: int) -> None:
        """Move the belt by `steps` motor steps (sign = direction)."""
        self._send(f"MOVE {int(steps)}")

    def speed(self, hz: int) -> None:
        """Set belt step rate in Hz."""
        self._send(f"SPEED {int(hz)}")

    def run(self, hz: int) -> None:
        """Start continuous (non-blocking) belt motion at `hz` steps/s.

        Sign sets direction. The firmware drives the steps from a hardware
        timer and returns immediately, so the link stays responsive (STATUS /
        STOP work mid-run). Halt with :meth:`stop` (MOVE/SORT also halt it).
        """
        self._send(f"RUN {int(hz)}")

    def stop(self) -> None:
        """Stop continuous belt motion started by :meth:`run`."""
        self._send("STOP")

    def set_leds(self, r: int, g: int, b: int) -> None:
        """Set the whole WS2812 ring to one RGB colour (0-255 each)."""
        self._send(f"LED {int(r)} {int(g)} {int(b)}")

    def ring(self, brightness: int) -> None:
        """Set the ring to neutral white at `brightness` (0-255). 0 = off."""
        self._send(f"RING {int(brightness)}")

    def enable(self) -> None:
        """Enable the motor driver."""
        self._send("ENABLE")

    def disable(self) -> None:
        """Disable the motor driver (free-wheel)."""
        self._send("DISABLE")

    def home(self) -> None:
        """Run the homing routine (future firmware command)."""
        self._send("HOME")

    def sort(self, bin_name: str) -> None:
        """Route the next coin into the named bin via the diverter."""
        if " " in bin_name or not bin_name:
            raise ValueError(f"Invalid bin name: {bin_name!r}")
        self._send(f"SORT {bin_name}")

    def status(self) -> dict[str, str]:
        """Return a dict of key=value pairs parsed from STATUS."""
        resp = self._send("STATUS")
        # Expect: "OK key1=val1 key2=val2 ..."
        body = resp[2:].strip() if resp.startswith("OK") else resp
        out: dict[str, str] = {}
        for tok in body.split():
            if "=" in tok:
                k, v = tok.split("=", 1)
                out[k] = v
        return out
