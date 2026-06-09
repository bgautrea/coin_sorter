# coin_sorter.py - RP2040 firmware
#
# Currently active hardware:
#   - Belt drive: NEMA17 via TB6600-style driver
#       PUL/DIR/ENA on GP2/3/4, driver + pins to Pico 3V3
#       Inverted ENA logic (this clone disables on current flow)
#
# Add later by flipping the flags below:
#   DIVERTER_ENABLED    - 28BYJ-48 + ULN2003 on GP6/7/8/9
#   HOME_SWITCH_ENABLED - microswitch on GP10
#   COIN_SENSOR_ENABLED - IR break-beam on GP11

import sys, select, time
from machine import Pin, mem32, Timer
from neopixel import NeoPixel

# ============================================================
# Feature flags - flip to True as you wire each subsystem
# ============================================================
DIVERTER_ENABLED    = False
HOME_SWITCH_ENABLED = False
COIN_SENSOR_ENABLED = False

# ============================================================
# Belt stepper (TB6600 / NEMA17)
# ============================================================
PUL = Pin(2, Pin.OUT, value=1)
DIR = Pin(3, Pin.OUT, value=0)
ENA = Pin(4, Pin.OUT, value=0)

# Bump GP2/3/4 to 12mA drive strength via PADS_BANK0
PADS_BASE = 0x4001C000
for _gpio in (2, 3, 4):
    _reg = PADS_BASE + 0x04 + _gpio * 4
    mem32[_reg] = (mem32[_reg] & ~0x30) | 0x30

def belt_enable(on):
    """Inverted ENA logic for this driver clone."""
    ENA.value(1 if on else 0)

LED = Pin(25, Pin.OUT, value=0)

# Software timer used for non-blocking continuous belt motion (RUN/STOP).
belt_timer = Timer()

# ============================================================
# Ring light (WS2812 on GP16) - camera illumination
# ============================================================
RING_PIN   = 16
RING_COUNT = 10
ring = NeoPixel(Pin(RING_PIN), RING_COUNT)  # GRB order, 800kHz

def ring_fill(r, g, b):
    """Set the whole ring to one colour (0-255 each) and latch it."""
    c = (r & 255, g & 255, b & 255)
    for i in range(RING_COUNT):
        ring[i] = c
    ring.write()

ring_fill(0, 0, 0)  # start dark

# ============================================================
# Diverter stepper (28BYJ-48 / ULN2003) - optional
# ============================================================
HALF_STEP_SEQ = [
    (1,0,0,0), (1,1,0,0), (0,1,0,0), (0,1,1,0),
    (0,0,1,0), (0,0,1,1), (0,0,0,1), (1,0,0,1),
]
DIV_STEPS_PER_REV = 4096
DIV_STEP_DELAY_MS = 2

if DIVERTER_ENABLED:
    DIV_PINS = [Pin(p, Pin.OUT, value=0) for p in (6, 7, 8, 9)]
else:
    DIV_PINS = None

BIN_POSITIONS = {
    "home":    0,
    "penny":   400,
    "nickel":  800,
    "dime":    1200,
    "quarter": 1600,
    "reject":  2000,
}
BELT_STEPS_PER_COIN = 400

# ============================================================
# Optional sensors and outputs
# ============================================================
HOME_SW     = Pin(10, Pin.IN, Pin.PULL_UP) if HOME_SWITCH_ENABLED else None
COIN_SENSOR = Pin(11, Pin.IN, Pin.PULL_UP) if COIN_SENSOR_ENABLED else None
ACT         = Pin(15, Pin.OUT, value=0)   # legacy actuator output

# ============================================================
# State
# ============================================================
state = {
    "speed_hz": 1500,
    "busy": False,
    "belt_position": 0,
    "div_position": 0,
    "div_step_idx": 0,
    "homed": not HOME_SWITCH_ENABLED,   # auto-homed when no switch
    "running": False,                   # True while the belt free-runs (RUN)
}

# ============================================================
# Belt motion
# ============================================================
def belt_move(steps):
    """Blocking belt move. Negative steps = reverse."""
    belt_stop()  # a finite move and a free-run must not fight over PUL
    direction = 1 if steps >= 0 else 0
    DIR.value(direction)
    time.sleep_us(5)
    belt_enable(True)
    half_us = max(50, 500_000 // state["speed_hz"])
    state["busy"] = True
    LED.on()
    for _ in range(abs(steps)):
        PUL.value(0); time.sleep_us(half_us)
        PUL.value(1); time.sleep_us(half_us)
        state["belt_position"] += 1 if direction else -1
    state["busy"] = False
    LED.off()

def _belt_tick(t):
    """Timer ISR: toggle PUL to emit a square wave. Allocation-free.

    Step rate is half the timer frequency (one rising edge per full cycle).
    belt_position is intentionally not tracked while free-running to keep the
    callback safe.
    """
    PUL.value(0 if PUL.value() else 1)

def belt_run(hz):
    """Start non-blocking continuous belt motion. hz>0 forward, hz<0 reverse.

    Returns the signed step rate actually applied. The hardware timer drives
    PUL in the background, so the command loop stays responsive (STATUS/STOP
    work mid-run). STOP, MOVE or SORT halt it.
    """
    belt_stop()
    hz = int(hz)
    if hz == 0:
        return 0
    DIR.value(1 if hz > 0 else 0)
    state["speed_hz"] = abs(hz)
    belt_enable(True)
    state["running"] = True
    state["busy"] = True
    LED.on()
    belt_timer.init(freq=abs(hz) * 2, mode=Timer.PERIODIC, callback=_belt_tick)
    return hz

def belt_stop():
    """Halt continuous belt motion (no-op if not running). Leaves PUL idle."""
    belt_timer.deinit()
    PUL.value(1)
    if state["running"]:
        state["running"] = False
        state["busy"] = False
        LED.off()

# ============================================================
# Diverter motion (no-ops when disabled)
# ============================================================
def div_step_once(direction):
    if not DIVERTER_ENABLED: return
    state["div_step_idx"] = (state["div_step_idx"] + (1 if direction else -1)) % 8
    seq = HALF_STEP_SEQ[state["div_step_idx"]]
    for pin, val in zip(DIV_PINS, seq):
        pin.value(val)
    state["div_position"] += 1 if direction else -1

def div_move(steps, delay_ms=None):
    if not DIVERTER_ENABLED: return
    if delay_ms is None:
        delay_ms = DIV_STEP_DELAY_MS
    direction = 1 if steps >= 0 else 0
    state["busy"] = True
    for _ in range(abs(steps)):
        div_step_once(direction)
        time.sleep_ms(delay_ms)
    state["busy"] = False

def div_move_to(target):
    if not DIVERTER_ENABLED: return
    div_move(target - state["div_position"])

def div_release():
    if not DIVERTER_ENABLED: return
    for pin in DIV_PINS:
        pin.value(0)

def div_home(timeout_steps=4500):
    """Home the diverter. Returns True on success."""
    if not DIVERTER_ENABLED:
        state["homed"] = True
        return True
    if not HOME_SWITCH_ENABLED:
        state["div_position"] = 0
        state["homed"] = True
        return True
    if HOME_SW.value() == 0:
        div_move(300)
        time.sleep_ms(100)
    for _ in range(timeout_steps):
        if HOME_SW.value() == 0:
            state["div_position"] = 0
            state["homed"] = True
            div_release()
            return True
        div_step_once(0)
        time.sleep_ms(DIV_STEP_DELAY_MS)
    div_release()
    return False

# ============================================================
# Coin sensor
# ============================================================
def coin_present():
    if not COIN_SENSOR_ENABLED:
        return False
    return COIN_SENSOR.value() == 0

def wait_for_coin(timeout_ms=10000):
    if not COIN_SENSOR_ENABLED:
        return False
    deadline = time.ticks_add(time.ticks_ms(), timeout_ms)
    while time.ticks_diff(deadline, time.ticks_ms()) > 0:
        if coin_present():
            return True
        time.sleep_ms(5)
    return False

# ============================================================
# Legacy actuator
# ============================================================
def fire(ms):
    ACT.value(1); time.sleep_ms(ms); ACT.value(0)

# ============================================================
# High-level sort
# ============================================================
def sort_coin(bin_name):
    if bin_name not in BIN_POSITIONS:
        return f"ERR unknown bin: {bin_name}"
    if not state["homed"]:
        return "ERR not homed - send HOME first"
    if DIVERTER_ENABLED:
        div_move_to(BIN_POSITIONS[bin_name])
        time.sleep_ms(150)
    belt_move(BELT_STEPS_PER_COIN)
    time.sleep_ms(200)
    if DIVERTER_ENABLED:
        div_release()
    return f"OK sorted {bin_name}"

# ============================================================
# Command handler
# ============================================================
def handle(line):
    parts = line.strip().split()
    if not parts:
        return
    cmd = parts[0].upper()
    try:
        if cmd == "PING":
            print("OK PONG")
        elif cmd == "STATUS":
            features = "belt"
            if DIVERTER_ENABLED:    features += ",diverter"
            if HOME_SWITCH_ENABLED: features += ",home_sw"
            if COIN_SENSOR_ENABLED: features += ",coin_sensor"
            print(f"STATUS busy={state['busy']} running={state['running']} belt={state['belt_position']} div={state['div_position']} homed={state['homed']} speed={state['speed_hz']} coin={coin_present()} features={features}")
        elif cmd == "BINS":
            print("OK " + ",".join(f"{k}={v}" for k, v in BIN_POSITIONS.items()))
        # Belt
        elif cmd == "MOVE":
            belt_move(int(parts[1]))
            print(f"OK {state['belt_position']}")
        elif cmd == "RUN":
            hz = int(parts[1]) if len(parts) > 1 else state["speed_hz"]
            print(f"OK running {belt_run(hz)}")
        elif cmd == "STOP":
            belt_stop(); print("OK stopped")
        elif cmd == "SPEED":
            state["speed_hz"] = int(parts[1])
            print("OK")
        elif cmd == "ENABLE":
            belt_enable(True); print("OK")
        elif cmd == "DISABLE":
            belt_enable(False); print("OK")
        # Ring light
        elif cmd == "LED":
            r, g, b = int(parts[1]), int(parts[2]), int(parts[3])
            ring_fill(r, g, b)
            print(f"OK led {r & 255} {g & 255} {b & 255}")
        elif cmd == "RING":
            v = int(parts[1]) & 255
            ring_fill(v, v, v)
            print(f"OK ring {v}")
        # Diverter
        elif cmd == "DIV":
            div_move(int(parts[1])); div_release()
            print(f"OK {state['div_position']}")
        elif cmd == "DIVTO":
            div_move_to(int(parts[1])); div_release()
            print(f"OK {state['div_position']}")
        elif cmd == "HOME":
            print("OK homed" if div_home() else "ERR home timeout")
        elif cmd == "SORT":
            print(sort_coin(parts[1].lower()))
        # Sensors / misc
        elif cmd == "WAIT":
            ms = int(parts[1]) if len(parts) > 1 else 10000
            print("OK seen" if wait_for_coin(ms) else "ERR no coin")
        elif cmd == "ACT":
            fire(int(parts[1])); print("OK")
        else:
            print(f"ERR unknown: {cmd}")
    except Exception as e:
        print(f"ERR {e}")

# ============================================================
# Main loop
# ============================================================
poller = select.poll()
poller.register(sys.stdin, select.POLLIN)
print("READY")
buf = ""
while True:
    if poller.poll(10):
        ch = sys.stdin.read(1)
        if ch == "\n":
            handle(buf)
            buf = ""
        else:
            buf += ch
