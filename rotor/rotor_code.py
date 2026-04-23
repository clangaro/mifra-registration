"""
Microscope stage control — Pico + A4988 + NEMA stepper.

Extends the original control script with a `click` command that rotates
the stage by a fixed angle (default 5°) and maintains a running counter.

Gear train: rotor pinion  ->  2 cm idler  ->  9.5 cm stage gear.
Measured gear ratio from calibration: 37.5 rotor rev / (510°/360°) ≈ 26.47.

Wiring (unchanged):
    STEP -> GP19 (orange)
    DIR  -> GP18 (yellow)
    SLP  -> GP20 (light purple)
    MS1  -> GP16 (white)
    MS2  -> GP17 (green)
"""

import time
from machine import Pin

# ===================== Pin configuration =====================
STEP_PIN_PICO = Pin(19, Pin.OUT)
DIR_PIN_PICO  = Pin(18, Pin.OUT)
SLP_PIN_PICO  = Pin(20, Pin.OUT)
MS1_PIN_PICO  = Pin(16, Pin.OUT)
MS2_PIN_PICO  = Pin(17, Pin.OUT)

# ===================== Mechanical / click configuration =====================
GEAR_RATIO               = 8.82     # rotor turns per 1 stage turn
                                    # empirical: 24 clicks of 5° produced one
                                    # full stage revolution, so the previous
                                    # 26.47 was 3× too high -> 26.47 / 3
DEG_PER_CLICK            = 10       # stage degrees advanced per click
FULL_STEPS_PER_ROTOR_REV = 200      # standard NEMA 17 (1.8°/step)
CLICK_DIRECTION          = 0        # 0 or 1; flip if stage rotates wrong way
CLICK_DELAY_US           = 1000     # step pulse half-period during a click

# ===================== Driver init =====================
SLP_PIN_PICO.value(1)               # wake the A4988
MS1_PIN_PICO.value(0)               # start in full step, matching original
MS2_PIN_PICO.value(0)
time.sleep_ms(2)                    # A4988 needs ~1 ms settle after wake

# ===================== Runtime state =====================
current_microstep  = 1              # 1, 2, 4, or 8
click_count        = 0              # number of 5° clicks advanced
click_pulses_total = 0              # pulses sent on the click axis since last reset
                                    # (units: pulses at current microstep setting)


# ===================== Core motor functions =====================
def move_stepper(steps, direction, delay_us):
    """Send `steps` step pulses in the given direction."""
    if steps <= 0:
        return
    DIR_PIN_PICO.value(direction)
    for _ in range(steps):
        STEP_PIN_PICO.value(1)
        time.sleep_us(delay_us)
        STEP_PIN_PICO.value(0)
        time.sleep_us(delay_us)


def move_timed(direction, delay_us, run_time_ms, pause_time_ms, iterations):
    """Kept from the original: run continuously for a duration, with pauses."""
    DIR_PIN_PICO.value(direction)
    for cycle in range(iterations):
        print("Cycle {}/{}: Motor ON for {} ms".format(
            cycle + 1, iterations, run_time_ms))
        t0 = time.ticks_ms()
        steps = 0
        while time.ticks_diff(time.ticks_ms(), t0) < run_time_ms:
            STEP_PIN_PICO.value(1); time.sleep_us(delay_us)
            STEP_PIN_PICO.value(0); time.sleep_us(delay_us)
            steps += 1
        print("  Completed {} steps".format(steps))
        if cycle < iterations - 1:
            print("  Motor OFF for {} ms".format(pause_time_ms))
            time.sleep_ms(pause_time_ms)
    print("Timed movement complete.")


# ===================== Microstep handling =====================
MICROSTEP_MODES = {
    "full":    (1, 0, 0),
    "half":    (2, 1, 0),
    "quarter": (4, 0, 1),
    "eighth":  (8, 1, 1),
}


def set_microstep(name):
    """Switch microstep mode and reset the click-pulse accumulator."""
    global current_microstep, click_pulses_total
    if name not in MICROSTEP_MODES:
        print("Use: full, half, quarter, eighth")
        return
    factor, ms1, ms2 = MICROSTEP_MODES[name]
    MS1_PIN_PICO.value(ms1)
    MS2_PIN_PICO.value(ms2)
    # Pulses taken before the switch are in a different unit, so rescale
    # them into the new microstep unit to keep the running position target
    # consistent.
    click_pulses_total = round(click_pulses_total * (factor / current_microstep))
    current_microstep = factor
    print("Microstepping set to 1/{}  ({:.2f} pulses per 5° click)".format(
        factor, pulses_per_click()))


# ===================== Click logic =====================
def pulses_per_click():
    """Float pulses required per click at the current microstep setting."""
    return (FULL_STEPS_PER_ROTOR_REV * current_microstep
            * GEAR_RATIO * DEG_PER_CLICK / 360.0)


def click_stage(n=1):
    """
    Advance the stage by n clicks (each = DEG_PER_CLICK). Uses a target-position
    scheme so the fractional pulses-per-click never drift.
    """
    global click_count, click_pulses_total
    ppc = pulses_per_click()
    for _ in range(n):
        click_count += 1
        target_pulses = round(click_count * ppc)
        to_send       = target_pulses - click_pulses_total
        move_stepper(to_send, CLICK_DIRECTION, CLICK_DELAY_US)
        click_pulses_total = target_pulses
        total_deg = click_count * DEG_PER_CLICK
        print("Click #{:<4}  stage +{}°  (total {}°)  |  {} pulses @ 1/{}".format(
            click_count, DEG_PER_CLICK, total_deg, to_send, current_microstep))


def reset_clicks():
    global click_count, click_pulses_total
    click_count        = 0
    click_pulses_total = 0
    print("Click counter reset.")


# ===================== Help =====================
def print_help():
    print("Stepper + stage control ready.")
    print("  gear ratio = {}, deg/click = {}, startup microstep = 1/{}".format(
        GEAR_RATIO, DEG_PER_CLICK, current_microstep))
    print("  pulses per click now: {:.2f}".format(pulses_per_click()))
    print("Commands:")
    print("  click [n]                              -- advance stage by n×5° (default 1)")
    print("  reset                                  -- reset click counter")
    print("  count                                  -- show clicks & total stage angle")
    print("  move <steps> <dir> <delay_us>")
    print("  timed <dir> <delay_us> <run_ms> <pause_ms> <iter>")
    print("  dir 0 | dir 1")
    print("  microstep full | half | quarter | eighth")
    print("  sleep | wake")
    print("  help")


print_help()


# ===================== Command loop =====================
while True:
    try:
        raw = input().strip()
        if not raw:
            continue
        cmd = raw.split()
        op  = cmd[0]

        if op == "click":
            n = int(cmd[1]) if len(cmd) > 1 else 1
            click_stage(n)

        elif op == "reset":
            reset_clicks()

        elif op == "count":
            print("Clicks: {}   Stage angle: {}°   Pulses sent: {} @ 1/{}".format(
                click_count, click_count * DEG_PER_CLICK,
                click_pulses_total, current_microstep))

        elif op == "move":
            steps     = int(cmd[1])
            direction = int(cmd[2])
            delay_us  = int(cmd[3])
            print("Moving {} steps, dir {}, delay {} us...".format(
                steps, direction, delay_us))
            move_stepper(steps, direction, delay_us)
            print("Move complete.")

        elif op == "timed":
            direction     = int(cmd[1])
            delay_us      = int(cmd[2])
            run_time_ms   = int(cmd[3])
            pause_time_ms = int(cmd[4])
            iterations    = int(cmd[5])
            print("Timed: {} cycles of {} ms run, {} ms pause".format(
                iterations, run_time_ms, pause_time_ms))
            move_timed(direction, delay_us, run_time_ms, pause_time_ms, iterations)

        elif op == "dir":
            DIR_PIN_PICO.value(int(cmd[1]))
            print("Direction set to {}".format(cmd[1]))

        elif op == "sleep":
            SLP_PIN_PICO.value(0)
            print("Driver put into sleep mode.")

        elif op == "wake":
            SLP_PIN_PICO.value(1)
            time.sleep_ms(2)
            print("Driver woken up.")

        elif op == "microstep":
            set_microstep(cmd[1])

        elif op == "help":
            print_help()

        else:
            print("Unknown command: {}".format(op))

    except Exception as e:
        print("Error: {}".format(e))
        print("Type 'help' for command list.")