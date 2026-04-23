"""
Microscope stage controller  (Pico + A4988 + NEMA stepper)

Extends the existing command-based REPL with a "click" command that
rotates the microscope stage by 10 degrees per invocation, using 1/8
microstepping for smooth motion.

Calibration used for the click geometry
---------------------------------------
Empirical test:
    - 30 s run, 2000 us half-pulse delay, full-step mode (1/1)
    - That's 7500 full-steps delivered to the rotor
    - The stage rotated 430 degrees
    =>  full steps per stage degree = 7500 / 430 = 17.4419
    =>  gear ratio (rotor turns per stage turn) ~= 31.4

    In 1/8 microstep mode:
        pulses per 10 degrees of stage = 17.4419 * 8 * 10 = 1395.35
    Tracked as a float with target-position accumulation so that
    rounding never drifts across many clicks.

Serial commands
---------------
    click                 - rotate the stage +10 degrees (one click)
    click N               - perform N consecutive 10-degree clicks
    count                 - print current click count / total stage angle
    reset                 - zero the click counter
    clickdir 0|1          - direction used for clicks (default 0)

    (existing commands still work)
    move <steps> <dir> <delay_us>
    timed <dir> <delay_us> <run_ms> <pause_ms> <iterations>
    dir 0|1
    microstep full|half|quarter|eighth
    sleep | wake
"""

import time
from machine import Pin

# ------------ Pin configuration ------------
STEP_PIN = Pin(19, Pin.OUT)
DIR_PIN  = Pin(18, Pin.OUT)
SLP_PIN  = Pin(20, Pin.OUT)
MS1_PIN  = Pin(16, Pin.OUT)
MS2_PIN  = Pin(17, Pin.OUT)

# ------------ Motor / mechanical constants ------------
NEMA_FULL_STEPS_PER_REV  = 200
FULL_STEPS_PER_DEG_STAGE = 7500.0 / 430.0      # = 17.4419  (from calibration)
DEGREES_PER_CLICK        = 10
CLICK_DIRECTION_DEFAULT  = 0                   # flip to 1 if stage turns wrong way
CLICK_DELAY_US           = 500                 # per half-pulse in 1/8 mode
                                               # -> 1 ms per pulse, click ~1.4 s

# ------------ Runtime state ------------
current_microstep = 1
click_count       = 0
stage_deg_nominal = 0.0
click_direction   = CLICK_DIRECTION_DEFAULT

# Target-position tracking (resets when microstep changes)
_steps_since_ref  = 0
_stage_deg_at_ref = 0.0


# ------------ Driver init: wake + 1/8 microstep ------------
SLP_PIN.value(1)
time.sleep_ms(5)
MS1_PIN.value(1); MS2_PIN.value(1)
current_microstep = 8
DIR_PIN.value(click_direction)


# ============================================================
# Low-level motor helpers
# ============================================================
def pulse_steps(n_pulses, delay_us):
    """Deliver n_pulses STEP pulses at the currently-set direction / microstep."""
    for _ in range(n_pulses):
        STEP_PIN.value(1)
        time.sleep_us(delay_us)
        STEP_PIN.value(0)
        time.sleep_us(delay_us)


def move_stepper(steps, direction, delay_us):
    DIR_PIN.value(direction)
    pulse_steps(steps, delay_us)


def move_timed(direction, delay_us, run_time_ms, pause_time_ms, iterations):
    DIR_PIN.value(direction)
    for cycle in range(iterations):
        print("Cycle {}/{}: Motor ON for {}ms".format(cycle + 1, iterations, run_time_ms))
        start_time = time.ticks_ms()
        steps_completed = 0
        while time.ticks_diff(time.ticks_ms(), start_time) < run_time_ms:
            STEP_PIN.value(1)
            time.sleep_us(delay_us)
            STEP_PIN.value(0)
            time.sleep_us(delay_us)
            steps_completed += 1
        print("  Completed {} steps".format(steps_completed))
        if cycle < iterations - 1:
            print("  Motor OFF for {}ms".format(pause_time_ms))
            time.sleep_ms(pause_time_ms)
    print("Timed movement complete.")


def set_microstep(mode_name):
    """Set A4988 MS1/MS2 and reset the click target-position reference."""
    global current_microstep, _steps_since_ref, _stage_deg_at_ref
    table = {"full": (0, 0, 1), "half": (1, 0, 2),
             "quarter": (0, 1, 4), "eighth": (1, 1, 8)}
    if mode_name not in table:
        print("Invalid microstep mode. Use full | half | quarter | eighth.")
        return
    ms1, ms2, factor = table[mode_name]
    MS1_PIN.value(ms1)
    MS2_PIN.value(ms2)
    current_microstep = factor
    _steps_since_ref  = 0
    _stage_deg_at_ref = stage_deg_nominal
    print("Microstepping set to 1/{}.".format(factor))


# ============================================================
# Click command
# ============================================================
def do_click(n=1):
    """Advance the stage by n clicks of DEGREES_PER_CLICK each."""
    global click_count, stage_deg_nominal, _steps_since_ref

    DIR_PIN.value(click_direction)
    pulses_per_deg = FULL_STEPS_PER_DEG_STAGE * current_microstep

    for _ in range(n):
        click_count       += 1
        stage_deg_nominal += DEGREES_PER_CLICK

        # Target pulses since last reference, so fractional residue never drifts
        target_since_ref  = round((stage_deg_nominal - _stage_deg_at_ref) * pulses_per_deg)
        pulses_this_click = target_since_ref - _steps_since_ref

        pulse_steps(pulses_this_click, CLICK_DELAY_US)
        _steps_since_ref = target_since_ref

        print("Click #{:>4}  |  stage +{}deg  (total {}deg)  |  +{} pulses at 1/{} step"
              .format(click_count, DEGREES_PER_CLICK, stage_deg_nominal,
                      pulses_this_click, current_microstep))


def reset_count():
    global click_count, stage_deg_nominal, _steps_since_ref, _stage_deg_at_ref
    click_count       = 0
    stage_deg_nominal = 0.0
    _steps_since_ref  = 0
    _stage_deg_at_ref = 0.0
    print("Counter reset. click_count = 0, stage_deg = 0.")


def show_count():
    print("Clicks: {}   |   stage angle (nominal): {}deg   |   microstep: 1/{}"
          .format(click_count, stage_deg_nominal, current_microstep))


# ============================================================
# REPL
# ============================================================
print("Stepper motor control ready.")
print("Click commands:  'click' | 'click N' | 'count' | 'reset' | 'clickdir 0|1'")
print("Motion:          'move <steps> <dir> <delay_us>'")
print("                 'timed <dir> <delay_us> <run_ms> <pause_ms> <iters>'")
print("Config:          'dir 0|1' | 'microstep full|half|quarter|eighth'")
print("                 'sleep' | 'wake'")
print("Default: 1/8 microstep, click direction {}, ~{} pulses per {}deg click."
      .format(click_direction,
              round(FULL_STEPS_PER_DEG_STAGE * 8 * DEGREES_PER_CLICK),
              DEGREES_PER_CLICK))

while True:
    try:
        command = input().strip().split()
        if not command:
            continue

        cmd = command[0]

        # ---- New click commands ----
        if cmd == "click":
            n = int(command[1]) if len(command) > 1 else 1
            do_click(n)

        elif cmd == "count":
            show_count()

        elif cmd == "reset":
            reset_count()

        elif cmd == "clickdir":
            click_direction = int(command[1])
            DIR_PIN.value(click_direction)
            print("Click direction set to {}.".format(click_direction))

        # ---- Existing commands ----
        elif cmd == "move":
            steps     = int(command[1])
            direction = int(command[2])
            delay_us  = int(command[3])
            print("Moving {} steps, direction {}, delay {}us...".format(steps, direction, delay_us))
            move_stepper(steps, direction, delay_us)
            print("Move complete.")

        elif cmd == "timed":
            direction     = int(command[1])
            delay_us      = int(command[2])
            run_time_ms   = int(command[3])
            pause_time_ms = int(command[4])
            iterations    = int(command[5])
            print("Starting timed movement: {} cycles of {}ms run, {}ms pause"
                  .format(iterations, run_time_ms, pause_time_ms))
            move_timed(direction, delay_us, run_time_ms, pause_time_ms, iterations)

        elif cmd == "dir":
            direction = int(command[1])
            DIR_PIN.value(direction)
            print("Direction set to {}".format(direction))

        elif cmd == "sleep":
            SLP_PIN.value(0)
            print("Driver put into sleep mode.")

        elif cmd == "wake":
            SLP_PIN.value(1)
            time.sleep_ms(5)
            print("Driver woken up.")

        elif cmd == "microstep":
            set_microstep(command[1])

        else:
            print("Unknown command.")

    except Exception as e:
        print("Error: {}".format(e))
        print("Please check command format.")