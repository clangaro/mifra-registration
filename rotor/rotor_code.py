#!/usr/bin/env python3
"""
Microscope stage rotator for Raspberry Pi.

Each button press rotates the microscope STAGE by 5°, accounting for the
gear reduction between the stepper rotor and the stage.

Gear train:
    rotor shaft  -->  small gear (r = 2 cm)  -->  stage gear (r = 9.5 cm)

The small gear is rigidly attached to the rotor shaft and meshes directly
with the stage's ring gear, giving a reduction ratio of 9.5 / 2 = 4.75.
So the rotor must turn 4.75x the angle we want at the stage.

Uses a target-position strategy so rounding never accumulates: we compute
the cumulative rotor step target after N clicks and only command the
*difference* from the steps already taken.

Hardware (default wiring, BCM numbering):
    28BYJ-48 stepper + ULN2003 driver board
        IN1 -> GPIO 17
        IN2 -> GPIO 18
        IN3 -> GPIO 27
        IN4 -> GPIO 22
        VCC -> 5V, GND -> GND
    Pushbutton between GPIO 23 and GND (internal pull-up used)

Run:
    python3 stage_rotator.py
Ctrl+C to quit.
"""

import time
import RPi.GPIO as GPIO

# ---------- Pin configuration (BCM numbering) ----------
COIL_PINS  = [17, 18, 27, 22]       # IN1, IN2, IN3, IN4
BUTTON_PIN = 23

# ---------- Mechanical configuration ----------
STAGE_RADIUS_CM     = 9.5
ROTOR_GEAR_RADIUS_CM = 2.0
GEAR_RATIO          = STAGE_RADIUS_CM / ROTOR_GEAR_RADIUS_CM   # = 4.75
DEGREES_PER_CLICK   = 5             # at the stage

# ---------- Stepper configuration ----------
STEPS_PER_REV = 4096                 # 28BYJ-48 half-step mode (with 1:64 gearing)
STEP_DELAY    = 0.002                # seconds between half-steps
DIRECTION     = 1                    # flip to -1 if stage turns the wrong way

# Rotor angle per stage click, in degrees -> steps (fractional, kept as float)
ROTOR_DEG_PER_CLICK   = DEGREES_PER_CLICK * GEAR_RATIO           # 23.75°
STEPS_PER_CLICK_FLOAT = ROTOR_DEG_PER_CLICK / 360 * STEPS_PER_REV # ~270.22

# Half-step drive sequence
HALF_STEP_SEQ = [
    [1, 0, 0, 0],
    [1, 1, 0, 0],
    [0, 1, 0, 0],
    [0, 1, 1, 0],
    [0, 0, 1, 0],
    [0, 0, 1, 1],
    [0, 0, 0, 1],
    [1, 0, 0, 1],
]

# ---------- GPIO setup ----------
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
for pin in COIL_PINS:
    GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

# ---------- State ----------
click_count       = 0      # stage clicks so far
steps_taken_total = 0      # cumulative rotor half-steps actually commanded
phase_index       = 0      # preserves coil phase across calls


def step_motor(num_steps: int, direction: int = 1) -> None:
    """Advance the stepper by `num_steps` half-steps in the given direction."""
    global phase_index
    if num_steps == 0:
        return
    seq_len = len(HALF_STEP_SEQ)
    for _ in range(abs(num_steps)):
        phase_index = (phase_index + direction) % seq_len
        for pin, value in zip(COIL_PINS, HALF_STEP_SEQ[phase_index]):
            GPIO.output(pin, value)
        time.sleep(STEP_DELAY)
    # De-energise coils when idle
    for pin in COIL_PINS:
        GPIO.output(pin, GPIO.LOW)


def on_click() -> None:
    """Advance the stage by one click (5°) without cumulative rounding error."""
    global click_count, steps_taken_total

    click_count += 1

    # Target rotor position, in steps, after this click
    target_steps_total = round(click_count * STEPS_PER_CLICK_FLOAT)
    steps_this_click   = target_steps_total - steps_taken_total

    stage_angle_total = click_count * DEGREES_PER_CLICK
    rotor_angle_total = click_count * ROTOR_DEG_PER_CLICK

    print(f"Click #{click_count:>4}  |  stage +{DEGREES_PER_CLICK}° "
          f"(total {stage_angle_total}°)  |  rotor +{steps_this_click} steps "
          f"(total {target_steps_total}, ~{rotor_angle_total:.2f}°)")

    step_motor(steps_this_click, DIRECTION)
    steps_taken_total = target_steps_total


def main() -> None:
    print("Microscope stage rotator ready.")
    print(f"Gear ratio:        {GEAR_RATIO:.3f}  (stage {STAGE_RADIUS_CM} cm / "
          f"rotor gear {ROTOR_GEAR_RADIUS_CM} cm)")
    print(f"Per click:         stage {DEGREES_PER_CLICK}°  -->  rotor "
          f"{ROTOR_DEG_PER_CLICK}°  ({STEPS_PER_CLICK_FLOAT:.3f} half-steps)")
    print("Press the button to rotate. Ctrl+C to quit.\n")

    last_state = GPIO.input(BUTTON_PIN)
    try:
        while True:
            state = GPIO.input(BUTTON_PIN)
            # Button to GND with pull-up: pressed == LOW. Trigger on falling edge.
            if last_state == GPIO.HIGH and state == GPIO.LOW:
                on_click()
                time.sleep(0.20)          # debounce
            last_state = state
            time.sleep(0.01)
    except KeyboardInterrupt:
        total_stage_deg = click_count * DEGREES_PER_CLICK
        print(f"\nStopped. {click_count} clicks, stage rotated {total_stage_deg}° "
              f"total (rotor {steps_taken_total} steps).")
    finally:
        GPIO.cleanup()


if __name__ == "__main__":
    main()