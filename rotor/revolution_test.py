
import time
from machine import Pin
 
# ---------- Pin configuration (matches the main control code) ----------
STEP_PIN = Pin(19, Pin.OUT)
DIR_PIN  = Pin(18, Pin.OUT)
SLP_PIN  = Pin(20, Pin.OUT)
MS1_PIN  = Pin(16, Pin.OUT)
MS2_PIN  = Pin(17, Pin.OUT)
 
# ---------- Stepper constants ----------
FULL_STEPS_PER_REV = 200          # standard NEMA 17 (1.8° per step)
DIRECTION          = 0            # 0 or 1; flip if the stage turns the wrong way
 
# ---------- Helpers ----------
def ask(prompt, default, cast=float):
    """Prompt with a default shown in brackets; Enter keeps the default."""
    raw = input(f"{prompt} [{default}]: ").strip()
    if raw == "":
        return default
    try:
        return cast(raw)
    except ValueError:
        print(f"  Couldn't parse '{raw}', using default {default}.")
        return default
 
 
def set_full_step():
    MS1_PIN.value(0)
    MS2_PIN.value(0)
 
 
def spin_for_duration(run_time_ms, delay_us, direction):
    """Spin rotor for run_time_ms; return (steps, elapsed_ms)."""
    DIR_PIN.value(direction)
    t0 = time.ticks_ms()
    last_status = t0
    steps = 0
 
    while time.ticks_diff(time.ticks_ms(), t0) < run_time_ms:
        STEP_PIN.value(1)
        time.sleep_us(delay_us)
        STEP_PIN.value(0)
        time.sleep_us(delay_us)
        steps += 1
 
        # Print live status every second
        now = time.ticks_ms()
        if time.ticks_diff(now, last_status) >= 1000:
            secs = time.ticks_diff(now, t0) / 1000
            rotor_revs = steps / FULL_STEPS_PER_REV
            print("  t = {:5.1f} s | {:>6} steps | {:5.2f} rotor rev"
                  .format(secs, steps, rotor_revs))
            last_status = now
 
    elapsed_ms = time.ticks_diff(time.ticks_ms(), t0)
    return steps, elapsed_ms
 
 
def main():
    print("=" * 60)
    print("STAGE REVOLUTION CALIBRATION  (Pico + A4988)")
    print("=" * 60)
 
    # Wake driver, full-step mode
    SLP_PIN.value(1)
    set_full_step()
    time.sleep_ms(5)   # A4988 needs ~1 ms after wake before stepping
 
    # --- Parameters ---
    run_s      = ask("Run duration (seconds)", 30, float)
    delay_us   = ask("Step delay (us per half-pulse; larger = slower)", 2000, int)
    direction  = ask("Direction (0 or 1)", DIRECTION, int)
 
    # Rough preview: at delay_us half-pulse, full period is 2*delay_us, so
    # full-step rate = 1e6 / (2*delay_us) steps/sec.
    step_rate = 1e6 / (2 * delay_us)
    rotor_rps = step_rate / FULL_STEPS_PER_REV
    print("\nEstimated rotor speed: {:.2f} rev/s ({:.0f} RPM)."
          .format(rotor_rps, rotor_rps * 60))
    print("Over {:g} s that's ~{:.1f} rotor revolutions.\n"
          .format(run_s, rotor_rps * run_s))
 
    print("1. Align your reference mark on the stage.")
    print("2. Press Enter to start. Count full STAGE revolutions as it spins.")
    input("   Ready? ")
 
    print("\nSpinning...\n")
    steps, elapsed_ms = spin_for_duration(int(run_s * 1000), delay_us, direction)
 
    # De-energise-ish: driver still awake but no pulses. Put to sleep for safety.
    SLP_PIN.value(0)
 
    elapsed_s  = elapsed_ms / 1000
    rotor_revs = steps / FULL_STEPS_PER_REV
    print("\nDone. {:.2f} s elapsed, {} steps ({:.3f} rotor revolutions).\n"
          .format(elapsed_s, steps, rotor_revs))
 
    stage_revs = ask("How many full STAGE revolutions did you count?", 1, float)
    if stage_revs <= 0:
        print("Can't compute with zero revolutions. Aborting.")
        return
 
    # --- Results ---
    time_per_stage_rev  = elapsed_s / stage_revs
    gear_ratio          = rotor_revs / stage_revs          # rotor turns per 1 stage turn
    steps_per_5deg_full = FULL_STEPS_PER_REV * gear_ratio * 5.0 / 360.0
 
    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print("  Elapsed time               : {:.2f} s".format(elapsed_s))
    print("  Stage revolutions observed : {:g}".format(stage_revs))
    print("  Time per stage revolution  : {:.2f} s".format(time_per_stage_rev))
    print("  Rotor turns per stage turn : {:.3f}   <-- GEAR RATIO"
          .format(gear_ratio))
    print()
    print("  Suggested steps per 5 deg of stage motion:")
    print("    full step    (1/1): {:6.2f}  -->  use {:d}"
          .format(steps_per_5deg_full,       round(steps_per_5deg_full)))
    print("    half step    (1/2): {:6.2f}  -->  use {:d}"
          .format(steps_per_5deg_full * 2,   round(steps_per_5deg_full * 2)))
    print("    quarter step (1/4): {:6.2f}  -->  use {:d}"
          .format(steps_per_5deg_full * 4,   round(steps_per_5deg_full * 4)))
    print("    eighth step  (1/8): {:6.2f}  -->  use {:d}"
          .format(steps_per_5deg_full * 8,   round(steps_per_5deg_full * 8)))
    print("=" * 60)
    print("\nRun it again a couple of times and check the gear ratio is")
    print("consistent. Longer runs / more revolutions reduce timing error.")
 
 
try:
    main()
except KeyboardInterrupt:
    print("\nAborted.")
finally:
    # Leave outputs in a safe state
    STEP_PIN.value(0)
    SLP_PIN.value(0)