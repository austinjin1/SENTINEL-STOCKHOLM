# Water-quality estimate from one drone frame, produced by WaterDroneNet on the
# ground computer. Built on both machines but only published by sentinel_inference.

std_msgs/Header header        # stamp/frame copied from the source camera frame

# Per-target predictions. Index-aligned: param_names[i] <-> mu[i] <-> sigma[i].
# Order matches WaterDroneNet.TARGET_PARAMS: [DO, pH, Turb, Temp, SpCond].
string[]  param_names         # human-readable target names
float32[] mu                  # predicted mean value (target's native unit)
float32[] sigma               # predicted aleatoric uncertainty (std, same unit)

# Per-frame confidence from the model's trust router.
float32 trust_score           # [0, 1]
string  trust_flag            # "green" | "yellow" | "red"

# Provenance: which input bands were real this frame.
bool nir_present              # false => NIR channel was synthesized/zero-filled
