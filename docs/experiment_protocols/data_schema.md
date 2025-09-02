# Minimal Data Schema (per trial)
trial_id,
t_src, t_set_A, t_set_B, t_det_A, t_det_B,          # timestamps
pos_src, pos_A, pos_B,                              # coarse positions (for controls)
setting_A (0/1), setting_B (0/1),
outcome_A (±1), outcome_B (±1),
condition (geom_high/geom_low | rate_fast/rate_slow),
shield_flag (0/1), buffer_depth (int),
cfg_hash, run_id
