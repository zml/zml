#!/usr/bin/env python3
"""
Analyze the per-block audio degradation curve from 48-block run output.
Determine if block 28 is truly a threshold or if the pattern continues linearly.
"""

# From the 48-block run output, extract per-block audio close_fraction:
audio_per_block = [
    (0, 0.99939546),
    (1, 0.99815151),
    (2, 0.99645415),
    (3, 0.99493505),
    (4, 0.99393911),
    (5, 0.99292380),
    (6, 0.99156746),
    (7, 0.99084666),
    (8, 0.99037776),
    (9, 0.98960271),
    (10, 0.98845176),
    (11, 0.98667302),
    (12, 0.98523918),
    (13, 0.98250713),
    (14, 0.98065864),
    (15, 0.97721354),
    (16, 0.97482251),
    (17, 0.97274151),
    (18, 0.96831597),
    (19, 0.96401057),
    (20, 0.95303974),
    (21, 0.94828482),
    (22, 0.94754077),
    (23, 0.94466921),
    (24, 0.94333225),
    (25, 0.93617854),
    (26, 0.93243893),
    (27, 0.91301618),
    (28, 0.85882471),  # SHARP DROP
    (29, 0.79205419),
    (30, 0.74598137),
    (31, 0.71509952),
    (32, 0.67162311),
    (33, 0.58406188),
    (34, 0.54018632),
    (35, 0.49972486),
    (36, 0.47271438),
    (37, 0.46647910),
    (38, 0.44923813),
    (39, 0.42285544),
    (40, 0.38587395),
    (41, 0.30940368),
    (42, 0.29768880),
    (43, 0.28633433),
    (44, 0.26658219),
    (45, 0.24049789),
    (46, 0.23085240),
    (47, 0.22834899),
]

video_per_block = [
    (0, 1.00000000),
    (1, 0.99980164),
    (2, 0.99936104),
    (3, 0.99927902),
    (4, 0.99922371),
    (5, 0.99907112),
    (6, 0.99904060),
    (7, 0.99898720),
    (8, 0.99897385),
    (9, 0.99893379),
    (10, 0.99891090),
    (11, 0.99886131),
    (12, 0.99881554),
    (13, 0.99876595),
    (14, 0.99875069),
    (15, 0.99870110),
    (16, 0.99861908),
    (17, 0.99859428),
    (18, 0.99849892),
    (19, 0.99840164),
    (20, 0.99834061),
    (21, 0.99825859),
    (22, 0.99818230),
    (23, 0.99797440),
    (24, 0.99728584),
    (25, 0.99613571),
    (26, 0.99467850),
    (27, 0.99139214),
    (28, 0.98377228),  # VISIBLE DROP
    (29, 0.96643448),
    (30, 0.88662148),
    (31, 0.86756325),
    (32, 0.72609711),
    (33, 0.65162086),
    (34, 0.64381218),
    (35, 0.61163330),
    (36, 0.60841560),
    (37, 0.59596252),
    (38, 0.53627586),
    (39, 0.48045349),
    (40, 0.45645142),
    (41, 0.38757896),
    (42, 0.32330894),
    (43, 0.26361656),
    (44, 0.19361877),
    (45, 0.13636589),
    (46, 0.08242226),
    (47, 0.07346153),
]

import math

print(f"\n{'='*60}\nAUDIO/VIDEO PER-BLOCK DEGRADATION ANALYSIS\n{'='*60}\n")

# Analyze pre-block-28 (blocks 0-27)
audio_pre28 = [v for i, v in audio_per_block if i < 28]
video_pre28 = [v for i, v in video_per_block if i < 28]

print(f"BLOCKS 0-27 (Before Block 28)")
print(f"{'='*60}")
print(f"Audio: {audio_pre28[0]:.8f} → {audio_pre28[-1]:.8f} (delta={audio_pre28[-1] - audio_pre28[0]:.8f})")
print(f"Video: {video_pre28[0]:.8f} → {video_pre28[-1]:.8f} (delta={video_pre28[-1] - video_pre28[0]:.8f})")

# Calculate slopes (degradation rate)
audio_slope_pre28 = (audio_pre28[-1] - audio_pre28[0]) / len(audio_pre28)
video_slope_pre28 = (video_pre28[-1] - video_pre28[0]) / len(video_pre28)

print(f"\nPer-block degradation rate (0-27):")
print(f"  Audio: {audio_slope_pre28:.8f} / block")
print(f"  Video: {video_slope_pre28:.8f} / block")

# Block 28 transition
audio_block_28_delta = audio_per_block[28][1] - audio_per_block[27][1]
video_block_28_delta = video_per_block[28][1] - video_per_block[27][1]

print(f"\nBlock 27 → Block 28 transition:")
print(f"  Audio: {audio_per_block[27][1]:.8f} → {audio_per_block[28][1]:.8f} (delta={audio_block_28_delta:.8f})")
print(f"  Video: {video_per_block[27][1]:.8f} → {video_per_block[28][1]:.8f} (delta={video_block_28_delta:.8f})")

# Is block 28's drop anomalous compared to pre-block-28 trend?
expected_audio_at_28_linear = audio_per_block[27][1] + audio_slope_pre28
expected_video_at_28_linear = video_per_block[27][1] + video_slope_pre28

print(f"\nLINEAR EXTRAPOLATION from blocks 0-27 trend:")
print(f"  If audio degradation continued linearly: would be {expected_audio_at_28_linear:.8f}")
print(f"  Actual at block 28: {audio_per_block[28][1]:.8f}")
print(f"  Anomaly: {audio_per_block[28][1] - expected_audio_at_28_linear:.8f} worse than expected")

print(f"\n  If video degradation continued linearly: would be {expected_video_at_28_linear:.8f}")
print(f"  Actual at block 28: {video_per_block[28][1]:.8f}")
print(f"  Anomaly: {video_per_block[28][1] - expected_video_at_28_linear:.8f} worse than expected")

# Post-block-28 analysis
audio_post28 = [v for i, v in audio_per_block if i >= 28]
video_post28 = [v for i, v in video_per_block if i >= 28]

audio_slope_post28 = (audio_post28[-1] - audio_post28[0]) / len(audio_post28)
video_slope_post28 = (video_post28[-1] - video_post28[0]) / len(video_post28)

print(f"\n{'='*60}")
print(f"BLOCKS 28-47 (After Block 28)")
print(f"{'='*60}")
print(f"Audio: {audio_post28[0]:.8f} → {audio_post28[-1]:.8f} (delta={audio_post28[-1] - audio_post28[0]:.8f})")
print(f"Video: {video_post28[0]:.8f} → {video_post28[-1]:.8f} (delta={video_post28[-1] - video_post28[0]:.8f})")

print(f"\nPer-block degradation rate (28-47):")
print(f"  Audio: {audio_slope_post28:.8f} / block")
print(f"  Video: {video_slope_post28:.8f} / block")

print(f"\nDEGRADATION ACCELERATION:")
print(f"  Audio: Pre-28 degradation was {audio_slope_pre28:.8f}/block")
print(f"         Post-28 degradation is {audio_slope_post28:.8f}/block")
print(f"         Acceleration factor: {audio_slope_post28 / (audio_slope_pre28 + 1e-8):.2f}x")

print(f"\n  Video: Pre-28 degradation was {video_slope_pre28:.8f}/block")
print(f"         Post-28 degradation is {video_slope_post28:.8f}/block")
print(f"         Acceleration factor: {video_slope_post28 / (video_slope_pre28 + 1e-8):.2f}x")

# Identify inflection point
print(f"\n{'='*60}")
print(f"INTERPRETATION")
print(f"{'='*60}")

if abs(audio_block_28_delta - audio_slope_pre28) > 0.01:
    print(f"✗ Block 28 shows DISCONTINUITY in audio degradation")
    print(f"  Pre-block-28 trend suggests degradation of ~{audio_slope_pre28:.8f} per block")
    print(f"  Block 28 actual degradation was {audio_block_28_delta:.8f}")
    print(f"  This is an extra ~{audio_block_28_delta - audio_slope_pre28:.8f} beyond trend")
else:
    print(f"✓ Audio degradation at block 28 is consistent with pre-block-28 trend")

if abs(video_block_28_delta - video_slope_pre28) > 0.001:
    print(f"\n✗ Block 28 shows DISCONTINUITY in video degradation")
    print(f"  Pre-block-28 trend suggests degradation of ~{video_slope_pre28:.8f} per block")
    print(f"  Block 28 actual degradation was {video_block_28_delta:.8f}")
    print(f"  This is an extra ~{video_block_28_delta - video_slope_pre28:.8f} beyond trend")
else:
    print(f"\n✓ Video degradation at block 28 is consistent with pre-block-28 trend")

print(f"\n{'='*60}")
print(f"CONCLUSION")
print(f"{'='*60}")
print(f"Block 28 represents a THRESHOLD or STATE INFLECTION:")
print(f"  - Both streams show accelerated degradation post-block-28")
print(f"  - Audio: {abs(audio_slope_post28)/abs(audio_slope_pre28):.1f}x acceleration")
print(f"  - Video: {abs(video_slope_post28)/abs(video_slope_pre28):.1f}x acceleration")
print(f"\nLikely root cause:")
print(f"  - Dense error accumulation through residuals reaches saturation at ~27 blocks")
print(f"  - Block 28 output corruption propagates through cross-stream attention")
print(f"  - Video degrades faster due to higher dimension (d=4096) and head complexity")
