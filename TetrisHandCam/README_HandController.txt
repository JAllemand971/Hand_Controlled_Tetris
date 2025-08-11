# =============================================================================
#                      hand_controller.py
# =============================================================================
# PURPOSE
# -------
# This module reads a webcam stream, detects a hand using MediaPipe, and turns
# simple hand poses into Tetris commands (move left/right by one cell, rotate
# once, hold soft drop). It runs in a background thread so the game stays
# responsive and only calls `poll()` once per frame to fetch intents.
#
# WHAT YOU NEED TO KNOW FIRST
# ---------------------------
# • "Landmarks": MediaPipe Hands returns 21 points (x, y) on your hand. All
#   coordinates are normalized between 0 and 1, where y increases downward.
# • "Poses": We classify which fingers are extended and whether the thumb is
#   pointing up or down. These pose checks are simple geometry tests on the
#   landmarks (no machine learning here).
# • "Debounce/Edge": To avoid multiple triggers from a single gesture, we only
#   fire actions on transitions (pose goes from OFF to ON) and require minimum
#   hold/release durations. This is the same idea used for physical button
#   debouncing in electronics.
#
# DESIGN OVERVIEW
# ---------------
# - HandController.start(): starts a daemon thread that loops:
#       capture frame → detect hands → compute poses → set intents
# - Game loop calls HandController.poll() once per frame to read:
#       move_dir (-1/0/+1), soft_drop (bool), rotate_cw_edge (bool), hard_drop_edge (bool)
# - Live calibration: press keys in the camera window to adjust thresholds in
#   real time (index_min_dist, thumb_dir_thresh, pose_on_ms, pose_off_ms, etc).
#
# SAFETY & STABILITY
# ------------------
# - Imports for OpenCV/MediaPipe are wrapped; if missing, we set a flag so the
#   game can fall back to keyboard input gracefully.
# - Thread and camera resources are closed in stop().
# - Preview window is optional and can be mirrored so it behaves like a mirror.
#
# HOW TO INTEGRATE
# ----------------
# - Use make_hand_controller_or_none() to get a controller or None.
# - In your game loop: if controller is not None, call poll() every frame and
#   apply the actions just like keyboard input (one-step left/right, single
#   rotation edges, continuous soft drop).
# =============================================================================

# hand_controller.py — Webcam hand-gesture controller for Tetris
# POSE-BASED: INDEX→RIGHT (single step, debounced), INDEX+MIDDLE→LEFT (single step, debounced), 4-UP ROTATE (edge), THUMB-DOWN DROP
"""
BEGINNER-FRIENDLY EXPLANATION (read this once):
------------------------------------------------
This file turns one hand shown to the webcam into simple Tetris commands.

• We use **MediaPipe Hands** to detect 21 hand landmarks (finger joints and tips).
• We turn these landmarks into **poses** (which fingers are extended).
• Each pose maps to a Tetris intent: move left/right exactly one cell, rotate once,
  or enable soft drop (faster falling).

Key ideas you will see in the code:
1) **Edge-triggered action**: An action fires only when a pose transitions from
   "not active" → "active", rather than continuously every frame. This gives
   you exactly one step/rotation per pose.

2) **Debounce**: We require the pose to be held for a small time (pose_on_ms)
   before firing. We also require a minimum release time (pose_off_ms) before
   we allow the next trigger. This prevents accidental double steps if a finger
   jitters between up/down in the camera frames.

3) **Arming / Re-arming**:
   • For LEFT/RIGHT: after an action fires, the controller becomes "disarmed".
     It will **not** fire again until you release the pose long enough
     (pose_off_ms). Then it becomes "armed" again for the next single step.
   • For ROTATE: similar, but with its own simple debounce time (rotate_debounce_s).

4) **Coordinate system** (important if you debug):
   • MediaPipe gives normalized coordinates in the image: x and y in [0, 1].
   • y grows downward (top of the image is small y, bottom is larger y).
   • Distances we compare are measured in normalized image space.

5) **Finger-extension test**:
   • We say a finger is "extended" if its tip is above (smaller y) than the joint
     below it (PIP), and if the tip is far enough from the knuckle (MCP). This
     rejects half-bent or ambiguous states.
   • index_min_dist controls how strict we are about the distance part.

6) **Thumb up/down**:
   • The thumb is special: we look at the vertical direction from the MCP to the tip.
     If the tip y is clearly below the MCP (remember y grows downward), we call it DOWN
     and turn soft drop ON. If it's clearly above, that's UP (we don't use it here).
   • thumb_dir_thresh controls how strong that up/down must be to count.

7) **Live calibration**:
   • While the camera window is focused, you can press keys to tune thresholds.
     Example: '-' or '=' adjusts index_min_dist to make the "index extended"
     detection stricter/looser. This is useful to adapt to different hands/lighting.

8) **Threading model**:
   • The HandController runs in a background thread reading camera frames,
     so your game loop stays responsive.
   • Your game reads the current "intents" by calling poll() once per frame.

Poses (Y-axis = "finger up" = finger extended):
  • RIGHT  → **Index up only** (I=1, M=0, R=0, P=0). One step; re-arm by lowering index.
  • LEFT   → **Index + Middle up** (I=1, M=1, R=0, P=0). One step; re-arm by lowering any of them.
  • ROTATE → **Four up** (I=1, M=1, R=1, P=1). Fires once; lower any finger to re-arm.
  • SOFT DROP → **Thumb DOWN** (no other constraint). While down, soft drop = True.

Minimal HUD (top-left of the camera window):
  • Counters for Left/Right/Rotate/Down and arming state.
  • Debug durations for how long you held/released LEFT/RIGHT poses.
  • All of this helps you calibrate thresholds live.

Live calibration keys (focus the OpenCV window):
  • '-' / '='    : index_min_dist  -/+ 0.02  (finger extension strictness)
  • ',' / '.'    : thumb_dir_thresh (DOWN)  -/+ 0.02
  • '[' / ']'    : pose_on_ms (L/R)  -/+ 10 ms  (must hold pose this long before it fires)
  • ';' / '\''  : pose_off_ms (L/R) -/+ 10 ms  (must release this long before re-arming)
  • 'h'          : toggle use_hand  auto → left → right → auto
  • 'm'          : mirror preview (flip horizontally) for convenience
  • 'q'          : quit

Game API (unchanged, what your game reads each frame):
    move_dir, soft_drop, rotate_cw_edge, hard_drop_edge = controller.poll()
  • move_dir: -1 (left), 0 (no horizontal move this frame), +1 (right)
  • soft_drop: True while thumb-down pose is active, else False
  • rotate_cw_edge: True only on the single frame where rotation is triggered
  • hard_drop_edge: always False here (feature placeholder, not used)

Dependencies to install (Python):
    pip install opencv-python mediapipe
"""