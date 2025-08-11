# =============================================================================
#                      HAND-GESTURE CONTROLLER (for Tetris)
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
from __future__ import annotations

import threading
import math
import time
from typing import Tuple, Optional, List

# Silence noisy logs before importing mediapipe.
# Many beginners get spammed by TF/absl logs; these lines make the console quieter.
import os, warnings
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "3")
warnings.filterwarnings("ignore", category=UserWarning, module=r"google\.protobuf")
try:
    from absl import logging as absl_logging
    absl_logging.set_verbosity(absl_logging.ERROR)
except Exception:
    # If absl isn't available, it's fine — we just skip it.
    pass

# Third-party libs (camera and hand tracking).
# We defensively set a flag in case import fails (e.g., missing package or no OpenCV on the machine).
try:
    import cv2
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except Exception:
    MEDIAPIPE_AVAILABLE = False


# ============================================================
# HandController: non-blocking webcam → high-level Tetris intents
# ============================================================
class HandController:
    """Background thread that converts hand landmarks into simple Tetris intents.

    Typical usage:
        ctrl = HandController(...); ctrl.start()
        ...
        move_dir, soft_drop, rotate_edge, hard_edge = ctrl.poll()

    Why a background thread?
      - Reading camera frames and running MediaPipe can take time.
      - Doing it in a separate thread lets your main game loop stay smooth.
    """

    def __init__(
        self,
        cam_index: int = 0,            # Which camera to open: 0 is usually the default webcam.
        width: int = 720,              # Capture width in pixels (camera request; actual may differ).
        height: int = 480,             # Capture height in pixels.
        # ===== Pose thresholds =====
        index_min_dist: float = 0.32,  # Tip-to-MCP (knuckle) minimum distance to call a finger "extended".
        rotate_debounce_s: float = 0.18,   # Minimum time between rotation triggers (prevents double-rotate).
        # Thumb DOWN detection
        thumb_dir_thresh: float = 0.10,    # How much thumb tip must be vertically below MCP to count as DOWN.
        # Debounce for L/R poses (single-step horizontal moves)
        pose_on_ms: int = 220,             # Must hold pose at least this long before a step fires.
        pose_off_ms: int = 120,            # Must release pose at least this long before next step can fire.
        # ===== hand selection =====
        use_hand: str = "auto",            # 'auto' uses the largest/closest hand; or force 'left'/'right'.
        # ===== preview & debug =====
        show_camera: bool = True,          # Whether to open the preview window and draw HUD/landmarks.
        mirror_preview: bool = False,      # Flip preview horizontally for convenience.
        preview_scale: float = 1.5,        # Enlarge the display window (visual only).
        debug: bool = True,                # Keep around for future extra logs.
    ) -> None:
        if not MEDIAPIPE_AVAILABLE:
            # If imports failed earlier, we stop here; the game can fall back to keyboard.
            raise RuntimeError("MediaPipe / OpenCV not available")

        # Store configuration values on the instance for later use.
        self.width = width
        self.height = height
        self.index_min_dist = index_min_dist
        self.rotate_debounce_s = rotate_debounce_s
        self.thumb_dir_thresh = thumb_dir_thresh
        self.pose_on_ms = pose_on_ms
        self.pose_off_ms = pose_off_ms
        self.use_hand = use_hand
        self.show_camera = show_camera
        self.mirror_preview = mirror_preview
        self.preview_scale = preview_scale
        self.debug = debug

        # Open the camera.
        # Note: some webcams ignore width/height; it's a "request", not a guarantee.
        self.cap = cv2.VideoCapture(cam_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        # Build MediaPipe Hands pipeline once (it's relatively heavy).
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_style = mp.solutions.drawing_styles
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,       # True would re-detect every frame (slower, for photos). We track video.
            max_num_hands=2,               # Detect up to two hands; we will select one to control the game.
            min_detection_confidence=0.6,  # Confidence threshold for initial hand detection.
            min_tracking_confidence=0.6,   # Confidence threshold for tracking between frames.
        )

        # Public state read by the game (reset every frame in the loop):
        self.move_dir: int = 0             # -1 left, 0 none, +1 right
        self.soft_drop: bool = False       # True while thumb-down pose is active
        self.rotate_cw_edge: bool = False  # True only on the single frame when rotate is triggered
        self.hard_drop_edge: bool = False  # Not used here (kept for future extension)

        # Internal arming/timing state for debouncing LEFT/RIGHT and ROTATE.
        self._left_armed: bool = True
        self._right_armed: bool = True
        self._rotate_armed: bool = True
        self._last_rotate_time: float = 0.0
        self._right_pose_since: float = 0.0
        self._left_pose_since: float = 0.0
        self._right_release_since: float = 0.0
        self._left_release_since: float = 0.0
        self._right_pose_prev: bool = False
        self._left_pose_prev: bool = False

        # For HUD: how long the pose has been held/released (seconds).
        self._hud_r_pose_s: float = 0.0
        self._hud_l_pose_s: float = 0.0
        self._hud_r_rel_s: float = 0.0
        self._hud_l_rel_s: float = 0.0

        # HUD counters and rotate indicator bits (for on-screen feedback).
        self._left_steps: int = 0
        self._right_steps: int = 0
        self._rot_idx = self._rot_mid = self._rot_rng = self._rot_pky = False
        self._rotate_flash_until: float = 0.0

        # Thread control: we create a daemon thread running self._loop().
        self._running = False
        self._th = threading.Thread(target=self._loop, daemon=True)

    # --------------------------- lifecycle ---------------------------
    def start(self) -> None:
        """Begin the background camera/pose loop in its own thread."""
        self._running = True
        self._th.start()

    def stop(self) -> None:
        """Signal the loop to stop and clean up resources gracefully."""
        self._running = False
        try:
            self._th.join(timeout=1.0)  # Wait a bit for the thread to exit.
        except Exception:
            pass
        # Close MediaPipe, camera, and window if they exist.
        try:
            self.hands.close()
        except Exception:
            pass
        try:
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass
        try:
            if self.show_camera:
                cv2.destroyWindow("Hand Camera")
        except Exception:
            pass

    # --------------------------- game API ---------------------------
    def poll(self) -> Tuple[int, bool, bool, bool]:
        """Return the current intents (one-shot edges are cleared after reading).

        Returns:
          move_dir         int   -1 (left), 0 (none), +1 (right) for THIS frame
          soft_drop        bool  True while thumb-down is held
          rotate_cw_edge   bool  True only on the exact frame rotation was triggered
          hard_drop_edge   bool  (always False here; placeholder for future)
        """
        rcw = self.rotate_cw_edge
        hd = self.hard_drop_edge
        # Clear edge flags so they only appear once per trigger.
        self.rotate_cw_edge = False
        self.hard_drop_edge = False
        return self.move_dir, self.soft_drop, rcw, hd

    # ----------------------------- helpers -----------------------------
    @staticmethod
    def _dist(a, b) -> float:
        """Euclidean distance between two MediaPipe landmarks (normalized image space)."""
        return math.hypot(a.x - b.x, a.y - b.y)

    def _is_finger_extended(self, lm, tip_id: int, pip_id: int, mcp_id: int, min_dist: float) -> bool:
        """Return True if a finger looks extended.

        We use two conditions:
          1) The tip is vertically above the PIP joint (tip.y < pip.y, since y grows downward).
          2) The tip is far enough from the knuckle (MCP) — at least 'min_dist' away.

        This helps avoid false positives when a finger is only slightly bent.
        """
        tip, pip, mcp = lm[tip_id], lm[pip_id], lm[mcp_id]
        return (tip.y < pip.y - 0.01) and (math.hypot(tip.x - mcp.x, tip.y - mcp.y) >= min_dist)

    def _four_fingers_flags(self, lm) -> Tuple[bool,bool,bool,bool]:
        """Check extension flags for index/middle/ring/pinky with slightly different strictness.

        We relax the threshold a little for the inner fingers (middle, ring, pinky) to make
        the "four up" pose easier to achieve in practice.
        """
        idx = self._is_finger_extended(lm, 8, 6, 5, self.index_min_dist)
        mid = self._is_finger_extended(lm, 12, 10, 9, self.index_min_dist * 0.95)
        rng = self._is_finger_extended(lm, 16, 14, 13, self.index_min_dist * 0.9)
        pky = self._is_finger_extended(lm, 20, 18, 17, self.index_min_dist * 0.85)
        return idx, mid, rng, pky

    def _thumb_up_down(self, lm) -> Tuple[bool, bool]:
        """Detect if the thumb points up or down strongly enough.

        We compare thumb tip (4) to MCP (2) vertically.
        - vy = tip.y - mcp.y, remember y grows downward:
            vy > 0 means tip is lower on the image → visually 'down'
            vy < 0 means tip is higher            → visually 'up'
        - We also check tip vs IP (3) to ensure the thumb is actually pointing,
          not curled close to itself (tip_vs_ip sign helps).

        Returns:
          (thumb_up, thumb_down)
        """
        tip, ip_, mcp = lm[4], lm[3], lm[2]
        vy = tip.y - mcp.y  # y grows downward
        vx = tip.x - mcp.x
        tip_vs_ip = tip.y - ip_.y

        # Only consider vertical-ish thumb directions to avoid sideways false positives.
        vertical_enough = abs(vy) > abs(vx) * 0.6

        down = (vy >= self.thumb_dir_thresh) and (tip_vs_ip > 0.005) and vertical_enough
        up = (vy <= -self.thumb_dir_thresh) and (tip_vs_ip < -0.005) and vertical_enough
        return up, down

    @staticmethod
    def _bbox_area(lm_list) -> float:
        """Approximate hand size by bounding box area in normalized space."""
        xs = [lm.x for lm in lm_list]
        ys = [lm.y for lm in lm_list]
        return max(0.0, (max(xs)-min(xs))) * max(0.0, (max(ys)-min(ys)))

    def _select_active_hand(self, results) -> Optional[Tuple[any, str]]:
        """Pick which detected hand we will use this frame.

        Strategy:
          • If user forced 'left' or 'right', return that specific hand (if seen).
          • Otherwise ('auto'), pick the hand with the largest bounding box area,
            which usually corresponds to the closer, clearer hand.

        Returns:
          (hand_landmarks, label) or None if no hands were detected.
        """
        if not results.multi_hand_landmarks:
            return None

        hands_lm = results.multi_hand_landmarks
        hands_handed = results.multi_handedness if getattr(results, 'multi_handedness', None) else [None]*len(hands_lm)

        pairs: List[Tuple[any, str, float]] = []
        for lm, hd in zip(hands_lm, hands_handed):
            label = None
            if hd is not None and hasattr(hd, 'classification') and len(hd.classification):
                label = hd.classification[0].label  # 'Left' or 'Right'
            else:
                label = 'Unknown'
            area = self._bbox_area(lm.landmark)
            pairs.append((lm, label, area))

        # Respect user's fixed-hand preference if set.
        if self.use_hand in ("left", "right"):
            target = 'Left' if self.use_hand == 'left' else 'Right'
            for lm, label, _ in pairs:
                if label == target:
                    return lm, label

        # Otherwise, choose the largest hand by area.
        lm, label, _ = max(pairs, key=lambda p: p[2])
        return lm, label

    # ----------------------------- core -----------------------------
    def _loop(self) -> None:
        """Main background loop: read frames, detect hand, compute poses, update intents."""
        # Prepare preview window if enabled.
        if self.show_camera:
            cv2.namedWindow("Hand Camera", cv2.WINDOW_NORMAL)
            try:
                cv2.resizeWindow(
                    "Hand Camera",
                    int(self.width * self.preview_scale),
                    int(self.height * self.preview_scale),
                )
            except Exception:
                # Some backends can't resize easily; not critical.
                pass

        prev_time = time.time()
        fps = 0.0

        while self._running:
            # 1) Grab a frame from the camera.
            ok, frame_bgr = self.cap.read()
            if not ok:
                # If the camera is momentarily unavailable, wait a little.
                time.sleep(0.01)
                continue

            # 2) Measure an approximate FPS (for HUD).
            now_frame = time.time()
            dt = now_frame - prev_time
            prev_time = now_frame
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)  # Low-pass filter makes it smoother.

            # 3) Make a copy for drawing; convert BGR→RGB for MediaPipe.
            draw_bgr = frame_bgr.copy()
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            # 4) Run MediaPipe’s hand detection/tracking on this frame.
            results = self.hands.process(frame_rgb)

            # 5) Reset per-frame intents (move_dir, soft_drop). Edge flags are cleared in poll().
            self.move_dir = 0
            self.soft_drop = False

            # 6) Select the active hand (or None if no hands detected).
            sel = self._select_active_hand(results)
            active_label = None

            if sel is not None:
                hand_lms, active_label = sel
                lm = hand_lms.landmark  # List of 21 landmarks

                # --------- Compute finger-extension flags ---------
                idx, mid, rng, pky = self._four_fingers_flags(lm)
                # Store for HUD (we display 1/0 for I/M/R/P).
                self._rot_idx, self._rot_mid, self._rot_rng, self._rot_pky = idx, mid, rng, pky

                # --------- Soft drop (thumb DOWN) ---------
                _up, thumb_down = self._thumb_up_down(lm)
                self.soft_drop = bool(thumb_down)  # While true, your game can drop faster.

                # --------- Define our logical poses for this frame ---------
                right_pose_now = idx and (not mid) and (not rng) and (not pky)   # index only up
                left_pose_now  = idx and mid and (not rng) and (not pky)         # index + middle up
                rotate_pose    = idx and mid and rng and pky                     # all four up
                now = time.time()

                # ---- RIGHT: debounced edge (one cell step right) ----
                if right_pose_now:
                    # First frame we notice RIGHT active: remember the time.
                    if not self._right_pose_prev:
                        self._right_pose_since = now
                    held_ms = (now - self._right_pose_since) * 1000.0
                    self._hud_r_pose_s = held_ms / 1000.0  # HUD live display
                    # Fire only if currently armed AND held long enough.
                    if self._right_armed and held_ms >= self.pose_on_ms:
                        self.move_dir = 1
                        self._right_steps += 1
                        self._right_armed = False  # Disarm until released for pose_off_ms.
                else:
                    # Pose not active → reset hold duration for HUD.
                    self._hud_r_pose_s = 0.0
                    # If we just transitioned from active to inactive, start a release timer.
                    if self._right_pose_prev is True:
                        self._right_release_since = now
                    # How long have we been released?
                    rel_ms = (now - self._right_release_since) * 1000.0 if self._right_release_since else 1e9
                    self._hud_r_rel_s = (rel_ms / 1000.0) if rel_ms < 1e8 else 0.0
                    # Re-arm if released long enough.
                    if rel_ms >= self.pose_off_ms:
                        self._right_armed = True
                self._right_pose_prev = right_pose_now

                # ---- LEFT: debounced edge (one cell step left) ----
                if left_pose_now:
                    if not self._left_pose_prev:
                        self._left_pose_since = now
                    held_ms = (now - self._left_pose_since) * 1000.0
                    self._hud_l_pose_s = held_ms / 1000.0
                    if self._left_armed and held_ms >= self.pose_on_ms:
                        self.move_dir = -1
                        self._left_steps += 1
                        self._left_armed = False
                else:
                    self._hud_l_pose_s = 0.0
                    if self._left_pose_prev is True:
                        self._left_release_since = now
                    rel_ms = (now - self._left_release_since) * 1000.0 if self._left_release_since else 1e9
                    self._hud_l_rel_s = (rel_ms / 1000.0) if rel_ms < 1e8 else 0.0
                    if rel_ms >= self.pose_off_ms:
                        self._left_armed = True
                self._left_pose_prev = left_pose_now

                # --------- ROTATE (four up, edge + re-arm) ---------
                if rotate_pose:
                    # Fire rotate only if armed and enough time passed since last rotation.
                    if self._rotate_armed and (now - self._last_rotate_time) >= self.rotate_debounce_s:
                        self.rotate_cw_edge = True
                        self._last_rotate_time = now
                        self._rotate_flash_until = now + 0.30  # HUD visual feedback
                        self._rotate_armed = False
                else:
                    # Re-arm when rotate pose is released.
                    self._rotate_armed = True

                # --------- Draw landmarks on the preview (optional) ---------
                if self.show_camera:
                    self.mp_draw.draw_landmarks(
                        draw_bgr,
                        hand_lms,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_style.get_default_hand_landmarks_style(),
                        self.mp_style.get_default_hand_connections_style(),
                    )

            # ---------------- HUD (text overlay) ----------------
            if self.show_camera:
                def put(y, text):
                    """Helper to draw white HUD text at a given y row."""
                    cv2.putText(draw_bgr, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (255, 255, 255), 1, cv2.LINE_AA)

                # Basic counters and arming state for LEFT/RIGHT.
                put(24,  f"Left: {self._left_steps}  | ArmedL: {'on' if self._left_armed else 'off'}")
                put(48,  f"Right: {self._right_steps} | ArmedR: {'on' if self._right_armed else 'off'}")

                # Rotate status: flash 'TRIGGERED' for ~0.3s after a rotation.
                rotate_on = (time.time() <= self._rotate_flash_until)
                put(72,  f"Rotate: {'TRIGGERED' if rotate_on else '—'} (I/M/R/P={int(self._rot_idx)}/{int(self._rot_mid)}/{int(self._rot_rng)}/{int(self._rot_pky)})")

                # Down (soft drop) + hand source + FPS
                hand_label = self.use_hand if self.use_hand in ('left','right') else 'auto'
                put(96,  f"Down: {'ON' if self.soft_drop else 'off'} | hand={hand_label} | FPS={fps:4.1f}")

                # Durations for fine calibration (how long pose is held/released).
                put(120, f"R pose: {self._hud_r_pose_s:0.3f}s | R rel: {self._hud_r_rel_s:0.3f}s | on/off={self.pose_on_ms}/{self.pose_off_ms} ms")
                put(144, f"L pose: {self._hud_l_pose_s:0.3f}s | L rel: {self._hud_l_rel_s:0.3f}s | idxMin={self.index_min_dist:0.2f}")

                # Optional mirror so your preview moves like a mirror.
                if self.mirror_preview:
                    draw_bgr = cv2.flip(draw_bgr, 1)

                # Show the annotated frame.
                cv2.imshow("Hand Camera", draw_bgr)

                # -------------- Live hotkeys (window must be focused) --------------
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    # Quick exit (also stops the thread cleanly).
                    self.stop()
                    return
                elif key == ord('m'):
                    self.mirror_preview = not self.mirror_preview
                elif key == ord('-'):
                    # Make index extension stricter (need larger finger stretch).
                    self.index_min_dist = max(0.05, self.index_min_dist - 0.02)
                elif key == ord('='):
                    # Make index extension looser (easier to trigger).
                    self.index_min_dist = min(0.60, self.index_min_dist + 0.02)
                elif key == ord(','):
                    # Require less vertical thumb displacement for DOWN.
                    self.thumb_dir_thresh = max(0.05, self.thumb_dir_thresh - 0.02)
                elif key == ord('.'):
                    # Require more vertical thumb displacement for DOWN.
                    self.thumb_dir_thresh = min(0.60, self.thumb_dir_thresh + 0.02)
                elif key == ord('['):
                    # Must hold the pose less time before a step fires.
                    self.pose_on_ms = max(0, self.pose_on_ms - 10)
                elif key == ord(']'):
                    # Must hold the pose more time (harder to trigger).
                    self.pose_on_ms = min(400, self.pose_on_ms + 10)
                elif key == ord(';'):
                    # Shorter release required to re-arm (faster repeated steps possible).
                    self.pose_off_ms = max(0, self.pose_off_ms - 10)
                elif key == ord('\''):
                    # Longer release required to re-arm (reduces accidental double-steps).
                    self.pose_off_ms = min(600, self.pose_off_ms + 10)
                elif key == ord('h'):
                    # Cycle which hand we use: auto → left → right → auto ...
                    self.use_hand = {'auto':'left','left':'right','right':'auto'}[self.use_hand]

            # Small sleep to cap the background loop rate (~60 Hz). This is optional
            # and mostly to avoid hammering the CPU when vsync isn't present.
            time.sleep(1/60.0)


# ------------------------------------------------------------
# Convenience factory: returns a started controller or None
# ------------------------------------------------------------
def make_hand_controller_or_none(show_camera: bool = True, mirror: bool = False, **kwargs) -> Optional[HandController]:
    """Helper that constructs and starts a HandController.

    Returns:
      HandController if OpenCV/MediaPipe are available and camera opens,
      otherwise None (so your game can fall back to keyboard).
    """
    if not MEDIAPIPE_AVAILABLE:
        return None
    try:
        hc = HandController(show_camera=show_camera, mirror_preview=mirror, **kwargs)
        hc.start()
        return hc
    except Exception:
        # Any startup error (e.g., no camera) → return None gracefully.
        return None


# Manual test (optional): runs the preview only
if __name__ == "__main__":
    # When run directly, we just show the camera window and print poll() output.
    ctrl = make_hand_controller_or_none(show_camera=True, mirror=False)
    if ctrl is None:
        print("Failed to start HandController (check camera / dependencies).")
    else:
        print("Controller running. Close the window or press 'q' or Ctrl+C to exit. Keys: m=mirror, -/= minDist, ,/. thumbDOWN, [/] on-ms, ;/' off-ms, h hand")
        try:
            while True:
                # Poll every 0.2s just to demonstrate outputs in the console.
                print(ctrl.poll())
                time.sleep(0.2)
        except KeyboardInterrupt:
            pass
        finally:
            ctrl.stop()
