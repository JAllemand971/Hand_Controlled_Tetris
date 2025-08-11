# =============================================================================
#                               tetris.py
# =============================================================================
# WHAT THIS PROGRAM DOES
# ----------------------
# This file implements a classic Tetris game using the Pygame library.
# It draws a 10x20 playing field, spawns tetromino pieces, lets the player
# move and rotate them, locks pieces when they land, clears full lines, and
# keeps score and level. It also supports optional hand gestures via a
# separate "hand_controller" module. If a camera is present and the
# controller starts correctly, you can move and rotate with gestures. The
# keyboard always works.
#
# HOW TO RUN
# ----------
# 1) Install dependencies in your Python environment:
#       pip install pygame opencv-python mediapipe
#    Note: OpenCV and MediaPipe are only needed if you plan to use hand
#    controls. For keyboard only, Pygame is enough.
# 2) Make sure the sound files referenced in SOUND_DIR exist, otherwise
#    Pygame will raise an error when loading sounds.
# 3) Run this file:
#       python tetris.py
#
# CONTROLS
# --------
# Keyboard:
#   Left  or A  -> move left one cell
#   Right or D  -> move right one cell
#   Up    or X  -> rotate clockwise
#   Z           -> rotate counterclockwise
#   Down        -> soft drop (falls faster while held)
#   Space       -> hard drop (piece falls to the bottom instantly)
#   Esc or Q    -> quit from the game over menu
#   R           -> restart from the game over menu
#
# Gestures (if hand controller is active):
#   Index to the right or left           -> move right or left one step
#   Four fingers up                      -> rotate once
#   Thumb down                           -> soft drop while thumb is down
# These gestures are recognized by the external hand controller, not here.
#
# BIG PICTURE ARCHITECTURE
# ------------------------
# Game state lives in a few variables:
#   - "board": a 2D list (20 rows x 10 columns) holding either None or a color.
#              None means empty cell. A color tuple means a locked block.
#   - "piece": the current falling tetromino with fields:
#                kind, rot, x, y, shape, color
#   - bag and next spawn: tetromino pieces are drawn from a shuffled "7-bag".
#   - timers and flags: gravity timing, soft drop timing, input repeat.
#
# The main loop repeats many times per second (FPS). Each frame it:
#   1) Processes input events (keyboard, and possibly hands).
#   2) Applies horizontal movement with key-repeat logic.
#   3) Applies soft drop ticks when active.
#   4) Applies gravity at intervals based on the current level.
#      If the piece cannot fall further, it locks into the board.
#      Then full lines are cleared, score and level are updated,
#      and a new piece is spawned.
#   5) Draws the board, the current piece, and HUD text.
#
# COORDINATES AND UNITS
# ---------------------
# - Logical grid: 10 columns (x from 0 to 9) and 20 rows (y from 0 to 19).
# - Each logical cell is drawn as a square of "CELL" pixels.
# - The piece position is stored in grid units. We turn grid units into
#   pixel rectangles when drawing.
# - New pieces start with y < 0 so they can "enter" the field smoothly from
#   above. Cells with y < 0 are considered off screen and are not drawn.
#
# COLLISIONS
# ----------
# The function "collides" checks if a given movement or rotation would put any
# cell of the piece out of bounds or into an occupied cell on the board.
# Movement and rotation are only applied when collision is False.
#
# ROTATION AND WALL KICKS
# -----------------------
# The function "try_rotate" changes the rotation index and then tries small
# horizontal and vertical offsets (kicks) so that a rotation that would collide
# against a wall can still succeed if a near position is valid. This is a simple
# kick system, not the official SRS, but it is good enough for a classic feel.
#
# GRAVITY AND SOFT DROP
# ---------------------
# Gravity moves the piece down every "gravity_s" seconds. As you clear lines,
# your level increases and gravity becomes faster. Soft drop is an extra timer
# that tries to move the piece down more frequently while Down is held or while
# a hand soft drop is active.
#
# SCORING
# -------
# Points are awarded when lines are cleared. The scoring table used here is:
#   1 line: 100
#   2 lines: 300
#   3 lines: 500
#   4 lines (Tetris): 800
# Hard drop adds a small bonus: 2 points per row fallen during the drop.
#
# SOUNDS AND MUSIC
# ----------------
# Short sound effects are loaded into "s_move", "s_rotate", etc. Background
# music is played on loop. If the sound files are missing or the audio device
# cannot be initialized, Pygame may raise errors. Check SOUND_DIR.
#
# HAND CONTROLLER INTEGRATION
# ---------------------------
# The hand controller object is created by make_hand_controller_or_none.
# If it starts successfully, we keep it and call "hand.poll()" each frame to get
# the current actions. If it cannot start, we play in keyboard-only mode.
# There is a loading screen before starting the game, to give the camera a
# moment to initialize and to show the controls.
#
# GAME OVER
# ---------
# When a new piece locks and any of its blocks are above the top row (y < 0),
# the game is over. A wipe effect paints the board from bottom to top, then a
# menu lets you press R to restart or Esc or Q to quit.
#
# READING STRATEGY FOR BEGINNERS
# ------------------------------
# If you are new to this code, read in this order:
#   1) CONFIG constants to understand sizes and timing.
#   2) TETROMINOES to understand how shapes and rotations are stored.
#   3) Piece class and helper functions: new_board, collides, lock_piece,
#      clear_lines, draw_board, draw_piece, try_rotate, hard_drop.
#   4) run_single_game to see the whole game loop.
#   5) main to see how we initialize audio, create the hand controller,
#      and loop over games.
# =============================================================================