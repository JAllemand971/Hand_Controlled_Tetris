<h1>Hand-Controlled Tetris (Python)</h1>
A classic **Tetris** built with **Pygame**, enhanced with a **webcam hand controller** using **MediaPipe** + **OpenCV**.  
Plays perfectly with keyboard; if a camera is available, gestures can move/rotate/drop pieces with **debounced, single-step** actions.

<h2>Demo</h2>

<table align="center">
  <tr>
    <th>Title</th>
    <th>Description</th>
  </tr>
  <tr>
    <td>Hand-Controlled Tetris</td>
    <td>

https://github.com/user-attachments/assets/b5f18adc-2440-45f9-9d2b-6f635a6af3eb

     
    
  </tr>
</table>

---

## What it does

- Runs a **10×20** Tetris well with the 7 classic tetrominoes (I, O, T, S, Z, J, L).
- Applies **gravity**, **collision**, **locking**, and **line clear** rules.
- Tracks **score**, **lines**, and **level** (gravity speeds up with level).
- Keyboard controls always work; **gesture control** is optional and automatic if the camera initializes.
- Hand controller provides **one-cell** left/right steps, **single-edge rotation**, and **soft drop**.
- Includes a **loading / wait screen** while the camera initializes.
- Sound effects + background music (WAV files) with a simple **SOUND_DIR** path.

---

## How it works (high-level pipeline)

1. **State model**
   - **Board**: `20 x 10` grid of cells (None or color).
   - **Piece**: kind (`"I","O","T","S","Z","J","L"`), rotation `0..3`, position `(x,y)`, color.
   - **Bag/Spawner**: 7-bag shuffle; spawn next piece when the current one locks.

2. **Input layer**
   - **Keyboard**: arrow keys / WASD, Z/X rotate, Space for hard drop, Down for soft drop.
   - **Hand controller** (background thread):
     - Captures frames with OpenCV, tracks a hand with MediaPipe (21 landmarks).
     - Classifies simple **poses** (which fingers are extended) into Tetris intents.
     - Uses **debounce** + **edge-triggering** so each pose fires **once** per arm cycle.

3. **Tick**
   - Each frame: process input → soft-drop ticks (if active) → gravity tick (time-based) → draw.

4. **Collision & lock**
   - Moves/rotations happen only if **in bounds** and **not overlapping** existing blocks.
   - If the piece can’t fall, it **locks**, lines are cleared, score/level update, next piece spawns.

5. **Line clear & score**
   - Clear 1/2/3/4 lines and award points (Tetris for 4 lines).
   - **Hard drop** adds a small bonus per row instantly descended.

6. **Game over**
   - If a piece collides at spawn (topout), trigger a **wipe animation** + **menu** (Restart/Quit).

> Rotation uses a simple kick table (nearby offsets) for a classic feel.

---

## Controls

### Keyboard
- **Left / A**: move left  
- **Right / D**: move right  
- **Up / X**: rotate clockwise  
- **Z**: rotate counter-clockwise  
- **Down**: soft drop while held  
- **Space**: hard drop  
- **R** (on game-over screen): restart  
- **Esc / Q** (on game-over screen): quit

### Gestures (pose = which fingers are extended)
- **Right** → `01000` → **Index up** (one-cell step)
- **Left** → `01100` → **Index + Middle up** (one-cell step)
- **Rotate** → `01111` → **Index + Middle + Ring + Pinky up** (single edge)
- **Soft drop** → `10000` → **Thumb down** (continuous while held)

> Debounce/arming ensures a single step per pose. Lower fingers to re-arm.

---

## Live calibration (while camera window is focused)

- `-` / `=`: adjust **index_min_dist** (finger extension strictness)  
- `,` / `.`: adjust **thumb_dir_thresh** (how “down” the thumb must be)  
- `[` / `]`: adjust **pose_on_ms** (hold time before L/R fires)  
- `;` / `'`: adjust **pose_off_ms** (release time to re-arm L/R)  
- `m`: mirror preview  
- `h`: select hand (auto → left → right → auto)  
- `q`: quit camera window

---

## Installation

1. **Clone the repository**
```bash
git clone https://github.com/YourUsername/Hand_Controlled_Tetris.git
cd Hand-Controlled-Tetris
```

2. **Create and activate a virtual environment (recommended)**
```bash
# Windows
python -m venv venv
venv\Scripts\activate
# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

3. **Install dependencies (Python 3.9+ required)**
```bash
pip install -r requirements.txt
# Or manually if requirements.txt is missing
pip install pygame opencv-python mediapipe
```

4. **Run the game**
```bash
python tetris.py
```
