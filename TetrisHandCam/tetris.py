# tetris.py
# =============================================================================
#                               TETRIS (Pygame)
# =============================================================================

import pygame, sys, random, time, os
from hand_controller import make_hand_controller_or_none

# ============================================================
#                           CONFIG
# ============================================================
# General window and gameplay configuration. Changing these changes the
# scale, timing, or general behavior of Tetris without touching logic.
#
# The big idea: keep all "magic numbers" in one place.
# This makes the code easier to tune and reason about.

# Grid width and height in cells. Standard Tetris uses 10x20.
COLS, ROWS = 10, 20

# Pixel size of a single cell square. Higher means bigger visuals.
CELL = 30

# Width of the side panel (HUD) on the right, in pixels.
SIDEBAR_W = 6 * CELL

# Total window size in pixels. Playing field width plus sidebar width.
W, H = COLS * CELL + SIDEBAR_W, ROWS * CELL

# Frames per second for drawing and input handling. This does not directly
# control gravity speed. A higher FPS means smoother animation and more
# frequent input checks.
FPS = 30

# Base gravity interval in seconds. The piece attempts to fall once every
# BASE_GRAVITY_S seconds at level 1. We reduce this as the level increases.
BASE_GRAVITY_S = 0.8

# Soft drop tick interval in seconds. While soft drop is active, we try an
# extra downward step each SOFT_DROP_INTERVAL_S seconds. This makes soft drop
# feel faster than normal gravity.
SOFT_DROP_INTERVAL_S = 0.2

# Folder of .wav files (short sound effects + background music).
# If this path is wrong or files are missing, sound loading will fail.
SOUND_DIR = r"C:\Users\jerem\Documents\TetrisHandCam\Sound_Effects"

# Colors used to draw the grid and pieces as RGB tuples.
BLACK   = (0, 0, 0)        # Background
GRID    = (40, 40, 40)     # Grid line color
WHITE   = (240, 240, 240)  # HUD text
GREY    = (90, 90, 90)     # Game over wipe color

# Classic-ish Tetris colors for each tetromino type.
C_I = (0, 255, 255)
C_O = (255, 255, 0)
C_T = (160, 0, 240)
C_S = (0, 255, 0)
C_Z = (255, 0, 0)
C_J = (0, 0, 255)
C_L = (255, 128, 0)

# ============================================================
#                        TETROMINO SHAPES
# ============================================================
# Each tetromino is defined by:
#   - "color": the color used when drawing the blocks.
#   - "rot": a list of four 4x4 matrices, one per rotation state.
# In each 4x4 matrix, 1 means there is a filled block, 0 means empty.
# The active piece scans its 4x4 matrix and for each 1, it draws a square
# at (piece.x + i, piece.y + j) on the board.

TETROMINOES = {
    "I": {"color": C_I, "rot": [
        [[0,0,0,0],[1,1,1,1],[0,0,0,0],[0,0,0,0]],
        [[0,0,1,0],[0,0,1,0],[0,0,1,0],[0,0,1,0]],
        [[0,0,0,0],[0,0,0,0],[1,1,1,1],[0,0,0,0]],
        [[0,1,0,0],[0,1,0,0],[0,1,0,0],[0,1,0,0]],
    ]},
    # O piece is symmetrical, so we reuse the same matrix 4 times.
    "O": {"color": C_O, "rot": [[[0,1,1,0],[0,1,1,0],[0,0,0,0],[0,0,0,0]]]*4},
    "T": {"color": C_T, "rot": [
        [[0,1,0,0],[1,1,1,0],[0,0,0,0],[0,0,0,0]],
        [[0,1,0,0],[0,1,1,0],[0,1,0,0],[0,0,0,0]],
        [[0,0,0,0],[1,1,1,0],[0,1,0,0],[0,0,0,0]],
        [[0,1,0,0],[1,1,0,0],[0,1,0,0],[0,0,0,0]],
    ]},
    "S": {"color": C_S, "rot": [
        [[0,1,1,0],[1,1,0,0],[0,0,0,0],[0,0,0,0]],
        [[0,1,0,0],[0,1,1,0],[0,0,1,0],[0,0,0,0]],
        [[0,0,0,0],[0,1,1,0],[1,1,0,0],[0,0,0,0]],
        [[1,0,0,0],[1,1,0,0],[0,1,0,0],[0,0,0,0]],
    ]},
    "Z": {"color": C_Z, "rot": [
        [[1,1,0,0],[0,1,1,0],[0,0,0,0],[0,0,0,0]],
        [[0,0,1,0],[0,1,1,0],[0,1,0,0],[0,0,0,0]],
        [[0,0,0,0],[1,1,0,0],[0,1,1,0],[0,0,0,0]],
        [[0,1,0,0],[1,1,0,0],[1,0,0,0],[0,0,0,0]],
    ]},
    "J": {"color": C_J, "rot": [
        [[1,0,0,0],[1,1,1,0],[0,0,0,0],[0,0,0,0]],
        [[0,1,1,0],[0,1,0,0],[0,1,0,0],[0,0,0,0]],
        [[0,0,0,0],[1,1,1,0],[0,0,1,0],[0,0,0,0]],
        [[0,1,0,0],[0,1,0,0],[1,1,0,0],[0,0,0,0]],
    ]},
    "L": {"color": C_L, "rot": [
        [[0,0,1,0],[1,1,1,0],[0,0,0,0],[0,0,0,0]],
        [[0,1,0,0],[0,1,0,0],[0,1,1,0],[0,0,0,0]],
        [[0,0,0,0],[1,1,1,0],[1,0,0,0],[0,0,0,0]],
        [[1,1,0,0],[0,1,0,0],[0,1,0,0],[0,0,0,0]],
    ]},
}

# ============================================================
#                  UTILITY / GAME OBJECTS
# ============================================================

def generate_bag():
    # Returns a shuffled list of all 7 tetromino kinds.
    # This is the "7-bag" system: you get each piece once per bag, in random order.
    bag = list(TETROMINOES.keys())
    random.shuffle(bag)
    return bag

class Piece:
    # Represents the current falling tetromino.
    # Fields:
    #   kind: string like "I" or "T"
    #   rot: current rotation index 0..3
    #   x, y: top-left grid position of the 4x4 bounding box of the piece
    #   shape: reference to the rotation matrices from TETROMINOES[kind]["rot"]
    #   color: color tuple from TETROMINOES
    def __init__(self, kind):
        self.kind = kind
        self.rot = 0
        self.x = 3        # Start near the center horizontally
        self.y = -2       # Start above the visible field so it "falls in"
        self.shape = TETROMINOES[kind]["rot"]
        self.color = TETROMINOES[kind]["color"]

    def cells(self, rot=None):
        # Returns a list of cell coordinates occupied by this piece, based on
        # the current or a hypothetical rotation index "r".
        # We scan the 4x4 matrix and for each "1" we add the absolute grid
        # position (x + i, y + j).
        r = self.rot if rot is None else rot
        out = []
        mat = self.shape[r]
        for j in range(4):
            for i in range(4):
                if mat[j][i]:
                    out.append((self.x + i, self.y + j))
        return out

def new_board():
    # Creates a 20x10 grid filled with None. Each entry will later hold either
    # None (empty) or a color tuple (locked block).
    return [[None for _ in range(COLS)] for _ in range(ROWS)]

def collides(board, piece, rot=None, dx=0, dy=0):
    # Checks if applying a movement (dx, dy) and an optional rotation "rot"
    # would cause any cell of "piece" to go out of bounds or hit an occupied cell.
    for (x, y) in piece.cells(rot=rot):
        nx, ny = x + dx, y + dy
        # Out of bounds on left or right, or below the bottom
        if nx < 0 or nx >= COLS or ny >= ROWS:
            return True
        # If within visible rows and the cell is already occupied, that is a collision
        if ny >= 0 and board[ny][nx] is not None:
            return True
    return False

def lock_piece(board, piece):
    # Converts the falling piece into locked blocks on the "board".
    # If any part of the piece is above the top (y < 0) when locking,
    # we signal a "topout" which ends the game.
    topout = False
    for (x, y) in piece.cells():
        if y < 0:
            topout = True
            continue
        if 0 <= y < ROWS:
            board[y][x] = piece.color
    return topout

def clear_lines(board):
    # Removes any full rows and counts how many were cleared.
    # Strategy:
    #   - Keep rows that are NOT full (at least one None).
    #   - Count how many we removed.
    #   - Insert that many empty rows at the top to maintain height.
    keep = [row for row in board if any(cell is None for cell in row)]
    cleared = ROWS - len(keep)
    while len(keep) < ROWS:
        keep.insert(0, [None for _ in range(COLS)])
    return keep, cleared

def draw_board(screen, board):
    # Draws the background, then each cell of the board, and a grid line on top.
    screen.fill(BLACK)
    for y in range(ROWS):
        for x in range(COLS):
            rect = pygame.Rect(x*CELL, y*CELL, CELL, CELL)
            if board[y][x] is not None:
                pygame.draw.rect(screen, board[y][x], rect)
            pygame.draw.rect(screen, GRID, rect, 1)

def draw_piece(screen, piece):
    # Draws the currently falling piece. Cells with y < 0 are off screen and are
    # not drawn.
    for (x, y) in piece.cells():
        if y >= 0:
            rect = pygame.Rect(x*CELL, y*CELL, CELL, CELL)
            pygame.draw.rect(screen, piece.color, rect)
            pygame.draw.rect(screen, (20,20,20), rect, 1)  # darker outline

def try_rotate(board, piece, dir_=+1):
    # Attempts to rotate the piece by +1 or -1 rotation steps.
    # If the rotation collides, we try a series of small "kicks" (offsets).
    # If any kicked position is valid, we accept the rotation and offset.
    new_rot = (piece.rot + dir_) % len(piece.shape)
    kicks = [(0,0), (-1,0), (1,0), (0,-1), (-2,0), (2,0)]
    for dx, dy in kicks:
        if not collides(board, piece, rot=new_rot, dx=dx, dy=dy):
            piece.rot = new_rot
            piece.x += dx
            piece.y += dy
            return True
    return False

def hard_drop(board, piece):
    # Moves the piece straight down as far as it can go, then returns how many
    # rows it fell. The caller usually locks the piece right after and uses this
    # return value to award a small score bonus.
    d = 0
    while not collides(board, piece, dy=d+1):
        d += 1
    piece.y += d
    return d

# ============================================================
#                      GAME OVER: WIPE + MENU
# ============================================================

def game_over_wipe(screen, board, font):
    # Visual effect that paints the board from bottom to top with a grey color,
    # one cell at a time. We process Pygame events during the effect so the
    # window remains responsive.
    draw_board(screen, board)
    pygame.display.flip()
    delay_ms = 12
    for y in range(ROWS - 1, -1, -1):
        for x in range(COLS):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); sys.exit()
            rect = pygame.Rect(x * CELL, y * CELL, CELL, CELL)
            pygame.draw.rect(screen, GREY, rect)
            pygame.draw.rect(screen, GRID, rect, 1)
            pygame.display.update(rect)
            pygame.time.wait(delay_ms)

def game_over_menu(screen, score, lines, level):
    # Draws a translucent overlay with "GAME OVER" and your final stats.
    # Then waits for the player to press R to restart, or Q or Esc to quit.
    overlay = pygame.Surface((W, H), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 160))
    screen.blit(overlay, (0, 0))

    title_font = pygame.font.SysFont("Consolas", 36, bold=True)
    info_font  = pygame.font.SysFont("Consolas", 22)
    hint_font  = pygame.font.SysFont("Consolas", 20)

    title = title_font.render("GAME OVER", True, WHITE)
    stats = info_font.render(f"Score: {score}   Lines: {lines}   Level: {level}", True, WHITE)
    hint1 = hint_font.render("Press R to Restart", True, (230, 230, 230))
    hint2 = hint_font.render("Press Q or Esc to Quit", True, (200, 200, 200))

    screen.blit(title, (W//2 - title.get_width()//2, H//2 - 80))
    screen.blit(stats, (W//2 - stats.get_width()//2, H//2 - 30))
    screen.blit(hint1, (W//2 - hint1.get_width()//2, H//2 + 20))
    screen.blit(hint2, (W//2 - hint2.get_width()//2, H//2 + 50))
    pygame.display.flip()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    return False
                if event.key in (pygame.K_r,):
                    return True

# ============================================================
#                           MAIN GAME
# ============================================================

def run_single_game(hand=None):
    # Runs one full play session until topout, then shows game over and returns
    # True if the user wants to restart, or False to exit.
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Tetris — keyboard + hand controls")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Consolas", 16)

    # Load sound effects and music. These must exist in SOUND_DIR or loading
    # will fail. Music plays on loop at -1.
    s_move       = pygame.mixer.Sound(os.path.join(SOUND_DIR, "move.wav"))
    s_rotate     = pygame.mixer.Sound(os.path.join(SOUND_DIR, "rotate.wav"))
    s_lock       = pygame.mixer.Sound(os.path.join(SOUND_DIR, "piece_landed.wav"))
    s_line       = pygame.mixer.Sound(os.path.join(SOUND_DIR, "line.wav"))
    s_tetris     = pygame.mixer.Sound(os.path.join(SOUND_DIR, "4_lines.wav"))
    s_level_up   = pygame.mixer.Sound(os.path.join(SOUND_DIR, "level_up.wav"))
    s_game_over  = pygame.mixer.Sound(os.path.join(SOUND_DIR, "game_over.wav"))
    pygame.mixer.music.load(os.path.join(SOUND_DIR, "background.wav"))
    pygame.mixer.music.play(-1)

    # If caller did not pass a hand controller, try to create one now.
    # If creation fails, we will play keyboard-only.
    if hand is None:
        hand = make_hand_controller_or_none(show_camera=True, mirror=False)

    # Small inner function used to show a clean loading or status screen before
    # the game starts. It waits for the player to press Enter or Return.
    def _draw_wait_screen(status_text: str) -> bool:
        title_font  = pygame.font.SysFont("Consolas", 28, bold=True)
        hint_font   = pygame.font.SysFont("Consolas", 18)
        howto_font  = pygame.font.SysFont("Consolas", 16)
        howto_text = "Gestures: Thumb DOWN = Soft Drop | Four Fingers UP = Rotate | Index or Index+Middle = Move"
        cx, cy = screen.get_rect().center
        maxw = screen.get_width() - 40
        # Utility for centered word wrapping so the how-to text fits nicely.
        def _blit_wrapped(text: str, y_start: int) -> None:
            words = text.split(" ")
            line = ""
            y = y_start
            for w in words:
                test = (line + " " + w).strip()
                if howto_font.size(test)[0] <= maxw:
                    line = test
                else:
                    surf = howto_font.render(line, True, (180, 180, 180))
                    screen.blit(surf, surf.get_rect(center=(cx, y)))
                    y += howto_font.get_linesize() + 2
                    line = w
            if line:
                surf = howto_font.render(line, True, (180, 180, 180))
                screen.blit(surf, surf.get_rect(center=(cx, y)))
        # If hand exists, we assume we are starting the camera.
        # Otherwise tell the player we are in keyboard-only mode.
        wait_status = (status_text if status_text
                       else ("Starting camera... (this can take a few seconds)" if hand is not None
                             else "Camera unavailable → Keyboard-only mode. Press ENTER to play."))
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False
                if event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                        return True
                    elif event.key in (pygame.K_ESCAPE, pygame.K_q):
                        return False
            screen.fill((10, 10, 20))
            title  = title_font.render("TETRIS — Loading", True, WHITE)
            hint   = hint_font.render("Press ENTER to start", True, WHITE)
            status = hint_font.render(wait_status, True, (200, 200, 200))
            screen.blit(title,  title.get_rect(center=(cx, cy - 90)))
            screen.blit(hint,   hint.get_rect(center=(cx, cy - 48)))
            screen.blit(status, status.get_rect(center=(cx, cy - 12)))
            _blit_wrapped(howto_text, cy + 24)
            pygame.display.flip()
            clock.tick(30)

    # Show the wait screen once so the user knows what is going on.
    wait_msg = "Starting camera... (this can take a few seconds)" if hand is not None else \
               "Camera unavailable → Keyboard-only mode. Press ENTER to play."
    if not _draw_wait_screen(wait_msg):
        return False

    # Initialize the game state: empty board, first bag and piece, and stats.
    board = new_board()
    bag = generate_bag()
    piece = Piece(bag.pop())
    if not bag:
        bag = generate_bag()

    score = 0
    level = 1
    lines_total = 0

    # Timing for gravity and soft drop.
    gravity_s = BASE_GRAVITY_S
    last_fall = time.time()

    keyboard_soft_drop = False
    hand_soft_drop = False
    last_soft_drop = time.time()

    # Horizontal key repeat support. When you hold left or right, we first move
    # once instantly, then after a short delay we repeat at a fixed rate.
    move_left_held = move_right_held = False
    move_repeat_delay = 0.15
    last_move_time = 0.0

    running = True
    try:
        while running:
            now = time.time()
            _dt = clock.tick(FPS) / 1000.0  # Time since last frame in seconds

            # 1) INPUT: process window events and keyboard presses.
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False

                elif event.type == pygame.KEYDOWN:
                    # Move left if not colliding
                    if event.key in (pygame.K_LEFT, pygame.K_a):
                        if not collides(board, piece, dx=-1):
                            piece.x -= 1
                            s_move.play()
                        move_left_held = True
                        last_move_time = now

                    # Move right if not colliding
                    elif event.key in (pygame.K_RIGHT, pygame.K_d):
                        if not collides(board, piece, dx=+1):
                            piece.x += 1
                            s_move.play()
                        move_right_held = True
                        last_move_time = now

                    # Rotate clockwise
                    elif event.key in (pygame.K_UP, pygame.K_x):
                        if try_rotate(board, piece, +1):
                            s_rotate.play()

                    # Rotate counterclockwise
                    elif event.key in (pygame.K_z,):
                        if try_rotate(board, piece, -1):
                            s_rotate.play()

                    # Start soft drop
                    elif event.key == pygame.K_DOWN:
                        keyboard_soft_drop = True

                    # Hard drop: drop to bottom, then lock, clear, score, spawn
                    elif event.key == pygame.K_SPACE:
                        rows = hard_drop(board, piece)
                        topout = lock_piece(board, piece)
                        s_lock.play()
                        if topout:
                            running = False
                        else:
                            board, cleared = clear_lines(board)
                            if cleared:
                                (s_tetris if cleared == 4 else s_line).play()
                            # Hard drop bonus plus line clear points
                            score += 2 * rows + [0, 100, 300, 500, 800][cleared]
                            lines_total += cleared
                            prev_level = level
                            level = 1 + lines_total // 10
                            if level > prev_level:
                                s_level_up.play()
                            # Make gravity faster as level increases, but clamp
                            gravity_s = max(0.1, BASE_GRAVITY_S * (0.9 ** (level-1)))
                            # Spawn next piece from the bag
                            if not bag:
                                bag = generate_bag()
                            piece = Piece(bag.pop())
                            last_fall = now
                            # If new piece cannot fit, that is game over
                            if collides(board, piece):
                                running = False

                elif event.type == pygame.KEYUP:
                    # Stop repeating when keys are released
                    if event.key in (pygame.K_LEFT, pygame.K_a):
                        move_left_held = False
                    elif event.key in (pygame.K_RIGHT, pygame.K_d):
                        move_right_held = False
                    elif event.key == pygame.K_DOWN:
                        keyboard_soft_drop = False

            # 2) OPTIONAL HAND INPUT: get actions from hand controller if present.
            if hand is not None:
                # hand.poll() returns a tuple: (move_dir, soft_drop, rotate_edge, hard_drop_edge)
                move_dir, hand_soft_drop_now, rotate_edge, _hard = hand.poll()

                # Move one step per poll, guarded by collision
                if move_dir == -1 and not collides(board, piece, dx=-1):
                    piece.x -= 1
                    s_move.play()
                elif move_dir == 1 and not collides(board, piece, dx=+1):
                    piece.x += 1
                    s_move.play()

                # Rotate once per "edge" signal
                if rotate_edge:
                    if try_rotate(board, piece, +1):
                        s_rotate.play()

                # Hand soft drop is active while the gesture is active
                hand_soft_drop = bool(hand_soft_drop_now)

            # 3) KEY REPEAT: after a short delay, keep moving while held.
            if move_left_held or move_right_held:
                if now - last_move_time >= move_repeat_delay:
                    if move_left_held and not collides(board, piece, dx=-1):
                        piece.x -= 1
                        last_move_time = now
                        s_move.play()
                    if move_right_held and not collides(board, piece, dx=+1):
                        piece.x += 1
                        last_move_time = now
                        s_move.play()

            # 4) SOFT DROP TICKS: try to drop at a faster interval when active.
            soft_drop_active = keyboard_soft_drop or hand_soft_drop
            if soft_drop_active and (now - last_soft_drop) >= SOFT_DROP_INTERVAL_S:
                if not collides(board, piece, dy=1):
                    piece.y += 1
                last_soft_drop = now

            # 5) GRAVITY: every "current_gravity" seconds, try to fall one row.
            # If soft drop is active, we speed gravity by a factor of 0.15.
            current_gravity = gravity_s * (0.15 if soft_drop_active else 1.0)
            if now - last_fall >= current_gravity:
                if not collides(board, piece, dy=1):
                    piece.y += 1
                else:
                    # Cannot fall, so lock the piece into the board
                    topout = lock_piece(board, piece)
                    s_lock.play()
                    if topout:
                        running = False
                    else:
                        # Clear any full lines and update score and level
                        board, cleared = clear_lines(board)
                        if cleared:
                            (s_tetris if cleared == 4 else s_line).play()
                        score += [0, 100, 300, 500, 800][cleared]
                        lines_total += cleared
                        prev_level = level
                        level = 1 + lines_total // 10
                        if level > prev_level:
                            s_level_up.play()
                        # Speed up gravity with level increases
                        gravity_s = max(0.1, BASE_GRAVITY_S * (0.9 ** (level-1)))
                        # Spawn next piece
                        if not bag:
                            bag = generate_bag()
                        piece = Piece(bag.pop())
                        # If the new piece immediately collides, that means topout
                        if collides(board, piece):
                            running = False
                last_fall = now

            # 6) DRAW: board, piece, and HUD on the right.
            draw_board(screen, board)
            draw_piece(screen, piece)

            hud_lines = [
                f"Score: {score}",
                f"Lines: {lines_total}",
                f"Level: {level}",
                ("Hands: ON" if hand is not None else "Hands: OFF"),
                "Thumb DOWN = Soft Drop",
                "Thumb UP = Rotate",
                "Index Left/Right = Move",
            ]
            for i, line in enumerate(hud_lines):
                txt = font.render(line, True, WHITE)
                screen.blit(txt, (COLS * CELL + 8, 8 + i * (font.get_linesize() + 4)))

            pygame.display.flip()

    finally:
        # If an exception occurs inside the loop, we still want to stop music
        # and proceed to the game over flow. The "finally" block ensures we
        # reach the code below even if something goes wrong.
        pass

    # Game finished: play the game over sound, show wipe and menu.
    pygame.mixer.music.stop()
    s_game_over.play()
    game_over_wipe(screen, board, font)

    restart = game_over_menu(screen, score, lines_total, level)
    return restart


def main():
    # Audio initialization before Pygame init can reduce sound latency.
    pygame.mixer.pre_init(44100, -16, 2, 512)
    pygame.init()

    # Create one hand controller and reuse it across restarts.
    hand = make_hand_controller_or_none(show_camera=True, mirror=False)

    # Run repeated games until the user quits from the game over menu.
    while True:
        restart = run_single_game(hand=hand)
        if not restart:
            break

    # Best effort to stop the camera thread if it exists.
    if hand is not None:
        try:
            hand.stop()
        except Exception:
            pass

    # Clean shutdown.
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
