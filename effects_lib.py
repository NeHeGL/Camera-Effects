# 2026 Jeff Molofee (NeHe)
import time, math, random
import cv2
import numpy as np

def fx_none(frame, state):
    return frame

def fx_glitch(frame, state):
    intensity = state.get('glitchIntensity', 50) / 100.0
    if random.random() > 0.6 * intensity:
        return frame
    out = frame.copy()
    H, W = frame.shape[:2]
    num_slices = int(4 + intensity * 12)
    for _ in range(num_slices):
        y0 = random.randint(0, H - 1)
        sh = random.randint(2, 20)
        y1 = min(y0 + sh, H)
        sr = int((random.random() - 0.5) * intensity * 80)
        sg = int((random.random() - 0.5) * intensity * 40)
        sb = int((random.random() - 0.5) * intensity * 60)
        xs = np.arange(W)
        out[y0:y1, :, 2] = frame[y0:y1, (xs + sr) % W, 2]
        out[y0:y1, :, 1] = frame[y0:y1, (xs + sg) % W, 1]
        out[y0:y1, :, 0] = frame[y0:y1, (xs + sb) % W, 0]
    return out


# ── Radar scope — full-colour video always visible, arm = brightness ──
# The camera image is always shown in full colour.
# A brightness multiplier per-pixel is derived from angular distance
# to the sweep arm: 1.0 at the arm, fading to a dim minimum behind it.
_radar_angle_arr = None   # (H, W) precomputed pixel angles, cached
_radar_last_size = None   # (H, W) to detect resize

def fx_radar(frame, state):
    global _radar_angle_arr, _radar_last_size

    H, W = frame.shape[:2]
    t    = time.time()
    cx, cy = W / 2.0, H / 2.0

    # ── Precompute pixel angle map once ──────────────────────────────
    if _radar_angle_arr is None or _radar_last_size != (H, W):
        Y, X = np.mgrid[:H, :W].astype(np.float32)
        _radar_angle_arr = (np.arctan2(X - cx, -(Y - cy))) % (2 * math.pi)
        _radar_last_size = (H, W)

    # ── Current sweep angle (clockwise, ~4 s per rotation) ───────────
    sweep_speed = 2 * math.pi / 4.0
    sweep_angle = (t * sweep_speed) % (2 * math.pi)

    # ang_behind: 0 = just swept by the arm (bright), increases going forward
    # toward the arm's leading edge (dark, hasn't been swept yet)
    # We use (sweep_angle - pixel_angle) % 2π so that:
    #   0       = pixel is right at the arm (just swept) → brightest
    #   2π      = pixel is just ahead of the arm         → darkest (0)
    ang_behind = (sweep_angle - _radar_angle_arr) % (2 * math.pi)

    # Fade speed: how quickly the image goes dark behind the arm.
    # radarFade slider: 1=very slow (almost always bright), 100=very fast (goes dark quickly).
    # We raise the linear ramp to a power: power=1 is linear, higher = faster initial drop.
    fade_speed = state.get('radarFade', 75) / 100.0   # 0.25..1.0
    fade_power = max(0.1, fade_speed * 4.0)            # 0.04..4.0 — controls curve shape

    linear_ramp = np.clip(1.0 - ang_behind / (2 * math.pi), 0.0, 1.0)
    bright_map = np.where(
        ang_behind < math.radians(6),                    # crisp leading edge band
        1.0,
        np.power(linear_ramp, fade_power)
    ).astype(np.float32)

    # ── Convert frame to phosphor green (grayscale → green channel only) ─
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    green = np.zeros((*gray.shape, 3), dtype=np.float32)
    green[:, :, 1] = gray   # green channel only
    out = np.clip(green * bright_map[:, :, None], 0, 255).astype(np.uint8)

    # ── Dim green reticle overlay ─────────────────────────────────────
    max_r = min(H, W) // 2
    reticle_col = (0, 45, 0)
    for frac in [0.25, 0.5, 0.75, 1.0]:
        cv2.circle(out, (int(cx), int(cy)), int(max_r * frac), reticle_col, 1, cv2.LINE_AA)
    cv2.line(out, (int(cx), 0), (int(cx), H), reticle_col, 1, cv2.LINE_AA)
    cv2.line(out, (0, int(cy)), (W, int(cy)), reticle_col, 1, cv2.LINE_AA)

    # ── Bright sweep arm line ─────────────────────────────────────────
    arm_ex = int(cx + math.sin(sweep_angle) * max_r)
    arm_ey = int(cy - math.cos(sweep_angle) * max_r)
    cv2.line(out, (int(cx), int(cy)), (arm_ex, arm_ey), (200, 220, 200), 2, cv2.LINE_AA)

    return out

def fx_twist(frame, state):
    amount = state.get('twistAmount', 50) / 100.0 * math.pi * 2
    H, W = frame.shape[:2]
    cy, cx = H / 2.0, W / 2.0
    Y, X = np.mgrid[:H, :W].astype(np.float32)
    dx, dy = X - cx, Y - cy
    dist = np.sqrt(dx**2 + dy**2)
    angle = dist / max(math.sqrt(cx**2 + cy**2), 1) * amount
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    src_x = np.clip(cx + dx * cos_a - dy * sin_a, 0, W - 1).astype(np.float32)
    src_y = np.clip(cy + dx * sin_a + dy * cos_a, 0, H - 1).astype(np.float32)
    return cv2.remap(frame, src_x, src_y, cv2.INTER_LINEAR)

def fx_wave(frame, state):
    amp  = state.get('waveAmplitude', 20)   # px of whole-frame shift
    freq = state.get('waveFrequency', 5)    # relative speed
    H, W = frame.shape[:2]
    t = time.time()
    spd = freq * 0.15   # convert slider to Hz-ish

    # Whole-frame sloshing translation — different slow frequencies on each axis
    tx = amp * math.sin(t * spd * 0.97)
    ty = amp * math.sin(t * spd * 0.73 + 1.1) * 0.7

    # Slight rolling rotation that oscillates (adds queasiness)
    angle = math.radians(amp * 0.12 * math.sin(t * spd * 0.53 + 2.3))

    cx, cy = W / 2.0, H / 2.0
    ca, sa = math.cos(angle), math.sin(angle)
    # Rotation + translation affine matrix
    M = np.float32([
        [ca, -sa, cx*(1-ca) + cy*sa + tx],
        [sa,  ca, cy*(1-ca) - cx*sa + ty]
    ])
    return cv2.warpAffine(frame, M, (W, H),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT_101)

def fx_kaleidoscope(frame, state):
    segs = max(2, state.get('kaleidoSegments', 6))
    H, W = frame.shape[:2]
    t = time.time()
    rot_angle = t * 0.2

    cx, cy = W / 2.0, H / 2.0
    max_r = math.hypot(W, H) / 2.0

    # warpPolar: dsize=(cols,rows), output shape=(rows,cols)
    # rows = angle (0..2pi), cols = radius (0..max_r)
    polar_a = 2048  # high angular resolution — makes roll-jitter sub-pixel at edges
    polar_r = 512   # radial resolution

    # Cartesian -> polar
    polar = cv2.warpPolar(frame, (polar_r, polar_a), (cx, cy), max_r,
                          cv2.WARP_POLAR_LINEAR | cv2.INTER_LINEAR)
    # shape: (polar_a, polar_r, 3)

    # Rotate by integer row shift (jitter < 1 polar pixel = tiny fraction of degree)
    row_shift = int(rot_angle / (2.0 * math.pi) * polar_a) % polar_a
    polar = np.roll(polar, -row_shift, axis=0)

    # Extract first wedge (row slice), tile with alternating mirrors
    seg_h = max(1, polar_a // segs)
    wedge = polar[:seg_h, :, :]
    wedge_flip = cv2.flip(wedge, 0)

    strips = [wedge if i % 2 == 0 else wedge_flip for i in range(segs)]
    tiled = np.concatenate(strips, axis=0)
    if tiled.shape[0] != polar_a:
        tiled = cv2.resize(tiled, (polar_r, polar_a), interpolation=cv2.INTER_LINEAR)

    # polar -> Cartesian
    result = cv2.warpPolar(tiled, (W, H), (cx, cy), max_r,
                           cv2.WARP_POLAR_LINEAR | cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
    return result

def fx_pixelate(frame, state):
    ps = max(2, state.get('pixelSize', 12))
    H, W = frame.shape[:2]
    s = cv2.resize(frame, (max(1,W//ps), max(1,H//ps)), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(s, (W, H), interpolation=cv2.INTER_NEAREST)

# NV scope: dynamic — mask is computed per-frame with a moving centre,
# so we only cache the pixel coordinate grids (Y, X arrays), not the mask itself.
_nv_grid_cache = {}   # (H, W) -> (Y, X) float32 grids

def _get_nv_grids(H, W):
    key = (H, W)
    if key not in _nv_grid_cache:
        Y, X = np.mgrid[:H, :W].astype(np.float32)
        _nv_grid_cache[key] = (Y, X)
    return _nv_grid_cache[key]

def fx_night_vision(frame, state):
    H, W = frame.shape[:2]
    t = time.time()

    # ── Lissajous figure-8 scope movement ────────────────────────────
    # Scope radius: smaller than before so it has room to wander
    scope_r = min(H, W) * 0.32

    # Travel range: how far the centre moves from frame centre
    travel_x = (W / 2.0 - scope_r) * 0.55
    travel_y = (H / 2.0 - scope_r) * 0.55

    # Lissajous 1:2 sideways figure-8 (∞ on its side):
    #   x(t) = sin(t)     → sweeps left←→right once per period
    #   y(t) = sin(2t)    → sweeps up↕down TWICE per period
    # Together this traces a classic sideways ∞ path:
    #   start centre → top-right → centre → bottom-right →
    #   centre → top-left → centre → bottom-left → centre
    # Period ~20 s feels like a deliberate searching pan.
    period = 20.0
    ang = (t / period) * 2.0 * math.pi
    lx = math.sin(ang)
    ly = math.sin(2.0 * ang)

    cx = W / 2.0 + lx * travel_x
    cy = H / 2.0 + ly * travel_y

    # ── Build dynamic scope mask ──────────────────────────────────────
    Y, X = _get_nv_grids(H, W)
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)

    fringe_w = max(8.0, scope_r * 0.05)
    alpha = np.clip((dist - scope_r) / fringe_w, 0.0, 1.0)

    # Faint ring glow at the scope edge
    ring_w = max(3.0, scope_r * 0.02)
    ring = np.clip(1.0 - np.abs(dist - scope_r) / ring_w, 0.0, 1.0) * 0.35
    ring *= np.clip(1.0 - alpha, 0.0, 1.0)

    # ── Apply green image + mask ──────────────────────────────────────
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    noise = np.random.uniform(-15, 15, gray.shape).astype(np.float32)
    v = np.clip(gray * 1.4 + noise, 0, 255)

    green = np.zeros((H, W, 3), dtype=np.float32)
    green[:, :, 1] = np.clip(v * 1.2, 0, 255)

    green *= (1.0 - alpha[:, :, None])
    green[:, :, 1] = np.clip(green[:, :, 1] + ring * 180.0, 0, 255)

    out8 = green.astype(np.uint8)

    # ── Reticle centred on the moving scope ──────────────────────────
    cx_i, cy_i = int(cx), int(cy)
    cross_len = int(scope_r * 0.12)
    cross_gap = int(scope_r * 0.04)
    cross_w   = 1
    col = (0, 55, 0)
    cv2.line(out8, (cx_i - cross_len, cy_i), (cx_i - cross_gap, cy_i), col, cross_w, cv2.LINE_AA)
    cv2.line(out8, (cx_i + cross_gap, cy_i), (cx_i + cross_len, cy_i), col, cross_w, cv2.LINE_AA)
    cv2.line(out8, (cx_i, cy_i - cross_len), (cx_i, cy_i - cross_gap), col, cross_w, cv2.LINE_AA)
    cv2.line(out8, (cx_i, cy_i + cross_gap), (cx_i, cy_i + cross_len), col, cross_w, cv2.LINE_AA)
    return out8

_THERMAL_STOPS = np.array([[0,0,0],[0,0,255],[0,255,255],[0,255,0],[255,255,0],[255,0,0],[255,255,255]], dtype=np.float32)
_THERMAL_POS   = np.array([0, 0.15, 0.35, 0.5, 0.65, 0.85, 1.0], dtype=np.float32)

def _build_thermal_lut():
    """Precompute a 256-entry BGR LUT for the thermal palette."""
    lut = np.zeros((256, 3), dtype=np.uint8)
    t = np.arange(256, dtype=np.float32) / 255.0
    idx = np.clip(np.searchsorted(_THERMAL_POS, t, side='right') - 1, 0, len(_THERMAL_POS) - 2)
    lo, hi = _THERMAL_POS[idx], _THERMAL_POS[idx + 1]
    f = np.clip((t - lo) / np.maximum(hi - lo, 1e-6), 0, 1)
    rgb = (_THERMAL_STOPS[idx] + (_THERMAL_STOPS[idx+1] - _THERMAL_STOPS[idx]) * f[:, None]).astype(np.uint8)
    lut[:] = rgb[:, ::-1]   # RGB -> BGR
    return lut

_THERMAL_LUT = _build_thermal_lut()   # built once at import time

def fx_thermal(frame, state):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)   # uint8, no float conversion
    b = cv2.LUT(gray, _THERMAL_LUT[:, 0])
    g = cv2.LUT(gray, _THERMAL_LUT[:, 1])
    r = cv2.LUT(gray, _THERMAL_LUT[:, 2])
    return cv2.merge([b, g, r])

_sepia_vig_cache = {}   # (H, W) -> vignette array

def _get_sepia_vig(H, W):
    key = (H, W)
    if key not in _sepia_vig_cache:
        cy, cx = H / 2.0, W / 2.0
        Y, X = np.mgrid[:H, :W].astype(np.float32)
        vig = np.clip((np.sqrt((X-cx)**2+(Y-cy)**2) - H*0.2) / (H*0.65), 0, 1) * 0.72
        _sepia_vig_cache[key] = vig
    return _sepia_vig_cache[key]

def fx_vintage_sepia(frame, state):
    H, W = frame.shape[:2]
    f = frame.astype(np.float32)
    b, g, r = f[:,:,0], f[:,:,1], f[:,:,2]
    grain = np.random.uniform(-18, 18, (H, W)).astype(np.float32)
    out = np.zeros_like(frame, dtype=np.float32)
    out[:,:,2] = r*0.393 + g*0.769 + b*0.189 + grain
    out[:,:,1] = r*0.349 + g*0.686 + b*0.168 + grain*0.9
    out[:,:,0] = r*0.272 + g*0.534 + b*0.131 + grain*0.7
    vig = _get_sepia_vig(H, W)
    out *= (1.0 - vig[:, :, None])
    if random.random() < 0.3:
        x = random.randint(0, W-1)
        a = 0.15 + random.random() * 0.35
        out[:, x, :] = out[:, x, :] * (1-a) + np.array([180, 230, 255]) * a
    return np.clip(out, 0, 255).astype(np.uint8)

def fx_neon_edge(frame, state):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.clip(np.sqrt(gx**2 + gy**2) * 2, 0, 255).astype(np.uint8)
    H, W = mag.shape
    Y, X = np.mgrid[:H, :W]
    hue = ((X / W * 300 + Y / H * 60) % 180).astype(np.uint8)
    hsv = np.zeros((H, W, 3), dtype=np.uint8)
    hsv[:,:,0] = hue; hsv[:,:,1] = 255; hsv[:,:,2] = mag
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def fx_mirror(frame, state):
    # Vertical (left→right) and/or horizontal (top→bottom) mirror.
    # mirrorVert and mirrorHoriz are bool controls (1=on, 0=off).
    do_v = state.get('mirrorVert',  1)   # default: vertical on
    do_h = state.get('mirrorHoriz', 0)   # default: horizontal off
    H, W = frame.shape[:2]
    out = frame.copy()
    if do_v:
        left  = out[:, :W//2, :]
        right = cv2.flip(left, 1)
        out   = np.concatenate([left, right], axis=1)
    if do_h:
        top = out[:H//2, :]
        out[H//2:H//2+top.shape[0], :] = cv2.flip(top, 0)
    return out

def fx_chromatic(frame, state):
    shift = state.get('chromaticShift', 6)
    out = np.empty_like(frame)
    out[:, :, 0] = np.roll(frame[:, :, 0], -shift, axis=1)
    out[:, :, 1] = frame[:, :, 1]
    out[:, :, 2] = np.roll(frame[:, :, 2],  shift, axis=1)
    return out

_promist_lut = None

def _get_promist_lut():
    # Overexposed S-curve: deep blacks, highlights pushed to clip
    global _promist_lut
    if _promist_lut is None:
        x = np.linspace(0, 1, 256, dtype=np.float32)
        x = np.clip(x * 1.25 - 0.05, 0, 1)
        x = x * x * (3 - 2 * x)
        _promist_lut = (x * 255).astype(np.uint8)
    return _promist_lut

def fx_promist(frame, state):
    """Pro-Mist: high-contrast, slightly overexposed frame with highlight
    halation (tight + wide glow, screen-blended) and a light overall softening."""
    glow     = state.get('mistGlow', 60) / 100.0
    softness = state.get('mistSoft', 60) / 100.0
    mono     = state.get('mistMono', 100) / 100.0
    H, W = frame.shape[:2]
    frame = cv2.LUT(frame, _get_promist_lut())
    # Blur at quarter resolution: big sigmas are expensive and the result is soft anyway
    small = cv2.resize(frame, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
    k = min(W, H) * 0.015
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mask = np.clip((gray - 135) / 100.0, 0, 1)[:, :, None]
    hi = (small * mask).astype(np.uint8)
    tight = cv2.GaussianBlur(hi, (0, 0), max(1.0, k * 0.4))
    wide  = cv2.GaussianBlur(hi, (0, 0), max(2.0, k * (0.8 + 2.0 * softness)))
    bloom = cv2.addWeighted(tight, 0.6, wide, 0.3 + 0.5 * softness, 0)
    bloom = cv2.resize(bloom, (W, H), interpolation=cv2.INTER_LINEAR)
    # Faint overall diffusion; detail stays crisp
    mist = cv2.GaussianBlur(small, (0, 0), k)
    mist = cv2.resize(mist, (W, H), interpolation=cv2.INTER_LINEAR)
    base = cv2.addWeighted(frame, 1.0 - 0.25 * softness, mist, 0.25 * softness, 0)
    # Screen blend: 1 - (1-a)(1-b)
    a = base.astype(np.float32) / 255.0
    b = np.clip(bloom.astype(np.float32) / 255.0 * glow * 1.9, 0, 1)
    out = (255.0 * (1.0 - (1.0 - a) * (1.0 - b))).astype(np.uint8)
    if mono > 0:
        g = cv2.cvtColor(cv2.cvtColor(out, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        out = cv2.addWeighted(out, 1.0 - mono, g, mono, 0)
    return out

_ASCII_CHARS = ' .:-=+*#%@'
_ascii_tile_cache = {}  # (cw, ch) -> list of 10 grayscale tiles

def _get_ascii_tiles(cw, ch):
    key = (cw, ch)
    if key not in _ascii_tile_cache:
        fs = max(0.3, ch / 14.0)
        tiles = []
        for c in _ASCII_CHARS:
            tile = np.zeros((ch, cw), dtype=np.uint8)
            cv2.putText(tile, c, (1, ch - 2), cv2.FONT_HERSHEY_PLAIN, fs, 255, 1, cv2.LINE_AA)
            tiles.append(tile)
        _ascii_tile_cache[key] = tiles
    return _ascii_tile_cache[key]

def fx_ascii(frame, state):
    res = max(4, state.get('asciiRes', 8))
    H, W = frame.shape[:2]
    cols, rows = max(1, W // res), max(1, H // res)
    small = cv2.resize(frame, (cols, rows))
    gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    cw, ch = W // cols, H // rows
    tiles = _get_ascii_tiles(cw, ch)   # list of (ch, cw) uint8 grayscale tiles
    n = len(_ASCII_CHARS) - 1

    # Map each cell luminance -> char index  (rows x cols)
    char_idx = (gray.astype(np.float32) * (n / 255.0) + 0.5).astype(np.int32).clip(0, n)

    # Stack tiles into a lookup array: (n+1, ch, cw)
    tile_stack = np.stack(tiles, axis=0).astype(np.float32) / 255.0  # (10, ch, cw)

    # Select tile for every cell: (rows, cols, ch, cw)
    selected = tile_stack[char_idx]  # fancy index -> (rows, cols, ch, cw)

    # Colors per cell: (rows, cols, 3) -> (rows, cols, 1, 1, 3)
    colors = small.astype(np.float32)[:, :, None, None, :]  # broadcast over tile pixels

    # Apply: (rows, cols, ch, cw, 3)
    colored = selected[:, :, :, :, None] * colors

    # Rearrange to (rows*ch, cols*cw, 3)
    out = colored.clip(0, 255).astype(np.uint8)
    out = out.transpose(0, 2, 1, 3, 4).reshape(rows * ch, cols * cw, 3)

    return cv2.resize(out, (W, H), interpolation=cv2.INTER_NEAREST)

_roto_angle = 0.0
_roto_last  = None

def fx_roto_zoom(frame, state):
    global _roto_angle, _roto_last
    now = time.time()
    dt = min(0.1, now - _roto_last) if _roto_last else 0.016
    _roto_last = now
    # rotoSpeed: -100..100; default 15 gives a gentle auto-spin
    speed = state.get('rotoSpeed', 10) / 100.0
    _roto_angle += speed * math.pi * 2.0 * dt
    # Auto-pulsing zoom: oscillates between ~0.8x and ~2.2x on its own
    # rotoZoom slider (50..400) acts as a center offset / 100
    zoom_center = state.get('rotoZoom', 150) / 100.0
    zoom = zoom_center + math.sin(now * 0.4) * (zoom_center * 0.35)
    zoom = max(0.3, zoom)
    H, W = frame.shape[:2]
    cx, cy = W / 2.0, H / 2.0
    ca, sa = math.cos(_roto_angle) / zoom, math.sin(_roto_angle) / zoom
    M = np.float32([[ca, -sa, cx*(1-ca)+cy*sa], [sa, ca, cy*(1-ca)-cx*sa]])
    return cv2.warpAffine(frame, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)

_water_prev_s = None   # previous tiny gray frame
_water_disp   = None   # full-res displacement field (H, W, 2)
_water_vel    = None   # full-res velocity field (H, W, 2)

# Tiny flow resolution — fast enough for real-time
_FLOW_W, _FLOW_H = 160, 90

def fx_water_push(frame, state):
    global _water_prev_s, _water_disp, _water_vel
    H, W = frame.shape[:2]
    strength = state.get('waterStrength', 30)

    # Downscale to tiny gray for flow computation
    small = cv2.resize(frame, (_FLOW_W, _FLOW_H))
    gray_s = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    if _water_prev_s is None or _water_prev_s.shape != gray_s.shape:
        _water_prev_s = gray_s.copy()
        _water_disp = np.zeros((H, W, 2), dtype=np.float32)
        _water_vel  = np.zeros((H, W, 2), dtype=np.float32)

    # Fast optical flow at tiny resolution — minimal params
    flow_s = cv2.calcOpticalFlowFarneback(
        _water_prev_s, gray_s, None,
        pyr_scale=0.5, levels=2, winsize=9,
        iterations=2, poly_n=5, poly_sigma=1.1, flags=0)
    # flow_s shape: (90, 160, 2)

    # Scale flow to full-res pixel units and upsample
    flow_s[:, :, 0] *= float(W) / _FLOW_W
    flow_s[:, :, 1] *= float(H) / _FLOW_H
    flow_up = cv2.resize(flow_s, (W, H), interpolation=cv2.INTER_LINEAR)

    # Motion magnitude mask — only inject where something is actually moving
    mag = np.sqrt(flow_s[:, :, 0]**2 + flow_s[:, :, 1]**2)
    mag_norm = np.clip(mag / 3.0, 0, 1)   # threshold: >3px/frame at tiny res
    mag_up = cv2.resize(mag_norm, (W, H), interpolation=cv2.INTER_LINEAR)
    mag_up = cv2.GaussianBlur(mag_up, (21, 21), 0)   # spread around hand shape

    # Inject force only where motion is significant
    force_x = flow_up[:, :, 0] * mag_up * (strength * 0.4)
    force_y = flow_up[:, :, 1] * mag_up * (strength * 0.4)

    # Spring-mass: displacement springs back to zero (rubber-sheet bounce-back)
    spring_k = 0.10
    damping  = 0.82
    _water_vel[:, :, 0] = _water_vel[:, :, 0] * damping + force_x - _water_disp[:, :, 0] * spring_k
    _water_vel[:, :, 1] = _water_vel[:, :, 1] * damping + force_y - _water_disp[:, :, 1] * spring_k
    _water_disp[:, :, 0] = np.clip(_water_disp[:, :, 0] + _water_vel[:, :, 0], -strength, strength)
    _water_disp[:, :, 1] = np.clip(_water_disp[:, :, 1] + _water_vel[:, :, 1], -strength, strength)

    # Smooth the displacement so warps look fluid not blocky
    _water_disp[:, :, 0] = cv2.GaussianBlur(_water_disp[:, :, 0], (15, 15), 0)
    _water_disp[:, :, 1] = cv2.GaussianBlur(_water_disp[:, :, 1], (15, 15), 0)

    Y, X = np.mgrid[:H, :W].astype(np.float32)
    src_x = np.clip(X + _water_disp[:, :, 0], 0, W - 1)
    src_y = np.clip(Y + _water_disp[:, :, 1], 0, H - 1)

    _water_prev_s = gray_s
    return cv2.remap(frame, src_x, src_y, cv2.INTER_LINEAR)

# Rotating cube (software rasterizer using OpenCV)
_cube_angleY = 0.0
_cube_last   = None

def fx_rotating_cube(frame, state):
    global _cube_angleY, _cube_last
    now = time.time()
    dt = min(0.1, now - _cube_last) if _cube_last else 0.016
    _cube_last = now
    # cubeSpeed: -100..100, normalize so 100 = ~1 rev/sec
    speed = state.get('cubeSpeed', 10) / 100.0
    _cube_angleY += speed * math.pi * 2.0 * dt
    aX = 0.4 + math.sin(now * 0.18) * 0.35
    aY = _cube_angleY
    H, W = frame.shape[:2]
    out = np.full((H, W, 3), 17, dtype=np.uint8)
    size = min(W, H) * 0.468
    fov, dist_z = 2.8, 2.2
    cyY, syY = math.cos(aY), math.sin(aY)
    cxX, sxX = math.cos(aX), math.sin(aX)

    def project(x, y, z):
        x1 =  cyY*x + syY*z
        z1 = -syY*x + cyY*z
        y2 =  cxX*y - sxX*z1
        z2 =  sxX*y + cxX*z1
        d  = fov / (fov + z2 + dist_z)
        return (W/2 + x1*size*d, H/2 + y2*size*d, z2)

    corners = [(-1,-1,-1),(1,-1,-1),(1,1,-1),(-1,1,-1),
               (-1,-1, 1),(1,-1, 1),(1, 1, 1),(-1, 1, 1)]
    proj = [project(*c) for c in corners]

    faces = [(4,5,6,7),(1,0,3,2),(5,1,2,6),(0,4,7,3),(0,1,5,4),(7,6,2,3)]
    face_data = []
    for fi in faces:
        pts  = [proj[i] for i in fi]
        avgZ = sum(p[2] for p in pts) / 4
        p0,p1,p2 = pts[0],pts[1],pts[2]
        cross = (p1[0]-p0[0])*(p2[1]-p0[1]) - (p1[1]-p0[1])*(p2[0]-p0[0])
        face_data.append((avgZ, cross, pts))
    face_data.sort(key=lambda x: -x[0])

    cs = min(W, H)
    cx0, cy0 = (W - cs)//2, (H - cs)//2
    src_pts = np.float32([[cx0,cy0],[cx0+cs,cy0],[cx0+cs,cy0+cs],[cx0,cy0+cs]])
    for avgZ, cross, pts in face_data:
        if cross >= 0:
            continue
        dst = np.float32([[p[0],p[1]] for p in pts])
        M   = cv2.getPerspectiveTransform(src_pts, dst)
        warped = cv2.warpPerspective(frame, M, (W, H))
        mask = np.zeros((H, W), dtype=np.uint8)
        cv2.fillConvexPoly(mask, dst.astype(np.int32), 255)
        out = np.where(mask[:,:,None] > 0, warped, out)
        cv2.polylines(out, [dst.astype(np.int32)], True, (200,200,200), 1, cv2.LINE_AA)
    return out

# ── Emboss — raised-relief with user-controlled light direction ───
def fx_emboss(frame, state):
    depth     = state.get('embossDepth', 50) / 100.0   # 0..1
    # lightDir: -100 = light far left, 0 = classic upper-left, +100 = light far right
    # Centre (0) = 225° = classic upper-left emboss angle; sweeps ±90° to full left/right
    light_dir = state.get('embossLight', 0) / 100.0    # -1..+1
    azimuth = math.radians(225.0 + light_dir * 90.0)   # 135°..315° sweep

    # Low elevation = strong raking light = genuine emboss shadow depth
    elev = math.radians(25)
    lx   =  math.cos(azimuth) * math.cos(elev)
    ly   =  math.sin(azimuth) * math.cos(elev)
    lz   =  math.sin(elev)

    # ── Surface normals from blurred grayscale ────────────────────────
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 1.5).astype(np.float32)

    bump_scale = 1.5 + depth * 3.5
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3) * (bump_scale / 255.0)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3) * (bump_scale / 255.0)

    nx = -gx
    ny = -gy
    nz = np.ones_like(nx)
    norm = np.sqrt(nx*nx + ny*ny + nz*nz)
    nx /= norm;  ny /= norm;  nz /= norm

    # ── Lambertian diffuse + ambient ─────────────────────────────────
    diffuse = np.clip(nx * lx + ny * ly + nz * lz, 0.0, 1.0)
    ambient = 0.18
    lit = np.clip(ambient + diffuse * (1.0 - ambient), 0.0, 1.0)
    lit = np.clip(0.05 + lit * 0.92, 0.0, 1.0)

    out8 = (lit * 255.0).astype(np.uint8)
    return cv2.cvtColor(out8, cv2.COLOR_GRAY2BGR)

# ── TV Snow — static noise with ghost person faintly visible ──────
def fx_tv_snow(frame, state):
    ghost = state.get('snowGhost', 30) / 100.0   # 0=invisible, 1=fully visible
    H, W = frame.shape[:2]

    # Full-screen random noise (true analog TV static)
    noise = np.random.randint(0, 256, (H, W), dtype=np.uint8)

    # Occasional horizontal scan-line banding (clumps of brighter/darker rows)
    band = np.random.randint(60, 200, (H, 1), dtype=np.uint8)
    noise = np.clip(noise.astype(np.int16) * band.astype(np.int16) // 128, 0, 255).astype(np.uint8)

    # Rare full-frame white flash (channel roll)
    if np.random.random() < 0.02:
        noise = np.clip(noise.astype(np.int16) + np.random.randint(80, 160), 0, 255).astype(np.uint8)

    # Grayscale snow as BGR
    snow_bgr = cv2.cvtColor(noise, cv2.COLOR_GRAY2BGR)

    # Ghost: desaturate + dim the camera frame, then blend over snow
    gray_ghost = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ghost_bgr  = cv2.cvtColor(gray_ghost, cv2.COLOR_GRAY2BGR).astype(np.float32)

    out = np.clip(
        snow_bgr.astype(np.float32) * (1.0 - ghost * 0.6) +
        ghost_bgr * ghost * 0.7,
        0, 255
    ).astype(np.uint8)
    return out

# ── Hamster Eyes — comically enlarged eyes using FaceLandmarker ───
_hamster_landmarker      = None
_hamster_landmarker_ready = False

# MediaPipe 478-landmark Face Mesh eye contour indices
# (same indices as the legacy face_mesh model, still valid for face_landmarker.task)
_LEFT_EYE_IDX  = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
_RIGHT_EYE_IDX = [33,  7,  163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

_FACE_LANDMARKER_MODEL_URL = (
    'https://storage.googleapis.com/mediapipe-models/'
    'face_landmarker/face_landmarker/float16/latest/face_landmarker.task'
)
_FACE_LANDMARKER_MODEL_FILE = 'face_landmarker.task'

def _get_model_path():
    """Return a persistent path for face_landmarker.task next to the exe (or script)."""
    import os, sys
    if getattr(sys, 'frozen', False):
        # Running as PyInstaller exe — save next to the exe so it persists across runs
        base = os.path.dirname(sys.executable)
    else:
        base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, _FACE_LANDMARKER_MODEL_FILE)

def _ensure_hamster_landmarker():
    global _hamster_landmarker, _hamster_landmarker_ready
    if _hamster_landmarker_ready:
        return _hamster_landmarker
    try:
        import os
        model_path = _get_model_path()
        if not os.path.exists(model_path):
            print('[HamsterEyes] Downloading face_landmarker.task …')
            import urllib.request
            urllib.request.urlretrieve(_FACE_LANDMARKER_MODEL_URL, model_path)
            print('[HamsterEyes] Download complete.')
        from mediapipe.tasks import python as _mpt
        from mediapipe.tasks.python import vision as _mpv
        opts = _mpv.FaceLandmarkerOptions(
            base_options=_mpt.BaseOptions(model_asset_path=model_path),
            running_mode=_mpv.RunningMode.IMAGE,
            num_faces=4,
            min_face_detection_confidence=0.4,
            min_face_presence_confidence=0.4,
            min_tracking_confidence=0.4,
        )
        _hamster_landmarker = _mpv.FaceLandmarker.create_from_options(opts)
        _hamster_landmarker_ready = True
        print('[HamsterEyes] FaceLandmarker ready.')
    except Exception as e:
        print(f'[HamsterEyes] Failed to load FaceLandmarker: {e}')
        _hamster_landmarker = None
    return _hamster_landmarker


# ── Angry Eyes — angry V-shaped brows + red eye tint ─────────────
_angry_landmarker      = None
_angry_landmarker_ready = False

def _ensure_angry_landmarker():
    global _angry_landmarker, _angry_landmarker_ready
    if _angry_landmarker_ready:
        return _angry_landmarker
    # Reuse the already-downloaded face_landmarker.task model
    try:
        import os
        model_path = _get_model_path()
        if not os.path.exists(model_path):
            print('[AngryEyes] Downloading face_landmarker.task …')
            import urllib.request
            urllib.request.urlretrieve(_FACE_LANDMARKER_MODEL_URL, model_path)
            print('[AngryEyes] Download complete.')
        from mediapipe.tasks import python as _mpt
        from mediapipe.tasks.python import vision as _mpv
        opts = _mpv.FaceLandmarkerOptions(
            base_options=_mpt.BaseOptions(model_asset_path=model_path),
            running_mode=_mpv.RunningMode.IMAGE,
            num_faces=4,
            min_face_detection_confidence=0.4,
            min_face_presence_confidence=0.4,
            min_tracking_confidence=0.4,
        )
        _angry_landmarker = _mpv.FaceLandmarker.create_from_options(opts)
        _angry_landmarker_ready = True
        print('[AngryEyes] FaceLandmarker ready.')
    except Exception as e:
        print(f'[AngryEyes] Failed to load FaceLandmarker: {e}')
        _angry_landmarker = None
    return _angry_landmarker

# Brow landmark indices (MediaPipe 478-point model)
# Left brow  (person's right, camera-left):  inner=285, outer=276
# Right brow (person's left,  camera-right): inner=55,  outer=46
# Full arch for natural brow shape:
_LEFT_BROW_IDX  = [336, 296, 334, 293, 300, 276, 283, 282, 295, 285]
_RIGHT_BROW_IDX = [107,  66, 105,  63,  70,  46,  53,  52,  65,  55]

# Eye outline indices for red tinting
_LEFT_EYE_OUTLINE  = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
_RIGHT_EYE_OUTLINE = [33,  7,  163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

def _draw_angry_brow(out, lm, brow_idx, H, W, intensity, flip_anger):
    """Draw a thick angry eyebrow — fully vectorised, zero Python loops."""
    pts = np.array([(lm[i].x * W, lm[i].y * H) for i in brow_idx], dtype=np.float32)
    pts = pts[np.argsort(pts[:, 0])]          # sort left-to-right

    n   = len(pts)
    t   = np.linspace(0.0, 1.0, n, dtype=np.float32)  # 0=leftmost, 1=rightmost

    # Lift whole brow up
    pts[:, 1] -= intensity * 6.0

    # Vectorised tilt: inner corner (right side for left brow, left for right brow) goes up
    if flip_anger:
        tilt = (t - 0.5) * intensity * 20.0
    else:
        tilt = (0.5 - t) * intensity * 20.0
    pts[:, 1] -= tilt

    thickness = max(3, int(intensity * 14))
    pts_i = pts.astype(np.int32)
    for i in range(n - 1):
        cv2.line(out, tuple(pts_i[i]), tuple(pts_i[i + 1]), (0, 0, 0), thickness, cv2.LINE_AA)

def _tint_eye_red(out, lm, eye_idx, H, W, alpha_f):
    """Fill eye region with a red tint — only touches a small bounding-box crop."""
    pts = np.array([(int(lm[i].x * W), int(lm[i].y * H)) for i in eye_idx], dtype=np.int32)
    hull = cv2.convexHull(pts)

    # Tight bounding box so we only operate on a small patch
    x, y, bw, bh = cv2.boundingRect(hull)
    x  = max(0, x - 4);  y  = max(0, y - 4)
    x2 = min(W, x + bw + 8); y2 = min(H, y + bh + 8)
    if x2 <= x or y2 <= y:
        return

    # Local mask for the eye hull
    local_mask = np.zeros((y2 - y, x2 - x), dtype=np.uint8)
    shifted = hull - np.array([x, y])
    cv2.fillConvexPoly(local_mask, shifted, 255)

    # Blend red into the patch in float32 — tiny patch, very cheap
    patch = out[y:y2, x:x2].astype(np.float32)
    m     = (local_mask > 0)[:, :, None]          # boolean mask, no allocation loop
    patch[m[:,:,0], 2] = np.clip(patch[m[:,:,0], 2] * (1 - alpha_f) + 220 * alpha_f, 0, 255)
    patch[m[:,:,0], 1] = np.clip(patch[m[:,:,0], 1] * (1 - alpha_f), 0, 255)
    patch[m[:,:,0], 0] = np.clip(patch[m[:,:,0], 0] * (1 - alpha_f), 0, 255)
    out[y:y2, x:x2] = patch.astype(np.uint8)

def fx_angry_eyes(frame, state):
    import mediapipe as mp
    intensity = state.get('angryIntensity', 60) / 100.0
    lm_tool = _ensure_angry_landmarker()
    if lm_tool is None:
        return frame
    H, W = frame.shape[:2]
    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = lm_tool.detect(mp_img)
    out = frame.copy()
    alpha_f = intensity * 0.55
    for face_lm_list in result.face_landmarks:
        _tint_eye_red(out, face_lm_list, _LEFT_EYE_OUTLINE,  H, W, alpha_f)
        _tint_eye_red(out, face_lm_list, _RIGHT_EYE_OUTLINE, H, W, alpha_f)
        _draw_angry_brow(out, face_lm_list, _LEFT_BROW_IDX,  H, W, intensity, flip_anger=True)
        _draw_angry_brow(out, face_lm_list, _RIGHT_BROW_IDX, H, W, intensity, flip_anger=False)
    return out


# ── Sid Eyes — smooth thin-plate-spline face warp ─────────────────
# (Sid the sloth's wide-set stare.)  Earlier attempts tried to patch a
# small circular region around each eye (inpaint / blur / clone-stamp /
# radial-pull) — every version showed some visible seam or texture
# mismatch, because a small patch is a fundamentally different kind of
# content dropped next to real, detailed skin. A Delaunay triangle
# mesh (tried next) avoids the patch problem but is piecewise-linear —
# with few enough points to stay fast it shows visible flat facets.
# A thin-plate spline is the fix: a single globally C1-smooth surface
# fit through the control points (no facets, ever, regardless of point
# count), so moving the eye points pulls the whole nearby area
# smoothly with zero seam anywhere, by construction. It's expensive
# per-pixel, so it's restricted to a tight per-face crop, and an
# explicit ring of fixed points pinned to the crop's own edges keeps
# the boundary perfectly seamless against the untouched surrounding
# frame (without it, sparse landmarks alone don't reliably close the
# crop rectangle's corners, which shows up as black wedges).
_SID_EYE_L  = _LEFT_EYE_OUTLINE[::3]
_SID_EYE_R  = _RIGHT_EYE_OUTLINE[::3]
_SID_BROW_L = _LEFT_BROW_IDX[::3]
_SID_BROW_R = _RIGHT_BROW_IDX[::3]
_SID_ANCHOR_IDX = [168, 6, 4, 205, 425, 454, 234]   # nose bridge, cheeks, temples — fixed

def fx_sid_eyes(frame, state):
    import mediapipe as mp
    shift_px = state.get('sidShift', 15)
    lm_tool = _ensure_angry_landmarker()
    if lm_tool is None:
        return frame
    H, W = frame.shape[:2]
    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = lm_tool.detect(mp_img)
    out = frame.copy()

    all_idx = _SID_EYE_L + _SID_EYE_R + _SID_BROW_L + _SID_BROW_R + _SID_ANCHOR_IDX
    eye_set = set(_SID_EYE_L) | set(_SID_EYE_R)

    for lm in result.face_landmarks:
        mid_x = lm[168].x * W   # glabella (between the brows) — stable midline reference
        pts_full = np.array([(lm[i].x * W, lm[i].y * H) for i in all_idx], dtype=np.float32)

        leye_c = np.mean([(lm[i].x * W, lm[i].y * H) for i in _SID_EYE_L], axis=0)
        reye_c = np.mean([(lm[i].x * W, lm[i].y * H) for i in _SID_EYE_R], axis=0)
        l_dir = 1.0 if leye_c[0] >= mid_x else -1.0
        r_dir = 1.0 if reye_c[0] >= mid_x else -1.0
        l_shift = l_dir * min(shift_px, abs(leye_c[0] - mid_x) * 0.7)
        r_shift = r_dir * min(shift_px, abs(reye_c[0] - mid_x) * 0.7)

        dst_full = pts_full.copy()
        for j, idx in enumerate(all_idx):
            if idx in _SID_EYE_L or idx in _SID_BROW_L:
                dst_full[j, 0] += l_shift * (1.0 if idx in eye_set else 0.4)
            elif idx in _SID_EYE_R or idx in _SID_BROW_R:
                dst_full[j, 0] += r_shift * (1.0 if idx in eye_set else 0.4)

        # Tight crop — eyes/brows/nose/cheeks only, no mouth/jaw needed —
        # padded enough that the shifted points stay comfortably inside.
        # Kept snug (not just generously padded) so the crop doesn't
        # sweep in a lot of background near the temples/ears, which is
        # exactly where any residual warp curvature is most visible
        # (straight background edges — a chair, a wall — show bending
        # that skin texture would hide).
        max_shift = max(abs(l_shift), abs(r_shift))
        xs, ys = pts_full[:, 0], pts_full[:, 1]
        pad_x = max_shift + 15
        pad_y = 30
        cx0 = int(max(0, xs.min() - pad_x)); cx1 = int(min(W, xs.max() + pad_x))
        cy0 = int(max(0, ys.min() - pad_y)); cy1 = int(min(H, ys.max() + pad_y))
        if cx1 - cx0 < 10 or cy1 - cy0 < 10:
            continue
        cw, ch = cx1 - cx0, cy1 - cy0

        # A sparse ring only pins zero-displacement exactly at its own
        # points — between them the spline can still bow toward nearby
        # moving control points. Denser sampling on every edge (not just
        # top/bottom) keeps the whole boundary hugging identity, so
        # nothing bends visibly even where the crop meets background.
        n_h, n_v = 6, 5
        top    = [(cw * i / (n_h - 1), 0)  for i in range(n_h)]
        bottom = [(cw * i / (n_h - 1), ch) for i in range(n_h)]
        left   = [(0, ch * i / (n_v - 1))  for i in range(1, n_v - 1)]
        right  = [(cw, ch * i / (n_v - 1)) for i in range(1, n_v - 1)]
        ring = np.array(top + bottom + left + right, dtype=np.float32) + [cx0, cy0]

        src_pts = np.vstack([pts_full, ring]) - [cx0, cy0]
        dst_pts = np.vstack([dst_full, ring]) - [cx0, cy0]

        crop = out[cy0:cy1, cx0:cx1]
        tps = cv2.createThinPlateSplineShapeTransformer()
        matches = [cv2.DMatch(i, i, 0) for i in range(len(src_pts))]
        tps.estimateTransformation(dst_pts.reshape(1, -1, 2), src_pts.reshape(1, -1, 2), matches)
        warped = tps.warpImage(crop, borderMode=cv2.BORDER_REPLICATE)

        # The ring only pins displacement to ~zero at the crop edge, not
        # exactly zero everywhere along it — under a strong local lighting
        # gradient (e.g. a forehead highlight) even a sub-pixel mismatch
        # there reads as a visible line. Feathering the blend forces true
        # continuity at the boundary regardless of any residual warp error.
        feather = 18.0
        fx_ = np.arange(cw, dtype=np.float32)
        fy_ = np.arange(ch, dtype=np.float32)
        FX, FY = np.meshgrid(fx_, fy_)
        edge_dist = np.minimum.reduce([FX, cw - 1 - FX, FY, ch - 1 - FY])
        alpha = np.clip(edge_dist / feather, 0.0, 1.0)[:, :, None]
        out[cy0:cy1, cx0:cx1] = np.clip(
            warped.astype(np.float32) * alpha + crop.astype(np.float32) * (1 - alpha),
            0, 255).astype(np.uint8)

    return out

# ── Hamster Eyes 2 — TPS eye enlargement (same technique as Sid Eyes) ─
# The original Hamster Eyes ("_magnify_eye") does a constant-factor
# "spherical bulge" remap in a circle: a uniform zoom that snaps to a
# hard identity boundary at the circle's edge. At high scale with
# close-set eyes, the two circles can overlap, and the second eye's
# rectangular paste cuts a straight seam through whatever the first
# eye already bulged — that's the "lines coming out of center" this
# rebuild targets. Same fix as Sid Eyes: each eye's full outline is
# control points expanded radially from its own centre via a thin-
# plate spline, brows/nose/cheeks stay fixed as anchors, and a dense
# zero-displacement ring pinned to the crop's own edges keeps the
# boundary seamless.
#
# Width is hard-capped at 150% regardless of the slider — past that,
# extra slider range goes entirely into height, so eyes get taller/
# rounder rather than wider. The vertical boost is intentionally
# conservative (capped at 2.0x): a more aggressive boost was tried and
# reintroduced a fold-over artifact on a genuinely wide-open eye — TPS
# displacement that's too large relative to the local point spacing
# folds the mapping over itself, pulling the wrong source content into
# the middle of the pupil, and the bending-energy strain from that
# near-degenerate solution rippled out to the nose too (one global
# smooth surface, not independent per-eye patches).
def fx_hamster_eyes2(frame, state):
    import mediapipe as mp
    scale = state.get('hamster2Scale', 150) / 100.0
    lm_tool = _ensure_hamster_landmarker()
    if lm_tool is None:
        return frame
    H, W = frame.shape[:2]
    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = lm_tool.detect(mp_img)
    out = frame.copy()

    all_idx = _LEFT_EYE_IDX + _RIGHT_EYE_IDX + _SID_BROW_L + _SID_BROW_R + _SID_ANCHOR_IDX

    for lm in result.face_landmarks:
        mid_x = lm[168].x * W
        pts_full = np.array([(lm[i].x * W, lm[i].y * H) for i in all_idx], dtype=np.float32)

        leye_pts = np.array([(lm[i].x * W, lm[i].y * H) for i in _LEFT_EYE_IDX], dtype=np.float32)
        reye_pts = np.array([(lm[i].x * W, lm[i].y * H) for i in _RIGHT_EYE_IDX], dtype=np.float32)

        # During a blink the 16 eye-outline points collapse toward a
        # nearly flat line (upper/lower eyelid converge), which can make
        # the TPS solver's system nearly singular — and since TPS is one
        # global smooth surface, a degenerate solve doesn't just glitch
        # the eye, it can throw chaotic values across the whole warped
        # crop. Skip this face entirely for a frame where either eye
        # looks closed/closing rather than feed the solver a degenerate
        # point configuration.
        leye_h = np.ptp(leye_pts[:, 1]); leye_w = np.ptp(leye_pts[:, 0])
        reye_h = np.ptp(reye_pts[:, 1]); reye_w = np.ptp(reye_pts[:, 0])
        if leye_h < max(3.0, leye_w * 0.12) or reye_h < max(3.0, reye_w * 0.12):
            continue

        leye_c, reye_c = leye_pts.mean(axis=0), reye_pts.mean(axis=0)
        leye_r = max(leye_w, leye_h) / 2.0
        reye_r = max(reye_w, reye_h) / 2.0

        # Keep the enlarged eye's inward edge clear of the midline —
        # otherwise the two eyes' crops overlap and the second eye's
        # paste cuts a seam through the first. This clamp must never go
        # below 1.0 (natural size): for closer-set eyes, a turned head,
        # or just being nearer the camera, dist-to-midline can be smaller
        # than eye_r, which without this floor turned a "don't grow too
        # much" safety ceiling into an accidental shrink — small dark
        # circles instead of enlarged eyes.
        l_max = max(1.0, 0.9 * abs(leye_c[0] - mid_x) / max(leye_r, 1.0))
        r_max = max(1.0, 0.9 * abs(reye_c[0] - mid_x) / max(reye_r, 1.0))
        l_scale, r_scale = min(scale, l_max, 1.5), min(scale, r_max, 1.5)

        # A human eye is naturally much wider than tall, so scaling both
        # axes by the same factor keeps that same flat, elongated
        # proportion — it gets wider without reading as rounder. Boosting
        # the vertical multiplier specifically (and shrinking that boost
        # as the eye's own natural aspect ratio already gets rounder, so
        # an already wide-open eye — with less headroom before folding —
        # doesn't get pushed as hard) is what makes it look round/bulgy.
        l_aspect = np.ptp(leye_pts[:, 1]) / max(np.ptp(leye_pts[:, 0]), 1.0)
        r_aspect = np.ptp(reye_pts[:, 1]) / max(np.ptp(reye_pts[:, 0]), 1.0)
        l_boost = np.clip(1.5 - l_aspect, 1.05, 1.35)
        r_boost = np.clip(1.5 - r_aspect, 1.05, 1.35)
        l_scale_y = min(l_scale * l_boost, 2.0)
        r_scale_y = min(r_scale * r_boost, 2.0)

        dst_full = pts_full.copy()
        for j, idx in enumerate(all_idx):
            if idx in _LEFT_EYE_IDX:
                off = pts_full[j] - leye_c
                dst_full[j] = leye_c + [off[0] * l_scale, off[1] * l_scale_y]
            elif idx in _RIGHT_EYE_IDX:
                off = pts_full[j] - reye_c
                dst_full[j] = reye_c + [off[0] * r_scale, off[1] * r_scale_y]
            # brows/anchors: left unchanged (fixed)

        xs = np.concatenate([pts_full[:, 0], dst_full[:, 0]])
        ys = np.concatenate([pts_full[:, 1], dst_full[:, 1]])
        pad = 20
        cx0 = int(max(0, xs.min() - pad)); cx1 = int(min(W, xs.max() + pad))
        cy0 = int(max(0, ys.min() - pad)); cy1 = int(min(H, ys.max() + pad))
        if cx1 - cx0 < 10 or cy1 - cy0 < 10:
            continue
        cw, ch = cx1 - cx0, cy1 - cy0

        n_h, n_v = 6, 5
        top    = [(cw * i / (n_h - 1), 0)  for i in range(n_h)]
        bottom = [(cw * i / (n_h - 1), ch) for i in range(n_h)]
        left   = [(0, ch * i / (n_v - 1))  for i in range(1, n_v - 1)]
        right  = [(cw, ch * i / (n_v - 1)) for i in range(1, n_v - 1)]
        ring = np.array(top + bottom + left + right, dtype=np.float32) + [cx0, cy0]

        src_pts = np.vstack([pts_full, ring]) - [cx0, cy0]
        dst_pts = np.vstack([dst_full, ring]) - [cx0, cy0]

        crop = out[cy0:cy1, cx0:cx1]
        tps = cv2.createThinPlateSplineShapeTransformer()
        matches = [cv2.DMatch(i, i, 0) for i in range(len(src_pts))]
        tps.estimateTransformation(dst_pts.reshape(1, -1, 2), src_pts.reshape(1, -1, 2), matches)
        warped = tps.warpImage(crop, borderMode=cv2.BORDER_REPLICATE)

        feather = 18.0
        fx_ = np.arange(cw, dtype=np.float32)
        fy_ = np.arange(ch, dtype=np.float32)
        FX, FY = np.meshgrid(fx_, fy_)
        edge_dist = np.minimum.reduce([FX, cw - 1 - FX, FY, ch - 1 - FY])
        alpha = np.clip(edge_dist / feather, 0.0, 1.0)[:, :, None]
        out[cy0:cy1, cx0:cx1] = np.clip(
            warped.astype(np.float32) * alpha + crop.astype(np.float32) * (1 - alpha),
            0, 255).astype(np.uint8)

    return out

# ── Pretty Pretty — glamour makeup effect ─────────────────────────
def _pp_lm_pt(lm, idx, W, H):
    """Return (x, y) pixel coords for landmark index idx."""
    p = lm[idx]
    return (int(p.x * W), int(p.y * H))

def fx_so_pretty(frame, state):
    import mediapipe as mp
    lm_tool = _ensure_angry_landmarker()
    if lm_tool is None:
        return frame
    H, W = frame.shape[:2]
    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = lm_tool.detect(mp_img)
    if not result.face_landmarks:
        return frame
    out = frame.copy()
    for face_lm_list in result.face_landmarks:
        lm = face_lm_list
        face_w = abs(_pp_lm_pt(lm, 454, W, H)[0] - _pp_lm_pt(lm, 234, W, H)[0])
        # Blush: average cheek landmark clusters for a stable anchor point
        lpts_c = [_pp_lm_pt(lm, i, W, H) for i in [36, 47, 50, 205, 187]]
        rpts_c = [_pp_lm_pt(lm, i, W, H) for i in [266, 277, 280, 425, 411]]
        lc = (sum(p[0] for p in lpts_c)//len(lpts_c), sum(p[1] for p in lpts_c)//len(lpts_c))
        rc = (sum(p[0] for p in rpts_c)//len(rpts_c), sum(p[1] for p in rpts_c)//len(rpts_c))
        # Soft blush: per-cheek patch with radial distance falloff
        r_draw = max(10, face_w // 8)
        for cx_b, cy_b in [lc, rc]:
            x0b = max(0, cx_b - r_draw)
            x1b = min(W, cx_b + r_draw)
            y0b = max(0, cy_b - r_draw)
            y1b = min(H, cy_b + r_draw)
            if x1b <= x0b or y1b <= y0b:
                continue
            gx = np.arange(x0b, x1b, dtype=np.float32) - cx_b
            gy = np.arange(y0b, y1b, dtype=np.float32) - cy_b
            GX, GY = np.meshgrid(gx, gy)
            dist = np.sqrt(GX*GX + GY*GY)
            alpha_patch = np.clip(1.0 - dist / r_draw, 0, 1) ** 1.5 * 0.55
            blush_color = np.array([80.0, 40.0, 220.0])
            patch = out[y0b:y1b, x0b:x1b].astype(np.float32)
            out[y0b:y1b, x0b:x1b] = np.clip(
                patch * (1 - alpha_patch[:,:,None]) + blush_color * alpha_patch[:,:,None],
                0, 255).astype(np.uint8)
        # Upper eyelid line
        liner_w = max(2, face_w // 50)
        for eye_idx in [[33,246,161,160,159,158,157,173,133],
                        [362,398,384,385,386,387,388,466,263]]:
            pts = np.array([_pp_lm_pt(lm, i, W, H) for i in eye_idx], np.int32)
            cv2.polylines(out, [pts], False, (20, 20, 20), liner_w, cv2.LINE_AA)
        # Upper + lower lips
        upper_lip = [61,185,40,39,37,0,267,269,270,409,291,308,415,310,311,312,13,82,81,80,191,78]
        lower_lip = [61,146,91,181,84,17,314,405,321,375,291,308,324,318,402,317,14,87,178,88,95,78]
        for lip_idx in [upper_lip, lower_lip]:
            lpts = np.array([_pp_lm_pt(lm, i, W, H) for i in lip_idx], np.int32)
            cv2.fillPoly(out, [lpts], (30, 30, 210))
        # Lip gloss highlight
        lc2 = _pp_lm_pt(lm, 0, W, H)
        gl_w = max(4, face_w // 12)
        cv2.ellipse(out, lc2, (gl_w, max(2, gl_w//3)), 0, 180, 360, (100, 100, 240), -1, cv2.LINE_AA)
    return out

# ── Vampire Teeth — fangs anchored to the upper lip ──────────────
def _fang_mask(root, tip, bend_dir, base_w, shape):
    """Curved, tapered fang polygon (root -> tip) rendered as a soft 0..1 mask."""
    t = np.linspace(0.0, 1.0, 14, dtype=np.float32)[:, None]
    axis = tip - root
    length = float(np.linalg.norm(axis)) + 1e-6
    normal = np.array([-axis[1], axis[0]], dtype=np.float32) / length
    # centerline bows slightly toward bend_dir, width tapers to a sharp point
    center = root + axis * t + bend_dir * (length * 0.10) * np.sin(np.pi * t)
    half_w = base_w * 0.5 * (1.0 - t) ** 0.75
    left, right = center + normal * half_w, center - normal * half_w
    poly = np.rint(np.vstack([left, right[::-1]]) * 16).astype(np.int32)
    mask = np.zeros(shape, np.uint8)
    cv2.fillPoly(mask, [poly], 255, cv2.LINE_AA, shift=4)
    return mask.astype(np.float32) / 255.0

_vampire_landmarker = None
_vampire_failed     = False
_vampire_ts_ms      = 0
_vampire_smooth     = []     # per-face smoothed mouth points, matched frame to frame
_VAMP_MOUTH_IDX = [61, 291, 13, 14, 78, 191, 80, 81, 82, 312, 311, 310, 415, 308,
                   324, 318, 402, 317, 87, 178, 88, 95]
_VAMP_EYES = [  # (outline indices, outer-corner landmark, iris center, iris ring)
    (_LEFT_EYE_OUTLINE,  263, 473, [474, 475, 476, 477]),
    (_RIGHT_EYE_OUTLINE, 33,  468, [469, 470, 471, 472]),
]
_VAMP_MOUTH_IDX = _VAMP_MOUTH_IDX + [234, 454, 152, 10] + sorted({i for o, _, c, r in _VAMP_EYES for i in o + [c] + r})

def _ensure_vampire_landmarker():
    """Video-mode landmarker: tracks faces across frames instead of re-detecting each one."""
    global _vampire_landmarker, _vampire_failed
    if _vampire_landmarker is not None or _vampire_failed:
        return _vampire_landmarker
    if _ensure_angry_landmarker() is None:     # makes sure the model file exists
        _vampire_failed = True
        return None
    try:
        from mediapipe.tasks import python as _mpt
        from mediapipe.tasks.python import vision as _mpv
        _vampire_landmarker = _mpv.FaceLandmarker.create_from_options(_mpv.FaceLandmarkerOptions(
            base_options=_mpt.BaseOptions(model_asset_path=_get_model_path()),
            running_mode=_mpv.RunningMode.VIDEO,
            num_faces=4,
            min_face_detection_confidence=0.4,
            min_face_presence_confidence=0.4,
            min_tracking_confidence=0.4,
        ))
    except Exception as e:
        print(f'[VampireTeeth] Failed to load FaceLandmarker: {e}')
        _vampire_failed = True
    return _vampire_landmarker

def _vampire_mouth_points(face_lms, W, H):
    """Mouth landmarks as {idx: (x, y)} smoothed over time; faster motion = less smoothing.
    The depth of every landmark rides along under the 'z' key ({idx: z}, in pixel units)."""
    global _vampire_smooth
    raw_all = [np.array([[lm[i].x * W, lm[i].y * H, lm[i].z * W] for i in _VAMP_MOUTH_IDX], np.float32)
               for lm in face_lms]
    old, new = _vampire_smooth, []
    for raw in raw_all:
        center = raw[:2, :2].mean(0)   # mouth corners
        prev = min(old, key=lambda o: np.linalg.norm(o[:2, :2].mean(0) - center), default=None)
        mouth_w = float(np.linalg.norm(raw[0, :2] - raw[1, :2])) + 1e-6
        if prev is not None and np.linalg.norm(prev[:2, :2].mean(0) - center) < mouth_w * 1.5:
            speed = float(np.linalg.norm(raw[:, :2].mean(0) - prev[:, :2].mean(0))) / mouth_w
            a = float(np.clip(0.30 + speed * 4.0, 0.30, 0.95))     # still: steady, moving: responsive
            raw = prev + (raw - prev) * a
        new.append(raw)
    _vampire_smooth = new
    out = []
    for pts in new:
        d = {i: pts[k, :2] for k, i in enumerate(_VAMP_MOUTH_IDX)}
        d['z'] = {i: float(pts[k, 2]) for k, i in enumerate(_VAMP_MOUTH_IDX)}
        out.append(d)
    return out

def _bezier(p0, p1, p2, n=10):
    t = np.linspace(0, 1, n, dtype=np.float32)[:, None]
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2

# Bat wing outline in local units (x outward from the base, y up): arm, wrist, 3 finger tips, body
_WING_ARM = [(0.0, 0.0), (0.7, 1.7), (2.2, 2.0)]               # base -> elbow -> wrist
_WING_TIPS = [(3.6, 1.2), (3.2, -0.2), (2.2, -0.9)]            # finger tips
_WING_BODY = (0.3, -0.7)                                       # where the membrane meets the head

def _vampire_wings(out, pt, size, W, H):
    """Small bat wings perched on each side of the top of the head."""
    left, right, chin, top = pt[234], pt[454], pt[152], pt[10]
    fw = float(np.linalg.norm(right - left))
    if fw < 20:
        return
    # Head frame in 3D so the wings ride the skull as it turns: ex across the head,
    # ey down the face, ef out through the forehead (toward the camera).
    z = pt['z']
    P = lambda i: np.array([pt[i][0], pt[i][1], z[i]], np.float32)
    l3, r3, t3, c3 = P(234), P(454), P(10), P(152)
    ex = r3 - l3
    fw3 = float(np.linalg.norm(ex))
    ex /= fw3
    ey = c3 - t3
    ey -= ex * float(ey @ ex)
    face_h = float(np.linalg.norm(ey))
    ey /= face_h + 1e-6
    ef = np.cross(ex, ey)
    if ef[2] > 0:                                     # MediaPipe: smaller z = closer to the camera
        ef = -ef
    xy = ex[:2]
    xy_len = float(np.linalg.norm(xy))
    if xy_len < 0.2:                                  # profile: no reliable sideways axis
        return
    u = (xy / xy_len).astype(np.float32)
    down = np.array([-u[1], u[0]], np.float32)
    cos_yaw = min(1.0, xy_len)
    unit = fw3 * 0.15 * size * (0.55 + 0.45 * cos_yaw)
    # Elliptical skull cross-section at the wing height (deeper than wide): wings sit
    # ~50 deg either side of the forehead, a little inside the silhouette
    ra, rb = fw3 * 0.46, fw3 * 0.70
    center = t3 - ef * rb * 0.9 + ey * face_h * 0.06
    phi = math.radians(50)
    tilt = 0.45                                       # rad: wings lean up and out
    ct, st = math.cos(tilt), math.sin(tilt)

    membrane_l = np.zeros((H, W), np.uint8)
    edge_l = np.zeros((H, W), np.uint8)
    bone_l = np.zeros((H, W), np.uint8)
    all_pts = []
    for side in (1, -1):
        base = (center + ex * side * ra * math.sin(phi) + ef * rb * math.cos(phi))[:2].astype(np.float32)
        def to_img(p):
            x, y = p[0] * ct - p[1] * st, p[0] * st + p[1] * ct
            return base + side * u * x * unit - down * y * unit
        arm = [np.array(to_img(q), np.float32) for q in _WING_ARM]
        tips = [np.array(to_img(q), np.float32) for q in _WING_TIPS]
        body = np.array(to_img(_WING_BODY), np.float32)
        centroid = (arm[1] + tips[1] + body) / 3

        trail = [tips[0]]                             # scalloped trailing edge: tip -> tip -> tip -> body
        chain = tips + [body]
        for a_, b_ in zip(chain, chain[1:]):
            mid = (a_ + b_) / 2
            trail += list(_bezier(a_, mid + (centroid - mid) * 0.45, b_, 9)[1:])
        arm_curve = list(_bezier(arm[0], arm[1], arm[2], 12))
        outline = arm_curve + [tips[0]] + trail[1:] + [arm[0]]
        cv2.fillPoly(membrane_l, [np.rint(np.array(outline, np.float32) * 16).astype(np.int32)],
                     255, cv2.LINE_AA, shift=4)
        edge = np.rint(np.array(arm_curve + [tips[0]] + trail[1:], np.float32) * 16).astype(np.int32)
        cv2.polylines(edge_l, [edge], False, 255, max(1, int(unit * 0.10)), cv2.LINE_AA, shift=4)
        for tip in tips + [body]:                     # finger bones from the wrist
            cv2.line(bone_l, tuple(np.rint(arm[2]).astype(int)), tuple(np.rint(tip).astype(int)),
                     255, max(1, int(unit * 0.12)), cv2.LINE_AA)
        all_pts += outline

    pts = np.array(all_pts)
    pad = int(unit * 1.2) + 4
    x0, y0 = max(0, int(pts[:, 0].min()) - pad), max(0, int(pts[:, 1].min()) - pad)
    x1, y1 = min(W, int(pts[:, 0].max()) + pad), min(H, int(pts[:, 1].max()) + pad)
    if x1 <= x0 or y1 <= y0:
        return
    sl = (slice(y0, y1), slice(x0, x1))
    f = lambda m, sg: cv2.GaussianBlur(m[sl], (0, 0), sg).astype(np.float32)[..., None] / 255.0
    wa, bones = f(membrane_l, 0.7), f(bone_l, 0.7)
    glow, core = f(edge_l, max(1.5, unit * 0.25)), f(edge_l, 0.7)

    roi = out[sl].astype(np.float32)
    dark = np.array([34, 24, 40], np.float32)         # glossy near-black with a violet cast
    membrane = dark * (1 - bones) + np.array([12, 8, 14], np.float32) * bones
    gloss = np.clip(bones * 0.0 + f(edge_l, 1.2) * 0.35, 0, 1)
    membrane = membrane + gloss * np.array([90, 80, 110], np.float32)
    roi = roi * (1 - wa * 0.96) + membrane * wa * 0.96
    layer = np.clip(glow * 1.2, 0, 1) * np.array([40, 30, 255], np.float32) * 0.8 + core * np.array([120, 110, 255], np.float32) * 0.5
    roi = 255 - (255 - roi) * (255 - np.clip(layer, 0, 255)) / 255
    out[sl] = np.clip(roi, 0, 255).astype(np.uint8)

def _vampire_eyes(out, pt, hue, W, H):
    """Neon almond outline (pointed outer corner) around each eye + recolored iris."""
    glow_bgr = np.array([40, 30, 255], np.float32)     # outline is always neon red
    core_bgr = np.array([200, 190, 255], np.float32)
    for outline, outer_idx, iris_c, iris_ring in _VAMP_EYES:
        pts = np.array([pt[i] for i in outline], np.float32)
        c = pts.mean(0)
        outer = pt[outer_idx]
        axis = outer - c
        eye_len = float(np.linalg.norm(axis)) * 2
        if eye_len < 6:
            continue
        u = axis / (eye_len / 2)
        v = np.array([-u[1], u[0]], np.float32)
        if v[1] > 0:           # make v point up the screen
            v = -v

        # ── iris recolor: hue swap, boosted saturation, highlights & pupil kept ──
        ic = pt[iris_c]
        radius = float(np.mean([np.linalg.norm(pt[i] - ic) for i in iris_ring])) * 1.05
        r = int(radius) + 3
        x0, y0 = max(0, int(ic[0]) - r), max(0, int(ic[1]) - r)
        x1, y1 = min(W, int(ic[0]) + r + 1), min(H, int(ic[1]) + r + 1)
        if x1 > x0 and y1 > y0:
            roi = out[y0:y1, x0:x1]
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            hsv[..., 0] = hue
            hsv[..., 1] = np.maximum(hsv[..., 1], 200)
            hsv[..., 2] = np.clip(hsv[..., 2].astype(np.float32) * 1.25 + 25, 0, 255).astype(np.uint8)
            recol = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            m = np.zeros(roi.shape[:2], np.uint8)
            # keep the eyelids out: clip the iris disc to the eye opening
            cv2.circle(m, (int(ic[0]) - x0, int(ic[1]) - y0), int(radius), 255, -1, cv2.LINE_AA)
            op = np.zeros(roi.shape[:2], np.uint8)
            cv2.fillPoly(op, [np.rint(pts - [x0, y0]).astype(np.int32)], 255)
            m = cv2.GaussianBlur(cv2.min(m, op), (0, 0), 1.2).astype(np.float32)[..., None] / 255.0 * 0.85
            out[y0:y1, x0:x1] = (recol * m + roi * (1 - m)).astype(np.uint8)

        # ── neon outline: stretched almond, raised lid, sharp outer tip ──
        rel = pts - c
        a, b = rel @ u, rel @ v
        b = np.where(b > 0, b * 2.8, b * 2.0) + eye_len * 0.05
        big = c + np.outer(a * 1.4, u) + np.outer(b, v)
        tip = c + u * eye_len * 1.5 + v * eye_len * 0.04
        k = int(np.argmin(np.linalg.norm(pts - outer, axis=1)))
        big = np.insert(big, k + 1, tip, axis=0)
        big = np.delete(big, k, axis=0)
        for _ in range(2):             # Chaikin corner-cutting: smooth the lid curves
            nxt = np.roll(big, -1, axis=0)
            smooth = np.empty((len(big) * 2, 2), np.float32)
            smooth[0::2] = 0.75 * big + 0.25 * nxt
            smooth[1::2] = 0.25 * big + 0.75 * nxt
            big = smooth

        m = int(eye_len * 0.9)
        bx0, by0 = max(0, int(big[:, 0].min()) - m), max(0, int(big[:, 1].min()) - m)
        bx1, by1 = min(W, int(big[:, 0].max()) + m), min(H, int(big[:, 1].max()) + m)
        if bx1 <= bx0 or by1 <= by0:
            continue
        poly = np.rint((big - [bx0, by0]) * 16).astype(np.int32)
        h, w = by1 - by0, bx1 - bx0
        line_w = max(3, int(eye_len * 0.09))
        wide = np.zeros((h, w), np.uint8)
        cv2.polylines(wide, [poly], True, 255, line_w * 2, cv2.LINE_AA, shift=4)
        core = np.zeros((h, w), np.uint8)
        cv2.polylines(core, [poly], True, 255, max(1, line_w // 2), cv2.LINE_AA, shift=4)
        sigma = max(2.0, eye_len * 0.08)
        glow = cv2.GaussianBlur(wide, (0, 0), sigma).astype(np.float32)[..., None] / 255.0
        core = cv2.GaussianBlur(core, (0, 0), 0.8).astype(np.float32)[..., None] / 255.0
        layer = np.clip(glow * 2.2, 0, 1) * glow_bgr + core * core_bgr   # additive light
        roi = out[by0:by1, bx0:bx1].astype(np.float32)
        out[by0:by1, bx0:bx1] = np.clip(255 - (255 - roi) * (255 - np.clip(layer, 0, 255)) / 255, 0, 255).astype(np.uint8)

def fx_vampire_teeth(frame, state):
    import mediapipe as mp
    global _vampire_ts_ms, _vampire_smooth
    lm_tool = _ensure_vampire_landmarker()
    if lm_tool is None:
        return frame
    H, W = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    _vampire_ts_ms += 33                       # video mode needs strictly increasing timestamps
    result = lm_tool.detect_for_video(mp_img, _vampire_ts_ms)
    if not result.face_landmarks:
        _vampire_smooth = []
        return frame
    mouths = _vampire_mouth_points(result.face_landmarks, W, H)

    out = frame.copy()
    scale = np.clip(state.get('vampireSize', 100) / 100.0, 0.6, 1.5)
    upper_inner_idx = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
    lower_inner_idx = [308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78]

    hue = int(np.clip(state.get('vampireEyeHue', 0), 0, 179))
    wing_size = state.get('vampireWings', 100) / 100.0
    for mouth in mouths:
        point = lambda idx: mouth[idx]
        if wing_size > 0.05:
            _vampire_wings(out, mouth, wing_size, W, H)
        _vampire_eyes(out, mouth, hue, W, H)
        mouth_left, mouth_right = point(61), point(291)
        mouth_width = float(np.linalg.norm(mouth_right - mouth_left))
        upper_center, lower_center = point(13), point(14)
        opening = float(np.linalg.norm(lower_center - upper_center))
        if mouth_width < 8 or opening < max(4, mouth_width * 0.045):
            continue

        opening_dir = (lower_center - upper_center) / opening
        mouth_axis = (mouth_right - mouth_left) / mouth_width
        inner_upper = np.array([point(idx) for idx in upper_inner_idx])
        inner_lower = np.array([point(idx) for idx in lower_inner_idx])
        segment_lengths = np.linalg.norm(np.diff(inner_upper, axis=0), axis=1)
        path_lengths = np.concatenate(([0.0], np.cumsum(segment_lengths)))
        path_length = float(path_lengths[-1])
        if path_length < 1:
            continue

        # Fangs are clipped to the visible mouth interior so they never paint over the lips
        inside = np.zeros((H, W), np.uint8)
        cv2.fillPoly(inside, [np.rint(np.vstack([inner_upper, inner_lower])).astype(np.int32)], 255)
        inside = cv2.GaussianBlur(inside, (0, 0), 1.0).astype(np.float32) / 255.0

        fang_mask = np.zeros((H, W), np.float32)
        shade = np.zeros((H, W), np.float32)    # 0 = cool shadow, 1 = bright highlight
        for fraction in (0.25, 0.75):   # 4 center teeth span 0.30-0.70; fangs sit just outside them
            target = path_length * fraction
            segment = min(np.searchsorted(path_lengths, target, side='right') - 1,
                          len(segment_lengths) - 1)
            segment = max(0, segment)
            segment_fraction = ((target - path_lengths[segment]) /
                                max(segment_lengths[segment], 1e-6))
            root = inner_upper[segment] + segment_fraction * (
                inner_upper[segment + 1] - inner_upper[segment])

            fang_width = mouth_width * 0.095 * scale
            fang_height = min(opening * 0.88 * scale, mouth_width * 0.24)
            inward = -mouth_axis if fraction < 0.5 else mouth_axis
            tip = root + opening_dir * fang_height + inward * fang_width * 0.12
            root = root - opening_dir * fang_height * 0.06   # tuck the base under the lip
            m = _fang_mask(root, tip, inward, fang_width, (H, W))
            fang_mask = np.maximum(fang_mask, m)

            # Shading: bright on the outer edge, cooler toward the inner edge and tip
            ys, xs = np.nonzero(m > 0.01)
            if len(xs):
                pts = np.stack([xs, ys], 1).astype(np.float32)
                across = (pts - root) @ mouth_axis / (fang_width * 0.5 + 1e-6)
                along = np.clip((pts - root) @ opening_dir / (fang_height + 1e-6), 0, 1)
                side = -across if fraction < 0.5 else across
                g = 0.75 - 0.30 * along + 0.25 * np.clip(side, -1, 1)
                shade[ys, xs] = np.maximum(shade[ys, xs], g)

        if fang_mask.max() <= 0:
            continue

        # Soft glow around the fangs (cool white, as in the reference look)
        glow = cv2.GaussianBlur(fang_mask, (0, 0), max(2.0, mouth_width * 0.03)) * inside
        glow_col = np.array([255, 235, 215], np.float32)           # BGR, bluish-white
        out_f = out.astype(np.float32)
        out_f = out_f + glow[..., None] * glow_col * 0.45

        # Fang body: ivory-white with subtle cool shading
        lit = np.array([255, 252, 248], np.float32)
        dark = np.array([215, 205, 195], np.float32)
        body = dark + (lit - dark) * np.clip(shade, 0, 1)[..., None]
        alpha = (fang_mask * inside)[..., None]
        out_f = out_f * (1 - alpha) + body * alpha
        out = np.clip(out_f, 0, 255).astype(np.uint8)

    return out

# ── Zombie — face-landmark makeup ────────────────
_FACE_OVAL_IDX = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378,
                  400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21,
                  54, 103, 67, 109]
_MONSTER_LIP_LOWER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317,
                      14, 87, 178, 88, 95, 78]

def _face_points(lm, W, H):
    return np.array([[p.x * W, p.y * H] for p in lm], np.float32)

def _monster_faces(frame):
    """Per detected face: (x0, y0, x1, y1, pts_local, face_width, z). Caller edits frame[y0:y1, x0:x1]."""
    import mediapipe as mp
    lm_tool = _ensure_angry_landmarker()
    if lm_tool is None:
        return []
    H, W = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = lm_tool.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    faces = []
    for lm in result.face_landmarks or []:
        pts = _face_points(lm, W, H)
        fw = float(np.linalg.norm(pts[454] - pts[234]))
        if fw < 30:
            continue
        x0, x1 = int(max(0, pts[:, 0].min() - fw * 0.7)), int(min(W, pts[:, 0].max() + fw * 0.7))
        y0, y1 = int(max(0, pts[:, 1].min() - fw * 1.1)), int(min(H, pts[:, 1].max() + fw * 0.7))
        if x1 - x0 < 8 or y1 - y0 < 8:
            continue
        z = np.array([p.z * W for p in lm], np.float32)
        faces.append((x0, y0, x1, y1, pts - np.array([x0, y0], np.float32), fw, z))
    return faces

def _axes(pts):
    u = pts[454] - pts[234]
    u = u / (np.linalg.norm(u) + 1e-6)
    down = np.array([-u[1], u[0]], np.float32)
    return u, down

def _poly_mask(shape, poly, blur=0.0, ss=4):
    m = np.zeros(shape, np.uint8)
    cv2.fillPoly(m, [np.rint(np.asarray(poly, np.float32) * (1 << ss)).astype(np.int32)], 255,
                 cv2.LINE_AA, shift=ss)
    m = m.astype(np.float32) / 255.0
    return cv2.GaussianBlur(m, (0, 0), blur) if blur > 0 else m

def _blend(img, color, alpha):
    """img (float32 HxWx3) <- lerp toward color ((3,) or HxWx3) by alpha (HxW).
    Only the bounding box where alpha is non-zero is touched, so small overlays stay cheap."""
    rows = np.flatnonzero(alpha.max(axis=1) > 0.004)
    if rows.size == 0:
        return
    r0, r1 = rows[0], rows[-1] + 1
    cols = np.flatnonzero(alpha[r0:r1].max(axis=0) > 0.004)
    c0, c1 = cols[0], cols[-1] + 1
    a = alpha[r0:r1, c0:c1, None]
    dst = img[r0:r1, c0:c1]
    col = color[r0:r1, c0:c1] if isinstance(color, np.ndarray) and color.ndim == 3 else np.asarray(color, np.float32)
    dst *= (1 - a)
    dst += col * a

_FOREHEAD_LIFT = {10: 0.13, 338: 0.11, 109: 0.11, 297: 0.09, 67: 0.09, 332: 0.06, 103: 0.06, 284: 0.03, 54: 0.03}

def _oval_mask(pts, shape, fw):
    """Face oval, lifted over the forehead (the landmark oval stops at mid-forehead)."""
    _, down = _axes(pts)
    oval = pts[_FACE_OVAL_IDX].copy()
    for k, idx in enumerate(_FACE_OVAL_IDX):
        if idx in _FOREHEAD_LIFT:
            oval[k] -= down * fw * _FOREHEAD_LIFT[idx]
    return _poly_mask(shape, oval, max(1.0, fw * 0.02))

def _tint_skin(img, mask, amount, tint, contrast=1.0):
    """Recolour skin: grayscale * tint (BGR 0-255), blended by mask * amount."""
    g = cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    g = np.clip((g - 0.5) * contrast + 0.5, 0, 1)
    _blend(img, g[..., None] * np.asarray(tint, np.float32), mask * amount)

def _soft_ellipse(shape, center, axes, angle, blur):
    m = np.zeros(shape, np.uint8)
    cv2.ellipse(m, tuple(np.rint(center).astype(int)), (max(1, int(axes[0])), max(1, int(axes[1]))),
                angle, 0, 360, 255, -1, cv2.LINE_AA)
    return cv2.GaussianBlur(m.astype(np.float32) / 255.0, (0, 0), max(0.5, blur))

_zombie_tex = None

def _get_zombie_tex():
    """Canonical 512x512 face-space decay overlay (BGR colour, alpha). Eyes at y=190, chin at y=450."""
    global _zombie_tex
    if _zombie_tex is None:
        rng = np.random.RandomState(7)
        n = 512
        def field(sig):
            f = cv2.GaussianBlur(rng.rand(n, n).astype(np.float32), (0, 0), sig)
            return (f - f.min()) / (f.max() - f.min() + 1e-6)
        blot = np.clip((field(22) * 0.65 + field(7) * 0.35 - 0.55) * 4.0, 0, 1)      # rot blotches
        bruise = np.clip((field(30) - 0.62) * 5.0, 0, 1)
        col = np.zeros((n, n, 3), np.float32)
        col[:] = (35, 70, 60)                                                          # dark green-brown
        col = col * (1 - bruise[..., None]) + np.array([95, 55, 85], np.float32) * bruise[..., None]
        alpha = np.clip(blot * 1.0 + bruise * 0.55, 0, 0.9)
        for cx, cy, r in [(115, 300, 42), (345, 125, 34), (400, 350, 30)]:             # open wounds
            ang = np.linspace(0, 2 * np.pi, 24, endpoint=False)
            rad = r * (0.75 + 0.5 * rng.rand(24))
            rim = _poly_mask((n, n), np.stack([cx + rad * np.cos(ang), cy + rad * np.sin(ang)], 1), 3.0)
            core = _poly_mask((n, n), np.stack([cx + rad * 0.6 * np.cos(ang), cy + rad * 0.6 * np.sin(ang)], 1), 2.5)
            col = col * (1 - rim[..., None]) + np.array([125, 140, 175], np.float32) * rim[..., None]
            col = col * (1 - core[..., None]) + np.array([20, 20, 95], np.float32) * core[..., None]
            alpha = np.maximum(alpha, rim * 0.9)
        _zombie_tex = (col, alpha.astype(np.float32))
    return _zombie_tex

def fx_zombie(frame, state):
    decay = state.get('zombieDecay', 60) / 100.0
    blood = state.get('zombieBlood', 70) / 100.0
    out = frame.copy()
    for x0, y0, x1, y1, pts, fw, z in _monster_faces(frame):
        roi = out[y0:y1, x0:x1].astype(np.float32)
        h, w = roi.shape[:2]
        u, down = _axes(pts)
        angle = math.degrees(math.atan2(u[1], u[0]))
        face = _oval_mask(pts, (h, w), fw)
        _tint_skin(roi, face, 0.55 + 0.35 * decay, (185, 225, 200), 0.9)              # sickly green-grey

        # Rot / bruise / wound overlay glued to the face
        tex_col, tex_a = _get_zombie_tex()
        M = cv2.getAffineTransform(np.float32([[150, 190], [362, 190], [256, 450]]),
                                   np.float32([pts[33], pts[263], pts[152]]))
        col = cv2.warpAffine(tex_col, M, (w, h), flags=cv2.INTER_LINEAR)
        a = cv2.warpAffine(tex_a, M, (w, h), flags=cv2.INTER_LINEAR)
        _blend(roi, col, a * face * decay)

        # Sunken eye sockets and hollow cheeks
        for outline in (_LEFT_EYE_OUTLINE, _RIGHT_EYE_OUTLINE):
            poly = pts[outline]
            ew = float(np.ptp(poly[:, 0]) + np.ptp(poly[:, 1])) * 0.5
            m = _soft_ellipse((h, w), poly.mean(0), (ew * 0.95, ew * 0.6), angle, ew * 0.3)
            _blend(roi, (55, 40, 60), m * (0.45 + 0.4 * decay) * face)
        for idx in (207, 427):
            m = _soft_ellipse((h, w), pts[idx], (fw * 0.11, fw * 0.15), angle, fw * 0.05)
            _blend(roi, (40, 50, 45), m * 0.3 * decay * face)

        # Milky dead eyes
        for outline, iris_c, ring in ((_LEFT_EYE_OUTLINE, 473, [474, 475, 476, 477]),
                                      (_RIGHT_EYE_OUTLINE, 468, [469, 470, 471, 472])):
            eye = _poly_mask((h, w), pts[outline], 1.0)
            _blend(roi, (205, 218, 205), eye * 0.85)
            rad = float(np.mean([np.linalg.norm(pts[i] - pts[iris_c]) for i in ring]))
            _blend(roi, (165, 170, 150), _soft_ellipse((h, w), pts[iris_c], (rad * 0.95, rad * 0.95), 0, 1.2) * eye * 0.7)
            _blend(roi, (35, 35, 35), _soft_ellipse((h, w), pts[iris_c], (rad * 0.35, rad * 0.35), 0, 0.8) * eye * 0.8)

        # Dead lips + blood running from the mouth corner
        _blend(roi, (60, 45, 85), _poly_mask((h, w), pts[_MONSTER_LIP_LOWER], 1.0) * 0.55)
        if blood > 0.02:
            bl = np.zeros((h, w), np.uint8)
            start = pts[291] + down * fw * 0.015
            chin_gap = float((pts[152] - start) @ down)                         # never run past the chin
            length = min(fw * (0.30 + 0.35 * blood), chin_gap * 0.92)
            prev = start
            for k in range(1, 13):
                t = k / 12.0
                p = start + down * length * t + u * math.sin(t * 6.0) * fw * 0.02 * t
                cv2.line(bl, tuple(np.rint(prev).astype(int)), tuple(np.rint(p).astype(int)), 255,
                         max(2, int(fw * 0.04 * (1 - 0.6 * t))), cv2.LINE_AA)
                prev = p
            cv2.circle(bl, tuple(np.rint(prev).astype(int)), max(2, int(fw * 0.016)), 255, -1, cv2.LINE_AA)
            bl = cv2.GaussianBlur(bl.astype(np.float32) / 255.0, (0, 0), 0.8)
            _blend(roi, (10, 10, 105), bl * 0.9)
            _blend(roi, (90, 90, 190), np.roll(bl, -max(1, int(fw * 0.006)), axis=1) * (1 - bl) * 0.35)   # wet highlight
        out[y0:y1, x0:x1] = np.clip(roi, 0, 255).astype(np.uint8)
    return out

# ── Skeleton / Mummy ───────────────────────────
_MOUTH_UPPER_OUTER = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
_MOUTH_LOWER_OUTER = [291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61]
_MOUTH_OUTER_LOOP  = _MOUTH_UPPER_OUTER + _MOUTH_LOWER_OUTER[1:-1]
_MOUTH_INNER_LOOP  = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308,
                      324, 318, 402, 317, 14, 87, 178, 88, 95]
_JAW_LINE = [234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397,
             288, 361, 323, 454]

def _eye_info(pts):
    """[(outline, centre, width)] for both eyes."""
    info = []
    for outline, a, b, iris in ((_LEFT_EYE_OUTLINE, 263, 362, 473), (_RIGHT_EYE_OUTLINE, 33, 133, 468)):
        info.append((outline, pts[iris], float(np.linalg.norm(pts[a] - pts[b])), iris))
    return info

def _contour_offset(contour, base_a, u, down, mouth_w, t):
    """Perpendicular offset (along down) of a mouth contour from the corner-to-corner line at fraction t."""
    rel = contour - base_a
    proj = (rel @ u) / mouth_w
    off = rel @ down
    order = np.argsort(proj)
    return float(np.interp(t, proj[order], off[order]))

def fx_skeleton(frame, state):
    decor = state.get('skullDecor', 60) / 100.0
    glow = state.get('skullGlow', 50) / 100.0
    out = frame.copy()
    for x0, y0, x1, y1, pts, fw, z in _monster_faces(frame):
        roi = out[y0:y1, x0:x1].astype(np.float32)
        h, w = roi.shape[:2]
        u, down = _axes(pts)
        angle = math.degrees(math.atan2(u[1], u[0]))
        face = _oval_mask(pts, (h, w), fw)
        g = cv2.cvtColor(np.clip(roi, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        bone = (205 + g[..., None] * 50) * np.array([0.93, 0.97, 1.0], np.float32)      # warm bone white, skin detail kept
        _blend(roi, bone, face * 0.92)

        # Cheekbone hollows, temples and a faint jaw line
        for idx in (207, 427):
            _blend(roi, (95, 95, 100), _soft_ellipse((h, w), pts[idx], (fw * 0.12, fw * 0.14), angle, fw * 0.05) * face * 0.35)
        jaw = np.zeros((h, w), np.uint8)
        cv2.polylines(jaw, [np.rint(pts[_JAW_LINE] + (pts[4] - pts[152]) * 0.05).astype(np.int32)], False, 255,
                      max(1, int(fw * 0.012)), cv2.LINE_AA)
        _blend(roi, (50, 50, 55), cv2.GaussianBlur(jaw.astype(np.float32) / 255.0, (0, 0), 1.0) * 0.55)

        # Nose cavity
        a_pt = pts[6] + (pts[4] - pts[6]) * 0.35
        nose = [a_pt, pts[327] + u * fw * 0.012, pts[2], pts[98] - u * fw * 0.012]
        _blend(roi, (12, 12, 14), _poly_mask((h, w), nose, 1.0) * 0.93)

        # Eye sockets
        eyes = _eye_info(pts)
        for outline, c, ew, iris in eyes:
            sock = _soft_ellipse((h, w), pts[outline].mean(0), (ew * 0.95, ew * 0.82), angle, 1.5)
            _blend(roi, (12, 12, 14), sock * 0.96)
            if glow > 0.02:
                r = max(2, int(ew * 0.12))
                dot = _soft_ellipse((h, w), c, (r, r), 0, 1.0)
                halo = _soft_ellipse((h, w), c, (r * 3, r * 3), 0, r * 1.5)
                _blend(roi, (255, 235, 200), halo * 0.35 * glow * sock)
                _blend(roi, (255, 250, 240), dot * 0.9 * glow * sock)

        # Día de los Muertos petals ringing the sockets
        if decor > 0.02:
            layer = np.zeros((h, w, 3), np.uint8)
            lmask = np.zeros((h, w), np.uint8)
            palette = [(40, 140, 255), (170, 70, 230), (200, 180, 40), (60, 200, 255)]   # orange, pink, teal, gold
            for outline, c, ew, iris in eyes:
                centre = pts[outline].mean(0)
                n_pet = 9
                for k in range(n_pet):
                    th = 2 * math.pi * k / n_pet
                    p = centre + np.array([math.cos(th), math.sin(th)], np.float32) * ew * (1.05 + 0.1 * decor)
                    ax = (int(max(2, ew * 0.30 * decor)), int(max(1, ew * 0.15 * decor)))
                    cv2.ellipse(layer, tuple(np.rint(p).astype(int)), ax, math.degrees(th), 0, 360,
                                palette[k % 4], -1, cv2.LINE_AA)
                    cv2.ellipse(layer, tuple(np.rint(p).astype(int)), ax, math.degrees(th), 0, 360,
                                (20, 20, 25), 1, cv2.LINE_AA)
                    cv2.ellipse(lmask, tuple(np.rint(p).astype(int)), (ax[0] + 1, ax[1] + 1), math.degrees(th), 0, 360, 255, -1, cv2.LINE_AA)
            lm = cv2.GaussianBlur(lmask.astype(np.float32) / 255.0, (0, 0), 0.6)
            for pt_ in (pts[10] + (pts[9] - pts[10]) * 0.2,):                           # forehead dots
                for t in np.linspace(-0.3, 0.3, 5):
                    p = pt_ + u * fw * t + down * fw * (0.05 + 0.04 * (1 - abs(t) / 0.3))
                    r = max(2, int(fw * 0.018 * decor))
                    cv2.circle(layer, tuple(np.rint(p).astype(int)), r, (255, 255, 255), -1, cv2.LINE_AA)
                    cv2.circle(layer, tuple(np.rint(p).astype(int)), r, (20, 20, 25), 1, cv2.LINE_AA)
                    cv2.circle(lmask, tuple(np.rint(p).astype(int)), r + 1, 255, -1, cv2.LINE_AA)
            lm = np.maximum(lm, cv2.GaussianBlur(lmask.astype(np.float32) / 255.0, (0, 0), 0.6))
            _blend(roi, layer.astype(np.float32), lm * 0.92)

        # Teeth: bone-white lips, black gap inside, vertical stitches across both lips
        lips = _poly_mask((h, w), pts[_MOUTH_OUTER_LOOP], 1.0)
        _blend(roi, (238, 242, 244), lips * 0.9)
        _blend(roi, (14, 14, 16), _poly_mask((h, w), pts[_MOUTH_INNER_LOOP], 0.8) * 0.88)
        ma, mb = pts[61], pts[291]
        mouth_w = float(np.linalg.norm(mb - ma)) + 1e-6
        mu = (mb - ma) / mouth_w
        md = np.array([-mu[1], mu[0]], np.float32)
        lines = np.zeros((h, w), np.uint8)
        thick = max(1, int(fw * 0.014))
        for t in np.linspace(0.06, 0.94, 11):
            base = ma + (mb - ma) * t
            top = base + md * (_contour_offset(pts[_MOUTH_UPPER_OUTER], ma, mu, md, mouth_w, t) - fw * 0.03)
            bot = base + md * (_contour_offset(pts[_MOUTH_LOWER_OUTER], ma, mu, md, mouth_w, t) + fw * 0.03)
            cv2.line(lines, tuple(np.rint(top).astype(int)), tuple(np.rint(bot).astype(int)), 255, thick, cv2.LINE_AA)
        cv2.line(lines, tuple(np.rint(ma - mu * fw * 0.02).astype(int)), tuple(np.rint(mb + mu * fw * 0.02).astype(int)),
                 255, thick, cv2.LINE_AA)
        _blend(roi, (18, 18, 22), cv2.GaussianBlur(lines.astype(np.float32) / 255.0, (0, 0), 0.6) * 0.92)
        out[y0:y1, x0:x1] = np.clip(roi, 0, 255).astype(np.uint8)
    return out

_mummy_tex = None

def _get_mummy_tex():
    """Canonical 768x768 bandage wrap (BGR + alpha) and grime noise. Eyes at y=300, chin at y=560."""
    global _mummy_tex
    if _mummy_tex is None:
        rng = np.random.RandomState(11)
        n = 768
        col = np.full((n, n, 3), (150, 180, 200), np.float32)
        base = np.array([170, 200, 220], np.float32)
        seam = np.array([95, 120, 140], np.float32)
        for _ in range(90):
            c = rng.uniform(0, n, 2)
            th = math.radians(rng.uniform(-38, 38))
            d = np.array([math.cos(th), math.sin(th)], np.float32)
            nrm = np.array([-d[1], d[0]], np.float32)
            wd = rng.uniform(20, 38)
            f = rng.uniform(0.78, 1.08)
            quad = np.array([c - d * 900 + nrm * wd, c + d * 900 + nrm * wd,
                             c + d * 900 - nrm * wd, c - d * 900 - nrm * wd], np.float32)
            m8 = np.zeros((n, n), np.uint8)
            cv2.fillPoly(m8, [np.rint(quad * 4).astype(np.int32)], 255, cv2.LINE_AA, shift=2)
            ys, xs = np.nonzero(m8)                                              # only touch pixels this strip covers
            m = (m8[ys, xs].astype(np.float32) / 255.0)[:, None]
            col[ys, xs] = col[ys, xs] * (1 - m) + base * f * m
            e = np.zeros((n, n), np.uint8)
            for sgn in (-1, 1):                                                  # dark seams along the long edges
                cv2.line(e, tuple(np.rint(c - d * 900 + nrm * wd * sgn).astype(int)),
                         tuple(np.rint(c + d * 900 + nrm * wd * sgn).astype(int)), 255, 2, cv2.LINE_AA)
            em = (e[ys, xs].astype(np.float32) / 255.0)[:, None] * m
            col[ys, xs] = col[ys, xs] * (1 - em * 0.7) + seam * em * 0.7
        col += rng.randn(n, n, 1).astype(np.float32) * 5
        noise = cv2.GaussianBlur(rng.rand(n, n).astype(np.float32), (0, 0), 14)
        noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-6)
        _mummy_tex = (np.clip(col, 0, 255).astype(np.float32), noise.astype(np.float32))
    return _mummy_tex

def _head_silhouette(pts, z, fw, shape, blur):
    """Soft mask of the whole head (hair included): a 3D ellipsoid placed from the face landmarks,
    projected to the screen, so it follows yaw/pitch instead of staying a flat oval."""
    P = lambda i: np.array([pts[i][0], pts[i][1], z[i]], np.float32)
    l3, r3, t3, c3 = P(234), P(454), P(10), P(152)
    ex = r3 - l3
    fw3 = float(np.linalg.norm(ex)) + 1e-6
    ex /= fw3
    ey = c3 - t3
    ey -= ex * float(ey @ ex)
    fh = float(np.linalg.norm(ey)) + 1e-6
    ey /= fh
    ef = np.cross(ex, ey)
    if ef[2] > 0:                                     # MediaPipe: smaller z = closer to the camera
        ef = -ef
    crown = t3 - ey * fw3 * 0.45
    vc = (crown + c3) * 0.5
    mid = (l3 + r3) * 0.5
    centre = vc + ex * float((mid - vc) @ ex) + ef * (float((mid - vc) @ ef) - fw3 * 0.08)
    R = np.stack([ex * fw3 * 0.56, ey * (fh + fw3 * 0.45) * 0.5, ef * fw3 * 0.66], 1)   # columns = semi-axes
    vals, vecs = np.linalg.eigh((R @ R.T)[:2, :2])
    axes = np.sqrt(np.maximum(vals[::-1], 1e-6))
    major = vecs[:, ::-1][:, 0]
    m = np.zeros(shape, np.uint8)
    cv2.ellipse(m, tuple(np.rint(centre[:2]).astype(int)), (int(axes[0]), int(axes[1])),
                math.degrees(math.atan2(major[1], major[0])), 0, 360, 255, -1, cv2.LINE_AA)
    return cv2.GaussianBlur(m.astype(np.float32) / 255.0, (0, 0), blur)

def fx_mummy(frame, state):
    grime = state.get('mummyGrime', 40) / 100.0
    out = frame.copy()
    for x0, y0, x1, y1, pts, fw, z in _monster_faces(frame):
        roi = out[y0:y1, x0:x1].astype(np.float32)
        h, w = roi.shape[:2]
        u, down = _axes(pts)
        up = -down
        angle = math.degrees(math.atan2(u[1], u[0]))
        centre = (pts[10] + pts[152]) * 0.5
        oval = centre + (pts[_FACE_OVAL_IDX] - centre) * 1.08
        sil = _head_silhouette(pts, z, fw, (h, w), fw * 0.015)
        hull = np.zeros((h, w), np.uint8)                                          # whole face mesh, enlarged a touch
        cv2.fillPoly(hull, [cv2.convexHull(np.rint(pts).astype(np.int32))], 255)
        r = max(1, int(fw * 0.04))
        hull = cv2.dilate(hull, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
        hull = cv2.GaussianBlur(hull.astype(np.float32) / 255.0, (0, 0), fw * 0.015)
        oval_m = _poly_mask((h, w), oval, fw * 0.015)
        extra = np.maximum(sil, hull)                                               # the wider 3D head + face mesh
        # Where an ear is showing, the wrap must stop at the (rounded) face outline instead of
        # the wider head shape. The ear is hidden on the side the nose points toward, so that
        # side may wrap all the way round. The crown (above the brows) is never restricted.
        mid = (pts[234] + pts[454]) * 0.5
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
        lat = (xs - mid[0]) * u[0] + (ys - mid[1]) * u[1]                           # across the face, along u
        vert = (xs - pts[10][0]) * down[0] + (ys - pts[10][1]) * down[1]            # down from the hairline
        v_brow = float(((pts[33] + pts[263]) * 0.5 - pts[10]) @ down)
        crown = np.clip((0.85 * v_brow - vert) / (fw * 0.10) + 0.5, 0, 1)
        r2 = max(1, int(fw * 0.025))
        near_oval = cv2.GaussianBlur(cv2.dilate(oval_m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r2 + 1, 2 * r2 + 1))),
                                     (0, 0), fw * 0.015)
        allowed = np.maximum(near_oval, crown)
        nose = float((pts[4] - mid) @ u) / fw
        restrict = np.zeros((h, w), np.float32)
        for sgn in (-1, 1):
            toward = sgn * nose                                                      # >0: nose points to this side
            clip_amt = 1.0 - float(np.clip((toward - 0.04) / 0.06, 0, 1))
            side = np.clip(sgn * lat / (fw * 0.05) + 0.5, 0, 1)
            restrict += clip_amt * side
        head = np.maximum(oval_m, extra * (1.0 - restrict * (1.0 - allowed)))
        M = cv2.getAffineTransform(np.float32([[230, 300], [442, 300], [336, 560]]),
                                   np.float32([pts[33], pts[263], pts[152]]))
        tex, noise = _get_mummy_tex()
        col = cv2.warpAffine(tex, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        nz = cv2.warpAffine(noise, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        col = col * (1 - 0.55 * grime * nz[..., None]) + np.array([60, 95, 120], np.float32) * (0.55 * grime * nz[..., None])
        g = cv2.GaussianBlur(cv2.cvtColor(np.clip(roi, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0,
                             (0, 0), fw * 0.02)
        col *= (0.62 + 0.55 * g)[..., None]                                       # keep the face's volume under the wrap

        # Eye slit
        hole = np.zeros((h, w), np.float32)
        ring = np.zeros((h, w), np.float32)
        for outline, c, ew, iris in _eye_info(pts):
            ec = pts[outline].mean(0)
            hole = np.maximum(hole, _soft_ellipse((h, w), ec, (ew * 0.78, ew * 0.40), angle, 1.5))
            ring = np.maximum(ring, _soft_ellipse((h, w), ec, (ew * 1.10, ew * 0.70), angle, ew * 0.25))
        cover = head * (1 - hole)
        _blend(roi, col, cover)
        _blend(roi, (20, 25, 35), np.clip(ring - hole, 0, 1) * head * 0.55)
        out[y0:y1, x0:x1] = np.clip(roi, 0, 255).astype(np.uint8)
    return out

# ── X-Ray — bone-scan look ───────────────────────────────────────
def fx_xray(frame, state):
    contrast = state.get('xrayContrast', 60) / 100.0   # 0..1
    H, W = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # High-frequency detail via unsharp mask — brings out edges/structure
    blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=3)
    sharp = np.clip(gray * (1 + contrast * 2.5) - blur * contrast * 2.5, 0, 255)

    # Invert so bright areas (flesh/clothing) become dark, edges stay bright
    inv = 255.0 - sharp
    inv = np.clip(inv * (0.5 + contrast * 0.8), 0, 255)

    # Gamma lift to push mid-tones toward lighter bone-like appearance
    gamma = 0.55
    inv_norm = np.power(inv / 255.0, gamma) * 255.0

    # Bluish-white X-ray tint: B=full, G=0.85, R=0.55
    out = np.zeros((H, W, 3), dtype=np.uint8)
    out[:, :, 0] = np.clip(inv_norm * 1.00, 0, 255).astype(np.uint8)   # B
    out[:, :, 1] = np.clip(inv_norm * 0.85, 0, 255).astype(np.uint8)   # G
    out[:, :, 2] = np.clip(inv_norm * 0.55, 0, 255).astype(np.uint8)   # R
    return out

# ── Blaze — heat shimmer warp + fire palette ─────────────────────
# The image is first warped by an animated noise field (heat distortion),
# then the luminance of the warped result drives the fire palette.
# This gives visible heat-haze rippling, not just a colour remap.
# All work is at half-res; coordinate grids are cached.
_blaze_t    = 0.0
_blaze_last = None
_blaze_grid = {}   # (sh, sw) -> (base_x, base_y, nx, ny) cached

def _get_blaze_grid(sh, sw):
    key = (sh, sw)
    if key not in _blaze_grid:
        X = np.linspace(0.0, 8.0, sw, dtype=np.float32)
        Y = np.linspace(0.0, 8.0, sh, dtype=np.float32)
        nx, ny = np.meshgrid(X, Y)
        # Base pixel coordinate grids for remapping
        bx = np.tile(np.arange(sw, dtype=np.float32), (sh, 1))
        by = np.repeat(np.arange(sh, dtype=np.float32)[:, None], sw, axis=1)
        _blaze_grid[key] = (bx, by, nx, ny)
    return _blaze_grid[key]

def fx_infrared(frame, state):
    global _blaze_t, _blaze_last
    glow = state.get('infraredGlow', 60) / 100.0
    now  = time.time()
    dt   = min(0.1, now - _blaze_last) if _blaze_last else 0.016
    _blaze_last = now
    _blaze_t   += dt * 0.5

    H, W = frame.shape[:2]
    sw, sh = W // 2, H // 2
    t = _blaze_t

    small = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_AREA)
    bx, by, nx, ny = _get_blaze_grid(sh, sw)

    # --- heat shimmer warp with strong upward bias (flames rise) ---
    warp_strength = 4.0
    # dx: side-to-side flicker
    dx = (np.sin(ny * 1.8 + t * 2.1) * np.cos(nx * 1.3 + t * 0.7)) * warp_strength
    # dy: upward bias — base upward drift + turbulence, net motion is upward
    dy_turb = (np.cos(ny * 2.2 - t * 2.8) * np.sin(nx * 1.6 - t * 1.9)) * warp_strength * 0.4
    dy_rise = -ny / 8.0 * warp_strength * 1.2   # steady upward pull (ny goes 0..8, top=0)
    dy = dy_turb + dy_rise
    src_x = np.clip(bx + dx, 0, sw - 1).astype(np.float32)
    src_y = np.clip(by + dy, 0, sh - 1).astype(np.float32)
    warped = cv2.remap(small, src_x, src_y, cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REFLECT_101)

    # --- heat source: thin edge lines + luminance interior ---
    gray_s = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY).astype(np.float32)
    lum = gray_s / 255.0

    gx_ = cv2.Sobel(gray_s, cv2.CV_32F, 1, 0, ksize=3)
    gy_ = cv2.Sobel(gray_s, cv2.CV_32F, 0, 1, ksize=3)
    # Divide by 120 (was 60) → edges are half as thick/bright on flat areas like faces
    edges = np.clip(np.sqrt(gx_*gx_ + gy_*gy_) / 120.0, 0, 1)
    # Reduce edge contribution slightly so interior fill dominates on smooth skin
    heat = np.clip(edges * 0.45 + lum * 0.55, 0, 1)

    # Bloom spreads the fire glow outward — upward-biased blur kernel
    heat8   = (heat * 255).astype(np.uint8)
    sig     = max(1, int(glow * 9))
    bloomed = cv2.GaussianBlur(heat8.astype(np.float32), (0, 0), sigmaX=sig * 0.6, sigmaY=sig)
    heat = np.clip(heat * 255 * 0.5 + bloomed * 0.6, 0, 255) / 255.0

    # Glow shifts palette: dim → red outlines, high → orange/yellow bloom
    scale = 2.5 + glow * 1.5
    r = np.clip(heat * scale,        0, 1)
    g = np.clip(heat * scale - 1.2,  0, 1)
    b = np.clip(heat * scale - 3.5,  0, 1)

    out_s = np.zeros((sh, sw, 3), dtype=np.uint8)
    out_s[:, :, 2] = (r * 255).astype(np.uint8)
    out_s[:, :, 1] = (g * 255).astype(np.uint8)
    out_s[:, :, 0] = (b * 255).astype(np.uint8)

    return cv2.resize(out_s, (W, H), interpolation=cv2.INTER_LINEAR)

# ── Psychedelic — per-pixel hue cycling based on luminance ───────
_psycho_offset = 0.0
_psycho_last   = None

def fx_psychedelic(frame, state):
    global _psycho_offset, _psycho_last
    speed = state.get('psychoSpeed', 50) / 100.0
    now   = time.time()
    dt    = min(0.1, now - _psycho_last) if _psycho_last else 0.016
    _psycho_last = now
    _psycho_offset = (_psycho_offset + speed * dt * 180.0) % 180.0

    H, W = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)

    # Per-pixel hue shift based on luminance — creates morphing rainbow bands
    lum_shift = (hsv[:, :, 2] / 255.0) * 90.0   # 0..90 degrees based on brightness
    hsv[:, :, 0] = (hsv[:, :, 0] + _psycho_offset + lum_shift) % 180.0

    # Boost saturation to full
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.5 + 40, 0, 255)

    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

# ── Oil Painting — cv2.xphoto.oilPainting (native C++, real-time) ─
def fx_oil_painting(frame, state):
    """Oil painting via cv2.xphoto.oilPainting — runs in C++ at full speed.

    oilRadius slider (2-8): brush neighbourhood size (odd values work best)
    oilLevels slider (3-16): intensity quantisation levels
    """
    radius = max(1, state.get('oilRadius', 4))
    levels = max(3, state.get('oilLevels', 8))

    # oilPainting needs an odd neighbourhood size
    size = radius * 2 + 1

    result = cv2.xphoto.oilPainting(frame, size, levels)

    # Saturation boost for that rich, paint-from-a-tube look
    hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.4 + 15, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

# ── Cartoon — cel shading (hard quantized light bands, video-game look) ──
def fx_cartoon(frame, state):
    """True cel shading: quantize luminance into hard bands, keep full hue,
    then add crisp ink outlines.  Looks like Borderlands / Wind Waker."""
    level  = max(2, min(8, state.get('cartoonLevel', 4)))   # number of shade bands
    H, W = frame.shape[:2]

    # 1. Light smooth to kill sensor noise without smearing colour boundaries
    smooth = cv2.bilateralFilter(frame, d=7, sigmaColor=60, sigmaSpace=60)

    # 2. Convert to HSV; quantize V (brightness) into `level` hard steps
    hsv = cv2.cvtColor(smooth, cv2.COLOR_BGR2HSV).astype(np.float32)
    v   = hsv[:, :, 2] / 255.0                          # 0..1
    # Hard quantize: floor to nearest band, then snap to band centre
    v_q = (np.floor(v * level) / level + 0.5 / level).clip(0, 1)
    # Boost saturation for that vivid toon look
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.4 + 20, 0, 255)
    hsv[:, :, 2] = (v_q * 255).astype(np.float32)
    cel = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # 3. Crisp ink outlines from the original frame's luminance
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray  = cv2.medianBlur(gray, 3)
    edges = cv2.Canny(gray, 60, 160)
    # Slightly thicken lines so they read well at any resolution
    edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
    ink   = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # 4. Stamp black ink onto cel colours
    return np.where(ink > 0, 0, cel).astype(np.uint8)

# Effect registry — None pinned first, rest sorted alphabetically by label
_EFFECTS_RAW = {
    'none':          {'label': 'None',               'icon': '🚫', 'fn': fx_none,          'controls': []},
    'ascii':         {'label': 'ASCII Art',          'icon': '📝', 'fn': fx_ascii,         'controls': [('asciiRes','Resolution','px',4,20,8,'int')]},
    'cartoon':       {'label': 'Cartoon',            'icon': '🖼', 'fn': fx_cartoon,       'controls': [('cartoonLevel','Shade Bands','',2,8,4,'int')]},
    # 'chromatic':     {'label': 'Chromatic Aberr.',   'icon': '🌈', 'fn': fx_chromatic,     'controls': [('chromaticShift','Shift','px',1,20,6,'int')]},
    'promist':       {'label': 'Pro-Mist',           'icon': '☁️', 'fn': fx_promist,       'controls': [('mistGlow','Glow','%',0,100,60,'int'),('mistSoft','Softness','%',0,100,60,'int'),('mistMono','Black & White','%',0,100,100,'int')]},
    'emboss':        {'label': 'Emboss',             'icon': '🗿', 'fn': fx_emboss,        'controls': [('embossDepth','Depth','%',10,100,50,'int'),('embossLight','Light','',-100,100,0,'int')]},
    'glitch':        {'label': 'Glitch',             'icon': '⚡', 'fn': fx_glitch,        'controls': [('glitchIntensity','Intensity','%',10,100,50,'int')]},
    'angry_eyes':    {'label': 'Angry Eyes',         'icon': '😤', 'fn': fx_angry_eyes,    'controls': [('angryIntensity','Intensity','%',10,100,60,'int')]},
    'hamster_eyes2': {'label': 'Hamster Eye',        'icon': '🐭', 'fn': fx_hamster_eyes2, 'controls': [('hamster2Scale','Eye Size','%',100,150,150,'int')]},
    'radar':         {'label': 'Radar',              'icon': '📡', 'fn': fx_radar,         'controls': [('radarFade','Fade Speed','%',25,100,75,'int')]},
    'kaleidoscope':  {'label': 'Kaleidoscope',       'icon': '🔯', 'fn': fx_kaleidoscope,  'controls': [('kaleidoSegments','Segments','',2,16,6,'int')]},
    'mirror':        {'label': 'Mirror',             'icon': '🪞', 'fn': fx_mirror,        'controls': [('mirrorVert','Mirror Vertical','',0,1,1,'bool'),('mirrorHoriz','Mirror Horizontal','',0,1,0,'bool')]},
    'neon_edge':     {'label': 'Neon Edge',          'icon': '💜', 'fn': fx_neon_edge,     'controls': []},
    'night_vision':  {'label': 'Night Vision',       'icon': '🌙', 'fn': fx_night_vision,  'controls': []},
    'pixelate':      {'label': 'Pixelate',           'icon': '🟦', 'fn': fx_pixelate,      'controls': [('pixelSize','Pixel Size','px',4,32,12,'int')]},
    'roto_zoom':     {'label': 'Roto-Zoom',          'icon': '🎡', 'fn': fx_roto_zoom,     'controls': [('rotoSpeed','Speed','',-100,100,10,'int'),('rotoZoom','Zoom','',50,400,150,'int')]},
    'cube':          {'label': 'Rotating Cube',      'icon': '🎲', 'fn': fx_rotating_cube, 'controls': [('cubeSpeed','Spin','',-100,100,10,'int')]},
    'thermal':       {'label': 'Thermal Camera',     'icon': '🌡', 'fn': fx_thermal,       'controls': []},
    'twist':         {'label': 'Twist/Spiral',       'icon': '🌀', 'fn': fx_twist,         'controls': [('twistAmount','Twist','%',0,100,50,'int')]},
    'tv_snow':       {'label': 'TV Snow',            'icon': '📺', 'fn': fx_tv_snow,       'controls': [('snowGhost','Ghost Strength','%',0,80,30,'int')]},
    'skeleton':      {'label': 'Skeleton',           'icon': '💀', 'fn': fx_skeleton,      'controls': [('skullDecor','Flower Decor (0 = off)','%',0,100,60,'int'),('skullGlow','Eye Glow','%',0,100,50,'int')]},
    'mummy':         {'label': 'Mummy',              'icon': '🩹', 'fn': fx_mummy,         'controls': [('mummyGrime','Grime','%',0,100,40,'int')]},
    'zombie':        {'label': 'Zombie',             'icon': '🧟', 'fn': fx_zombie,        'controls': [('zombieDecay','Decay','%',0,100,60,'int'),('zombieBlood','Blood','%',0,100,70,'int')]},
    'vampire_teeth': {'label': 'Vampire',      'icon': '🧛', 'fn': fx_vampire_teeth, 'controls': [('vampireSize','Fang Size','%',60,150,100,'int'), ('vampireEyeHue','Iris Color (hue)','',0,179,0,'int'), ('vampireWings','Bat Wing Size (0 = off)','%',0,150,100,'int')]},
    'vintage_sepia': {'label': 'Vintage/Sepia',      'icon': '📷', 'fn': fx_vintage_sepia, 'controls': []},
    'water_push':    {'label': 'Water Push',         'icon': '💧', 'fn': fx_water_push,    'controls': [('waterStrength','Strength','',1,80,30,'int')]},
    'so_pretty':     {'label': 'So Pretty',          'icon': '💄', 'fn': fx_so_pretty,     'controls': []},
    'sid_eyes':      {'label': 'Sid Eyes',           'icon': '🦥', 'fn': fx_sid_eyes,      'controls': [('sidShift','Eye Shift','px',0,30,15,'int')]},
    'xray':          {'label': 'X-Ray',              'icon': '🦴', 'fn': fx_xray,          'controls': [('xrayContrast','Contrast','%',10,100,60,'int')]},
    'infrared':      {'label': 'Blaze',              'icon': '🔥', 'fn': fx_infrared,      'controls': [('infraredGlow','Glow','%',10,100,60,'int')]},
    'psychedelic':   {'label': 'Psychedelic',        'icon': '🌈', 'fn': fx_psychedelic,   'controls': [('psychoSpeed','Speed','%',5,100,50,'int')]},
    'oil_painting':  {'label': 'Oil Painting',       'icon': '🖌', 'fn': fx_oil_painting,  'controls': [('oilRadius','Brush Size','',2,8,4,'int'),('oilLevels','Paint Levels','',3,16,8,'int')]},
    'wave':          {'label': 'Wave Distort',       'icon': '〰️', 'fn': fx_wave,          'controls': [('waveAmplitude','Amplitude','px',1,50,20,'int'),('waveFrequency','Frequency','',1,15,5,'int')]},
}

# Pin 'none' first, then sort the rest alphabetically by label
EFFECTS = {'none': _EFFECTS_RAW['none']} | dict(
    sorted(
        ((k, v) for k, v in _EFFECTS_RAW.items() if k != 'none'),
        key=lambda kv: kv[1]['label'].lower()
    )
)
