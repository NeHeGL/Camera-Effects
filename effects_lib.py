# Effects library for Camera Effects desktop app
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

def fx_hologram(frame, state):
    H, W = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    t = time.time()
    flicker = 0.97 + 0.03 * math.sin(t * 4.7)   # subtle, slow flicker
    out = np.zeros((H, W, 3), dtype=np.uint8)
    scan = (np.arange(H) % 4 < 2).astype(np.float32)
    v = np.clip(gray * 1.5 * flicker, 0, 255) * scan[:, None]
    out[:, :, 1] = np.clip(v * 0.9, 0, 255).astype(np.uint8)
    out[:, :, 0] = np.clip(v * 0.6, 0, 255).astype(np.uint8)
    bar_y = int((t * 200) % H)   # faster bar: 200 px/sec
    bar_h = max(2, H // 60)      # thinner bar
    out[bar_y:bar_y+bar_h, :, 1] = np.clip(out[bar_y:bar_y+bar_h, :, 1].astype(np.float32) * 1.8, 0, 255).astype(np.uint8)
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

def fx_night_vision(frame, state):
    H, W = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    noise = np.random.uniform(-15, 15, gray.shape).astype(np.float32)
    v = np.clip(gray * 1.4 + noise, 0, 255)
    green = np.zeros((H, W, 3), dtype=np.uint8)
    green[:, :, 1] = np.clip(v * 1.2, 0, 255).astype(np.uint8)
    cy, cx = H / 2, W / 2
    Y, X = np.ogrid[:H, :W]
    dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
    alpha = np.clip((dist - H*0.25) / (H*0.55), 0, 1) * 0.7
    for c in range(3):
        green[:, :, c] = np.clip(green[:, :, c].astype(np.float32) * (1 - alpha), 0, 255).astype(np.uint8)
    return green

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
    # Left half mirrored to fill the whole frame (true mirror effect)
    H, W = frame.shape[:2]
    left = frame[:, :W//2, :]
    right = cv2.flip(left, 1)
    return np.concatenate([left, right], axis=1)

def fx_mirror_h(frame, state):
    # Top half mirrored downward
    H, W = frame.shape[:2]
    top = frame[:H//2, :]
    out = frame.copy()
    out[H//2:H//2+top.shape[0], :] = cv2.flip(top, 0)
    return out

def fx_chromatic(frame, state):
    shift = state.get('chromaticShift', 6)
    out = np.empty_like(frame)
    out[:, :, 0] = np.roll(frame[:, :, 0], -shift, axis=1)
    out[:, :, 1] = frame[:, :, 1]
    out[:, :, 2] = np.roll(frame[:, :, 2],  shift, axis=1)
    return out

def fx_dream(frame, state):
    bloom = state.get('dreamBloom', 50) / 100.0
    r = max(1, int(bloom * 30)) | 1
    blurred = cv2.GaussianBlur(frame, (r, r), 0).astype(np.float32)
    out = np.clip(frame.astype(np.float32) * 0.6 + blurred * 0.7, 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.int32)
    hsv[:,:,0] = (hsv[:,:,0] + 20) % 180
    hsv[:,:,1] = np.clip(hsv[:,:,1] * 0.7, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

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

# ── Emboss — raised-relief grayscale effect ───────────────────────
def fx_emboss(frame, state):
    depth = state.get('embossDepth', 50) / 50.0   # 0..2, default 1.0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    # Classic emboss kernel (northwest light source)
    kernel = np.array([[-2*depth, -depth,  0],
                       [-depth,    1,      depth],
                       [ 0,        depth,  2*depth]], dtype=np.float32)
    embossed = cv2.filter2D(gray, -1, kernel) + 128.0
    embossed = np.clip(embossed, 0, 255).astype(np.uint8)
    return cv2.cvtColor(embossed, cv2.COLOR_GRAY2BGR)

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

# ── Cartoon — bilateral smooth + edge outline ─────────────────────
def fx_cartoon(frame, state):
    level = max(1, min(10, state.get('cartoonLevel', 5)))
    H, W = frame.shape[:2]

    # 1. Bilateral filter at half-res: more passes = flatter comic-book colours
    small = cv2.resize(frame, (W // 2, H // 2))
    passes = 1 + level // 3          # 1..4 passes as level goes 1..10
    sigma  = 40 + level * 8          # 48..120: stronger smoothing at high level
    for _ in range(passes):
        small = cv2.bilateralFilter(small, d=9, sigmaColor=sigma, sigmaSpace=sigma)
    smooth = cv2.resize(small, (W, H))

    # 2. Edge mask: lower C = more edges = more inky at high cartoonLevel
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray  = cv2.medianBlur(gray, 5)
    C_val = max(1, 8 - level // 2)   # 8..3: fewer missed edges as level rises
    edges = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        blockSize=9, C=C_val)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # 3. Combine
    return cv2.bitwise_and(smooth, edges)

# Effect registry — alphabetical by label (None pinned first)
EFFECTS = {
    'none':          {'label': 'None',               'icon': '🚫', 'fn': fx_none,          'controls': []},
    'ascii':         {'label': 'ASCII Art',          'icon': '📝', 'fn': fx_ascii,         'controls': [('asciiRes','Resolution','px',4,20,8,'int')]},
    'cartoon':       {'label': 'Cartoon',            'icon': '🖼', 'fn': fx_cartoon,       'controls': [('cartoonLevel','Intensity','',1,10,5,'int')]},
    'chromatic':     {'label': 'Chromatic Aberr.',   'icon': '🌈', 'fn': fx_chromatic,     'controls': [('chromaticShift','Shift','px',1,20,6,'int')]},
    'dream':         {'label': 'Dream/Bloom',        'icon': '✨', 'fn': fx_dream,         'controls': [('dreamBloom','Bloom','%',0,100,50,'int')]},
    'emboss':        {'label': 'Emboss',             'icon': '🗿', 'fn': fx_emboss,        'controls': [('embossDepth','Depth','%',10,100,50,'int')]},
    'glitch':        {'label': 'Glitch',             'icon': '⚡', 'fn': fx_glitch,        'controls': [('glitchIntensity','Intensity','%',10,100,50,'int')]},
    'hologram':      {'label': 'Hologram',           'icon': '👾', 'fn': fx_hologram,      'controls': []},
    'kaleidoscope':  {'label': 'Kaleidoscope',       'icon': '🔯', 'fn': fx_kaleidoscope,  'controls': [('kaleidoSegments','Segments','',2,16,6,'int')]},
    'mirror':        {'label': 'Mirror Vert.',       'icon': '🪞', 'fn': fx_mirror,        'controls': []},
    'mirror_h':      {'label': 'Mirror Horiz',       'icon': '↕️', 'fn': fx_mirror_h,      'controls': []},
    'neon_edge':     {'label': 'Neon Edge',          'icon': '💜', 'fn': fx_neon_edge,     'controls': []},
    'night_vision':  {'label': 'Night Vision',       'icon': '🌙', 'fn': fx_night_vision,  'controls': []},
    'pixelate':      {'label': 'Pixelate',           'icon': '🟦', 'fn': fx_pixelate,      'controls': [('pixelSize','Pixel Size','px',4,32,12,'int')]},
    'roto_zoom':     {'label': 'Roto-Zoom',          'icon': '🎡', 'fn': fx_roto_zoom,     'controls': [('rotoSpeed','Speed','',-100,100,10,'int'),('rotoZoom','Zoom','',50,400,150,'int')]},
    'cube':          {'label': 'Rotating Cube',      'icon': '🎲', 'fn': fx_rotating_cube, 'controls': [('cubeSpeed','Spin','',-100,100,10,'int')]},
    'thermal':       {'label': 'Thermal Camera',     'icon': '🌡', 'fn': fx_thermal,       'controls': []},
    'twist':         {'label': 'Twist/Spiral',       'icon': '🌀', 'fn': fx_twist,         'controls': [('twistAmount','Twist','%',0,100,50,'int')]},
    'tv_snow':       {'label': 'TV Snow',             'icon': '📺', 'fn': fx_tv_snow,       'controls': [('snowGhost','Ghost Strength','%',0,80,30,'int')]},
    'vintage_sepia': {'label': 'Vintage/Sepia',      'icon': '📷', 'fn': fx_vintage_sepia, 'controls': []},
    'water_push':    {'label': 'Water Push',         'icon': '💧', 'fn': fx_water_push,    'controls': [('waterStrength','Strength','',1,80,30,'int')]},
    'wave':          {'label': 'Wave Distort',       'icon': '〰️', 'fn': fx_wave,          'controls': [('waveAmplitude','Amplitude','px',1,50,20,'int'),('waveFrequency','Frequency','',1,15,5,'int')]},
}

