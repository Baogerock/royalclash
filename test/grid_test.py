from PIL import Image, ImageDraw, ImageFont

IMG_PATH = "top_80.png"
OUT_PATH = "regions_square_grid_with_ids_drop_small_cells_skip_buildings.png"

# ---------- 区域 2：拆成 13 行 ----------
region2_rows = [
    (53, 137.0000, 666, 163.5385),
    (53, 163.5385, 666, 190.0769),
    (53, 190.0769, 666, 216.6154),
    (53, 216.6154, 666, 243.1538),
    (53, 243.1538, 666, 269.6923),
    (53, 269.6923, 666, 296.2308),
    (53, 296.2308, 666, 322.7692),
    (53, 322.7692, 666, 349.3077),
    (53, 349.3077, 666, 375.8462),
    (53, 375.8462, 666, 402.3846),
    (53, 402.3846, 666, 428.9231),
    (53, 428.9231, 666, 455.4615),
    (53, 455.4615, 666, 482.0000),
]

# ---------- 区域 5：拆成 13 行 ----------
region5_rows = [
    (53, 586.0000, 666, 615),
    (53, 615, 666, 643),
    (53, 643, 666, 672),
    (53, 672, 666, 700),
    (53, 700, 666, 728),
    (53, 728, 666, 756),
    (53, 756, 666, 783),
    (53, 783, 666, 811),
    (53, 811, 666, 840),
    (53, 840, 666, 868),
    (53, 868, 666, 896),
    (53, 896, 666, 924),
    (53, 924, 666, 950.0000),
]

regions = [
    (258, 112, 461, 137),  # 1
    *region2_rows,         # 2（13 行）
    (85, 481, 633, 507),   # 3
(120, 506, 221, 559),   # 3
(499, 506, 597, 558),   # 3
    (85, 559, 632, 587),   # 4
    *region5_rows,         # 5（13 行）
    (257, 950, 460, 980),  # 6
]

# =========================
# 网格/编号参数
# =========================
S = 34  # 标准格子像素边长

# 若最后残格 < MIN_CELL_RATIO * S，则丢弃（不画最后线、不编号）
MIN_CELL_RATIO = 0.5

LINE_COLOR = (0, 255, 255, 160)
LINE_WIDTH = 2

FONT_SIZE = 14
TEXT_FILL = (255, 255, 255, 230)
TEXT_STROKE_FILL = (0, 0, 0, 200)
TEXT_STROKE_W = 2

DRAW_LABEL_EVERY = 1        # 1=每格都标；2=隔格标...
SHOW_GLOBAL_ID = True       # True: 画全局ID；False: 画 r,c（区域内）

# =========================
# 建筑占位：跳过这些 ID（不绘制编号）
# =========================
SKIP_IDS = {
    # 敌方/上半区（你给的）
    13,14,15,16,31,32,33,34,49,50,51,52,67,68,69,70,
    80,81,82,98,99,100,116,117,118,
    91,92,93,109,110,111,127,128,129,
    # 桥
    256,257,258,259,260,261,
    262,263,264,265,266,267,
    # 我方/下半区（你给的）
    382,383,384,400,401,402,418,419,420,
    393,394,395,411,412,413,429,430,431,
    441,442,443,444,459,460,461,462,477,478,479,480,495,496,497,498,
}

# =========================
# 画图初始化
# =========================
img = Image.open(IMG_PATH).convert("RGBA")
overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
draw = ImageDraw.Draw(overlay)

try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", FONT_SIZE)
except:
    font = ImageFont.load_default()


def make_lines_drop_small(a, b, step, min_ratio):
    """
    等步长铺线，但如果最后残余长度 < min_ratio*step，则丢弃残余，不补 b。
    返回线坐标列表（包含 a；可能不包含 b）。
    """
    lines = [float(a)]
    x = float(a) + step

    while x < float(b):
        lines.append(x)
        x += step

    remainder = float(b) - lines[-1]
    if remainder >= step * min_ratio:
        lines.append(float(b))  # 残余足够大，保留为最后一格

    return lines


def draw_grid_and_ids(x0, y0, x1, y1, step, min_ratio, start_id=0):
    """
    画网格 + 编号（丢弃过小的残格）
    注意：start_id 是全局递增 ID；即使跳过绘制，也会递增，保证 ID 不错位
    返回 next_id
    """
    x_lines = make_lines_drop_small(x0, x1, step, min_ratio)
    y_lines = make_lines_drop_small(y0, y1, step, min_ratio)

    if len(x_lines) < 2 or len(y_lines) < 2:
        return start_id

    # 画竖线
    for x in x_lines:
        draw.line([(x, y_lines[0]), (x, y_lines[-1])], fill=LINE_COLOR, width=LINE_WIDTH)

    # 画横线
    for y in y_lines:
        draw.line([(x_lines[0], y), (x_lines[-1], y)], fill=LINE_COLOR, width=LINE_WIDTH)

    next_id = start_id

    # 画编号
    for r in range(len(y_lines) - 1):
        for c in range(len(x_lines) - 1):

            # 这个格子的全局ID（不论画不画都要占位递增）
            cell_id = next_id
            next_id += 1

            # 建筑占位：编号直接跳过（不绘制）
            if cell_id in SKIP_IDS:
                continue

            # 控制稀疏标注（注意：这里不会影响 cell_id 连续性）
            if (r % DRAW_LABEL_EVERY != 0) or (c % DRAW_LABEL_EVERY != 0):
                continue

            cx = (x_lines[c] + x_lines[c + 1]) / 2.0
            cy = (y_lines[r] + y_lines[r + 1]) / 2.0

            label = str(cell_id) if SHOW_GLOBAL_ID else f"{r},{c}"

            bbox = draw.textbbox((0, 0), label, font=font, stroke_width=TEXT_STROKE_W)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]

            draw.text(
                (cx - tw / 2, cy - th / 2),
                label,
                font=font,
                fill=TEXT_FILL,
                stroke_width=TEXT_STROKE_W,
                stroke_fill=TEXT_STROKE_FILL,
            )

    return next_id


gid = 0
for (x0, y0, x1, y1) in regions:
    gid = draw_grid_and_ids(
        x0, y0, x1, y1,
        step=S,
        min_ratio=MIN_CELL_RATIO,
        start_id=gid
    )

out = Image.alpha_composite(img, overlay).convert("RGB")
out.save(OUT_PATH, quality=95)
print("saved:", OUT_PATH, "total_cells_including_skipped:", gid)
