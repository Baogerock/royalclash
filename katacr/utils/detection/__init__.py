import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from functools import partial
from PIL import Image, ImageDraw, ImageFont

def plot_box_PIL(
    image: Image.Image,
    box_params: tuple | np.ndarray,
    text="",
    fontsize=12,
    box_color="red",
    format="yolo",
    draw_center_point=False,
    alpha_channel=150,
):
  draw = ImageDraw.Draw(image)
  params, shape = np.array(box_params), image.size
  if np.max(params) <= 1.0:
    params[0] *= shape[0]
    params[2] *= shape[0]
    params[1] *= shape[1]
    params[3] *= shape[1]
  if format.lower() == "yolo":
    x_min = int(params[0] - params[2] / 2)
    y_min = int(params[1] - params[3] / 2)
    w = int(params[2])
    h = int(params[3])
  elif format.lower() == "coco":
    x_min = int(params[0])
    y_min = int(params[1])
    w = int(params[2])
    h = int(params[3])
  elif format.lower() == "voc":
    x_min = int(params[0])
    y_min = int(params[1])
    w = int(params[2] - params[0])
    h = int(params[3] - params[1])
  if isinstance(box_color, str) and box_color == "red":
    box_color = (255, 0, 0)
  if not isinstance(box_color, tuple):
    box_color = tuple(box_color)
  box_color = box_color + (alpha_channel,)
  draw.rectangle([x_min, y_min, x_min + w, y_min + h], outline=box_color, width=2)

  font_color = (255, 255, 255)
  font = ImageFont.load_default()
  import PIL
  pil_version = int(PIL.__version__.split(".")[0])
  w_text, h_text = font.getbbox(text)[-2:] if pil_version >= 10 else font.getsize(text)
  x_text = x_min
  y_text = y_min - h_text if y_min > h_text else y_min
  draw.rounded_rectangle([x_text, y_text, x_text + w_text, y_text + h_text], radius=1.5, fill=box_color)
  draw.text((x_text, y_text), text, fill=font_color, font=font)
  if draw_center_point:
    draw.rounded_rectangle([x_min + w / 2 - 2, y_min + h / 2 - 2, x_min + w / 2 + 2, y_min + h / 2 + 2],
                           radius=1.5, fill=(255, 0, 0))
  return image

def get_box_colors(n):
  cmap = plt.cm.brg
  step = cmap.N // n
  colors = cmap([i for i in range(0, cmap.N, step)])
  colors = (colors[:, :3] * 255).astype(int)
  return [tuple(color) for color in colors]

def build_label2colors(labels):
  if not len(labels):
    return {}
  labels = np.unique(labels).astype(np.int32)
  colors = get_box_colors(len(labels))
  return dict(zip(labels, colors))

BoxType = jax.Array

@partial(jax.jit, static_argnums=[2, 3, 4, 5])
def iou(
  box1: BoxType,
  box2: BoxType,
  format: str = "iou",
  scale: list | jax.Array = None,
  keepdim: bool = False,
  EPS: float = 1e-6,
):
  assert format in ["iou", "diou", "ciou"]
  if box1.ndim == 1:
    box1 = box1.reshape(1, -1)
  if box2.ndim == 1:
    box2 = box2.reshape(1, -1)
  assert box1.shape[-1] == box2.shape[-1]
  if box1.shape[-1] == 2:
    box1 = jnp.pad(box1, ((0, 0), (2, 0)))
    box2 = jnp.pad(box2, ((0, 0), (2, 0)))
  assert box1.shape[-1] == 4

  if scale is not None:
    if isinstance(scale, list):
      scale = jnp.array(scale)
    box1 *= scale
    box2 *= scale
  min1, min2 = box1[..., 0:2] - box1[..., 2:4] / 2, box2[..., 0:2] - box2[..., 2:4] / 2
  max1, max2 = box1[..., 0:2] + box1[..., 2:4] / 2, box2[..., 0:2] + box2[..., 2:4] / 2
  inter_h = (jnp.minimum(max1[..., 0], max2[..., 0]) - jnp.maximum(min1[..., 0], min2[..., 0])).clip(0.0)
  inter_w = (jnp.minimum(max1[..., 1], max2[..., 1]) - jnp.maximum(min1[..., 1], min2[..., 1])).clip(0.0)
  inter_size = inter_h * inter_w
  size1, size2 = jnp.prod(max1 - min1, axis=-1), jnp.prod(max2 - min2, axis=-1)
  union_size = size1 + size2 - inter_size
  result_iou = inter_size / (union_size + EPS)
  if format == "iou":
    ret = result_iou
    if keepdim:
      ret = ret[..., None]
    return ret

  outer_h = jnp.maximum(max1[..., 0], max2[..., 0]) - jnp.minimum(min1[..., 0], min2[..., 0])
  outer_w = jnp.maximum(max1[..., 1], max2[..., 1]) - jnp.minimum(min1[..., 1], min2[..., 1])
  center_dist = ((box1[..., :2] - box2[..., :2]) ** 2).sum(-1)
  diagonal_dist = jax.lax.stop_gradient(outer_h ** 2 + outer_w ** 2)
  result_diou = result_iou - center_dist / (diagonal_dist + EPS)
  if format == "diou":
    ret = result_diou
    if keepdim:
      ret = ret[..., None]
    return ret

  v = 4 / (jnp.pi ** 2) * (
    jnp.arctan(box1[..., 2] / (box1[..., 3] + EPS)) -
    jnp.arctan(box2[..., 2] / (box2[..., 3] + EPS))
  ) ** 2
  alpha = jax.lax.stop_gradient(v / (1 - result_iou + v))
  S = jax.lax.stop_gradient(result_iou >= 0.5)
  result_ciou = result_diou - S * alpha * v
  ret = result_ciou
  if keepdim:
    ret = ret[..., None]
  return ret

@partial(jax.jit, static_argnums=[2])
def iou_multiply(boxes1, boxes2, format="iou"):
  _, result = jax.lax.scan(
    f=lambda carry, x: (carry, iou(x, carry, format)),
    init=boxes2,
    xs=boxes1,
  )
  return result

@partial(jax.jit, static_argnums=[3, 4])
def nms(box, iou_threshold=0.3, conf_threshold=0.2, nms_multi=30, max_num_box=100, iou_format="iou"):
  M = min(max_num_box * nms_multi, box.shape[0])
  sort_idxs = jnp.argsort(-box[:, 4])[:M]
  box = box[sort_idxs]
  ious = iou_multiply(box[:, :4], box[:, :4], format=iou_format)
  mask = (box[:, 4] > conf_threshold) & (~jnp.diagonal(jnp.tri(M, k=-1) @ (ious > iou_threshold)).astype("bool"))
  idx = jnp.argwhere(mask, size=max_num_box, fill_value=-1)[:, 0]
  dbox = box[idx]
  pnum = (idx != -1).sum()
  return dbox, pnum
