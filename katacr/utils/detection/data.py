import numpy as np
from PIL import Image

def show_box(img, box, draw_center_point=False, verbose=True, format='yolo', use_overlay=True, num_state=1, show_conf=False, save_path=None, fontsize=12):
  from katacr.utils.detection import plot_box_PIL, build_label2colors
  from katacr.constants.label_list import idx2unit
  from katacr.constants.state_list import idx2state
  img = img.copy()
  if isinstance(img, np.ndarray):
    if img.max() <= 1.0:
      img *= 255
    img = Image.fromarray(img.astype('uint8'))
  if use_overlay:
    overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
  if len(box):
    label2color = build_label2colors(box[:, -1])
  for b in box:
    states = b[5:5+num_state] if len(b) == 7 else b[4:4+num_state]
    conf = float(b[4])
    label = int(b[-1])
    text = idx2unit[label]
    for i, s in enumerate(states):
      if i == 0:
        text += idx2state[int(s)]
      elif int(s) != 0:
        text += ' ' + idx2state[int(i*10 + s)]
    if show_conf:
      text += ' ' + f'{conf:.3f}'
    plot_box = lambda x: plot_box_PIL(
      x, b[:4],
      text=text,
      box_color=label2color[label],
      format=format, draw_center_point=draw_center_point,
      fontsize=fontsize
    )
    if use_overlay:
      overlay = plot_box(overlay)
    else:
      img = plot_box(img)
  if use_overlay:
    img = Image.alpha_composite(img.convert('RGBA'), overlay).convert('RGB')
  if verbose:
    img.show()
  if save_path is not None:
    img.save(str(save_path))
  return img
