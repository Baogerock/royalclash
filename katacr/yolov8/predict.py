import cv2
import glob
import os
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

IMG_FORMATS = ['jpeg', 'jpg', 'png', 'webp']
VID_FORMATS = ['avi', 'gif', 'm4v', 'mkv', 'mp4', 'mpeg', 'mpg', 'wmv']

class ImageAndVideoLoader:
  def __init__(self, path: str | Sequence, video_interval=1, cvt_part2=False):
    self.video_interval = video_interval
    self.cvt_part2 = cvt_part2
    if isinstance(path, str) and Path(path).suffix == '.txt':
      path = [p for p in Path(path).read_text().splitlines() if p and p[0] != '#']
    files = []
    for p in sorted(path) if isinstance(path, (list, tuple)) else [path]:
      p = str(Path(p).resolve())
      if '*' in str(p):
        files.extend(sorted(glob.glob(p, recursive=True)))
      elif os.path.isdir(p):
        files.extend(sorted(glob.glob(os.path.join(p, '*.*'))))
      elif os.path.isfile(p):
        files.append(p)
      else:
        raise FileNotFoundError(f"{p} does not exists!")

    imgs = [x for x in files if x.split('.')[-1].lower() in IMG_FORMATS]
    vids = [x for x in files if x.split('.')[-1].lower() in VID_FORMATS]
    ni, nv = len(imgs), len(vids)
    self.n = ni + nv
    self.files = imgs + vids
    self.video_flag = [False] * ni + [True] * nv
    self.mode = 'image'
    if len(vids):
      self._new_video(vids[0])
    else:
      self.cap = None

  def _new_video(self, path):
    self.frame = 0
    self.cap = cv2.VideoCapture(path)
    self.total_frame = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) // self.video_interval

  def __len__(self):
    return self.n

  def __iter__(self):
    self.count = 0
    return self

  def __next__(self):
    if self.count == self.n:
      raise StopIteration
    path = self.files[self.count]

    if self.video_flag[self.count]:
      self.mode = 'video'
      for _ in range(self.video_interval):
        flag = self.cap.grab()
      flag, img = self.cap.retrieve()
      while not flag:
        self.count += 1
        self.cap.release()
        if self.count >= self.n:
          raise StopIteration
        path = self.files[self.count]
        self._new_video(path)
        for _ in range(self.video_interval):
          flag = self.cap.grab()
        flag, img = self.cap.retrieve()
      img = img[..., ::-1]
      self.frame += 1
      s = f"video {self.count+1}/{self.n} ({self.frame}/{self.total_frame}) {path}:"
    else:
      self.count += 1
      img = np.array(Image.open(path).convert("RGB"))
      s = f"image {self.count}/{self.n} {path}:"

    img = np.ascontiguousarray(img[..., ::-1])
    return path, img, self.cap, s
