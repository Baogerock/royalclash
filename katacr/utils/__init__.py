import contextlib
import time

class Stopwatch(contextlib.ContextDecorator):
  def __init__(self, t=0.0):
    self.t = t
    self.dt = 0

  def __enter__(self):
    self.start = time.time()
    return self

  def __exit__(self, *args):
    self.dt = time.time() - self.start
    self.t += self.dt

def second2str(second):
  s = int(second)
  m = int(second // 60)
  h = int(m // 60)
  ret = ""
  if h:
    ret += f"{h:02}:"
    m = int(m % 60)
  s = int(s % 60)
  ret += f"{m:02}:{s:02}"
  return ret
