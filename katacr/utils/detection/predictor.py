from katacr.utils.related_pkgs.utility import *
from katacr.utils.related_pkgs.jax_flax_optax_orbax import *
from katacr.utils.detection import nms

class BasePredictor:
  """
  Minimal predictor wrapper for inference.
  Subclasses should implement `predict`.
  """

  def __init__(self, state: train_state.TrainState, image_shape: Tuple = (896, 576, 3)):
    self.state = state
    self.image_shape = image_shape

  def predict(self, state: train_state.TrainState, x: jax.Array):
    """
    Subclass implement. Return shape=(N,num_pbox,7) elem:(x,y,w,h,conf,side,cls)
    """
    raise NotImplementedError

  @partial(jax.jit, static_argnums=[0])
  def pred_bounding_check(self, pbox):
    x1 = jnp.maximum(pbox[..., 0] - pbox[..., 2] / 2, 0)
    y1 = jnp.maximum(pbox[..., 1] - pbox[..., 3] / 2, 0)
    x2 = jnp.minimum(pbox[..., 0] + pbox[..., 2] / 2, self.image_shape[1])
    y2 = jnp.minimum(pbox[..., 1] + pbox[..., 3] / 2, self.image_shape[0])
    w, h = x2 - x1, y2 - y1
    return jnp.concatenate([jnp.stack([x1 + w / 2, y1 + h / 2, w, h], -1), pbox[..., 4:]], -1)

  @partial(jax.jit, static_argnums=[0, 3, 4, 5])
  def pred_and_nms(self, state: train_state.TrainState, x: jax.Array, iou_threshold: float, conf_threshold: float, nms_multi: float = 30):
    pbox = self.predict(state, x)
    pbox = self.pred_bounding_check(pbox)
    pbox, pnum = jax.vmap(
      nms, in_axes=[0, None, None, None], out_axes=0
    )(pbox, iou_threshold, conf_threshold, nms_multi)
    return pbox, pnum
