import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[2]))

from ultralytics.engine.model import Model
from katacr.yolov8.custom_model import CRDetectionModel
from katacr.yolov8.custom_predict import CRDetectionPredictor

class YOLO_CR(Model):
  @property
  def task_map(self):
    return {
      "detect": {
        "model": CRDetectionModel,
        "trainer": None,
        "validator": None,
        "predictor": CRDetectionPredictor,
      },
    }
