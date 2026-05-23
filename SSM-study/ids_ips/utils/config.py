"""
Central configuration for the IDS/IPS pipeline.

Path resolution — works on any machine / cloud environment:
  BASE_DIR is derived from this file's location at runtime, so no
  hardcoded absolute paths. Override any path with environment variables:

  MS_PROJECT_DIR  — override the project root
  MS_TRAIN_CSV    — override path to training CSV
  MS_TEST_CSV     — override path to test CSV
  MS_CKPT_DIR     — override checkpoint directory
  MS_RESULTS_DIR  — override results directory

GCS example (Vertex AI or gcsfuse-mounted bucket):
  export MS_TRAIN_CSV=/gcs/vks_bucket/MS_PROJECT/UNSW_NB15_training-set.csv
  export MS_TEST_CSV=/gcs/vks_bucket/MS_PROJECT/UNSW_NB15_testing-set.csv
  export MS_CKPT_DIR=/gcs/vks_bucket/MS_PROJECT/checkpoints
  export MS_RESULTS_DIR=/gcs/vks_bucket/MS_PROJECT/results
"""
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

BASE_DIR: Path = Path(
    os.environ.get("MS_PROJECT_DIR",
                   Path(__file__).resolve().parent.parent.parent)
)

TRAIN_CSV: Path = Path(
    os.environ.get("MS_TRAIN_CSV",
                   BASE_DIR / "UNSW_NB15_training-set.csv")
)
TEST_CSV: Path = Path(
    os.environ.get("MS_TEST_CSV",
                   BASE_DIR / "UNSW_NB15_testing-set.csv")
)

CKPT_DIR: Path = Path(
    os.environ.get("MS_CKPT_DIR", BASE_DIR / "checkpoints")
)
RESULTS_DIR: Path = Path(
    os.environ.get("MS_RESULTS_DIR", BASE_DIR / "results")
)

ATTACK_CATEGORIES: List[str] = [
    "Normal",
    "Generic",
    "Exploits",
    "Fuzzers",
    "DoS",
    "Reconnaissance",
    "Analysis",
    "Backdoor",
    "Shellcode",
    "Worms",
]
NUM_CLASSES = len(ATTACK_CATEGORIES)

CAT_COLS  = ["proto", "service", "state"]
DROP_COLS = ["id", "attack_cat", "label"]

SEQ_LEN: Optional[int] = None

@dataclass
class MambaConfig:
    d_model:     int   = 256
    n_layers:    int   = 4
    d_state:     int   = 16
    d_conv:      int   = 4
    expand:      int   = 2
    dropout:     float = 0.1
    num_classes: int   = NUM_CLASSES

@dataclass
class RWKVConfig:
    d_model:     int   = 256
    n_layers:    int   = 4
    dropout:     float = 0.1
    num_classes: int   = NUM_CLASSES

@dataclass
class HybridConfig:
    d_model:     int   = 256
    n_layers:    int   = 4
    d_state:     int   = 16
    d_conv:      int   = 4
    expand:      int   = 2
    dropout:     float = 0.1
    num_classes: int   = NUM_CLASSES

@dataclass
class TransformerConfig:
    d_model:     int   = 256
    n_heads:     int   = 4
    n_layers:    int   = 4
    ff_dim:      int   = 512
    dropout:     float = 0.1
    num_classes: int   = NUM_CLASSES

@dataclass
class LSTMConfig:
    hidden_size:   int   = 128
    n_layers:      int   = 2
    dropout:       float = 0.1
    bidirectional: bool  = True
    num_classes:   int   = NUM_CLASSES

@dataclass
class TrainConfig:
    epochs:       int   = 30
    batch_size:   int   = 512
    lr:           float = 3e-4
    weight_decay: float = 1e-4
    focal_gamma:  float = 2.0
    grad_clip:    float = 1.0
    use_amp:      bool  = True
    seed:         int   = 42
    n_folds:      int   = 5
    save_dir:     Path  = CKPT_DIR
    results_dir:  Path  = RESULTS_DIR

IPS_CONFIDENCE_THRESHOLD: float = 0.85
IPS_RATE_LIMIT_THRESHOLD: float = 0.70
