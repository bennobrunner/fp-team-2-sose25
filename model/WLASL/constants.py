import pathlib

DATA_DIR = pathlib.Path('./data')
VIDEOS_DIR = DATA_DIR / 'videos'
TRAIN_PATH = DATA_DIR / 'train'
VAL_PATH = DATA_DIR / 'val'
TEST_PATH = DATA_DIR / 'test'

TRAIN_PATH.mkdir(parents=True, exist_ok=True)
VAL_PATH.mkdir(parents=True, exist_ok=True)
TEST_PATH.mkdir(parents=True, exist_ok=True)

LABELS = [d.name for d in TRAIN_PATH.iterdir() if d.is_dir()]

WIDTH = 250
HEIGHT = 250

CLASSES = 3
