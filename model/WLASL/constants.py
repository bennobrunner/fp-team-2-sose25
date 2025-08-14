import pathlib

DATA_DIR = pathlib.Path('./data')
VIDEOS_DIR = DATA_DIR / 'videos'
PROCESSED_VIDEOS_DIR = VIDEOS_DIR / 'processed'

WIDTH = 200
HEIGHT = 250

# TODO: Klassen automatisch auslesen
CLASSES = [
  'book',
  'drink',
  'computer',
  # 'before',
  # 'chair',
  # 'go',
  # 'clothes',
  # 'who',
  # 'candy',
  # 'cousin'
]
