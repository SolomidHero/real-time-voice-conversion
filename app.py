from pathlib import Path
from toolbox import Toolbox
import argparse
from pathlib import Path


if __name__ == '__main__':
  parser = argparse.ArgumentParser(
    description="Runs the toolbox",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
  )
  parser.add_argument("-d", "--datasets_root", type=Path, help= \
    "Path to the directory containing your datasets.", default=Path(__file__).parent / 'datasets')
  parser.add_argument("--seed", type=int, default=17, help=\
    "Optional random number seed value to make toolbox deterministic.")
  args = parser.parse_args()

  # Launch the toolbox
  Toolbox(**vars(args))