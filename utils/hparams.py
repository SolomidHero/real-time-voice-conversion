from omegaconf import OmegaConf
from pathlib import Path
import __main__


root_dir = Path(__file__).parent.parent.resolve()
cfg = OmegaConf.load(root_dir / 'config.yaml')
cfg.root_dir = str(root_dir.resolve())