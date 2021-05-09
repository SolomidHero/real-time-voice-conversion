# Toolbox package building guide

We use OS dependent utilite PyInstaller as package building tool. Result of steps below is a platform dependent for single application. It can be used directly on same platform as guide instructions are completed. For other platforms build on virtual machines or containers. [Read more about](https://pyinstaller.readthedocs.io/en/stable/usage.html) build tool.

**Note:** using of python virtual environment is highly recommended:
```
# creation
virtualenv ~/install_env
source ~/install_env/bin/activate

# after use (delete if needed)
deactivate
rm -rf ~/install_env
```

The procedure of distribution of this repo toolbox consists of several steps (steps performed from `dist/` directory).

## MacOS

1. Install all required packages from requirements.txt.
```bash
pip -r ../requirements.txt
```

2. Trying first build which will definetely fail, but provide us with important warning log.
```bash
pyinstaller --name="VCToolbox" --windowed --add-data="../config.yaml:./" --add-data="../datasets/*:datasets/" --hidden-import=typing_extensions -y --onefile ../app.py
```

**Note:** `--onefile` is optional and not fully supported option with PySide6 (main Qt for Python package used in toolbox).

After command above, there would be two new directories ('build/' and 'dist/'). Actually we don't need 'dist/' folder, because it stores our releasing app, but generated 'build/VCToolbox/warn-VCToolbox.txt' will be used for next stages.

3. Define installation hooks for not found modules. For this run `warn_processing.py` file:
```bash
python3 warn_preprocessing.py
```

This stage is cumbersome and depends on previous one. Some important modules that weren't found (such as librosa, etc) already added in script. After the command, there will be generated `hooks/` directory with files defining hooks for PyInstaller. Tree will look like:

```
.
├── BUILD.md
├── VCToolbox.spec
├── build
├── dist
├── hooks
└── warn_processing.py
```

4. Build with hooks for not found packages:
```bash
pyinstaller --name="VCToolbox" --windowed --hidden-import=typing_extensions -y --additional-hooks-dir=hooks --onefile ../app.py
```

If everything done right, there would be an executable file in dist/ folder.

5.(optional) For distribution purposes move app binary into root folder, because it uses `config.yaml` and `datasets/` pathes. Then zip into archive, or use other utility for `.app` and `.dmg` creation.

