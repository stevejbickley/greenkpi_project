# GreenKPI Project

A simple, proof-of-concept project that utilizes OpenAI's Assistants and/or Chat Completions API to develop the "GreenKPI Assistant" for https://www.greenkpi.com/.  

## Preparation
 
### Navigate to the folder

```cd ./greenkpi_project```

### Now run the following commands:

```poetry env use path_to_pyevn_python_version```

If you are not using pyenv, just replace the above command with:

```poetry env use path_to_python_interpreter```

Activate the virtual environment if needed:

```source ./.venv/bin/activate```

If using windows - to activate environment, run the following:

```poetry shell```

### Next, run the following to update poetry and install ffmpeg (if required):

```poetry update```

Macos
```brew install ffmpeg```

Linux
```apt install ffmpeg```

Running Python and Scripts:

```poetry run python script.py```

Note: ffmpeg is open-source suite of libraries and programs for handling video, audio, and other multimedia files and streams


## Creating your own poetry environment


### Initialize the existing directory (if required):

```poetry init```


### To add a new package to your project, use:

```poetry add package-name```



