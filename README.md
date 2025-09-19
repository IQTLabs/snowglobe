# Snow Globe
***Open-Ended Wargames with Large Language Models***

Snow Globe, an ongoing applied research project and resulting software package, uses large language models (LLMs) for automated play of "open-ended" text-based wargames, such as seminar games and political wargames.  LLMs enable a light, flexible architecture in which player actions are not restricted to predefined options.  The system allows humans to play against or alongside AI agents with specific personas.  Every stage of the wargame from scenario preparation to post-game analysis can be optionally carried out by AI, humans, or a combination thereof.

Read more [here](https://arxiv.org/abs/2404.11446).

## Installation

Build the Docker image and run a container.

```
./docker_setup.sh
```

Or, install Snow Globe from PyPI.  For CPU only:

```
pip install llm-snowglobe
```

For GPU support:

```
CMAKE_ARGS="-DGGML_CUDA=on" pip install llm-snowglobe
```

## AI Simulations

After installation, you can simulate a tabletop exercise about an AI incident response.

```
examples/haiwire.py
```

Or, simulate a political wargame about a geopolitical crisis.

```
examples/ac_sim.py
```

## Human+AI Wargames

To play a game between a human and an AI player, launch the server and start a game:

```
snowglobe_server &
examples/ac_game.py
```

Then, open a browser window and navigate to:

```
http://localhost:8000
```

The terminal output will begin with the ID number for the human player.  Type that number into the ID box in the browser window and click `Log In`.

Make sure to run `snowglobe_server` from the same file system location where you run the game.  Game-related files will be stored in that location.

## Custom Games

By default, Snow Globe uses a light LLM that runs locally.  For better results, try the OpenAI API:
- Create an environment variable called `OPENAI_API_KEY` with your OpenAI API key in it.  In a bash shell, for example: `export OPENAI_API_KEY=your0key0here`
- In the example game you want to use with OpenAI, find the line that says `super().__init__()` and change it to this: `super().__init__(source='openai', model='gpt-4o')` or similarly for whatever model you want to try.

Beyond that, you can use the examples as starting points for developing your own games, or you can create entirely new game formats using the Python classes provided by Snow Globe.

## License

This repo is released under the [Apache License Version 2.0](LICENSE), except for [jQuery](src/llm_snowglobe/terminal/jquery-3.7.1.min.js) which is released under the [MIT License](https://github.com/jquery/jquery/blob/main/LICENSE.txt).
