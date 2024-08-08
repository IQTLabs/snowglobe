# Snow Globe
***Open-Ended Wargames with Large Language Models***

Snow Globe, an ongoing applied research project and resulting software package, uses large language models (LLMs) for automated play of "open-ended" text-based wargames, such as seminar games and political wargames.  LLMs enable a light, flexible architecture in which player actions are not restricted to predefined options.  The system allows humans to play against or alongside AI agents with specific personas.  Every stage of the wargame from scenario preparation to post-game analysis can be optionally carried out by AI, humans, or a combination thereof.

## Setup

Build the Docker image and run a container.

```
./docker_setup.sh
```

## Demos

Once inside the running container (see Setup above), you can simulate a tabletop exercise about an AI incident response.

```
examples/haiwire.py
```

Or, simulate a political wargame about a geopolitical crisis.

```
examples/ac.py
```

In the latter case, you can use the chat interface to discuss the game afterwards, or just press `Enter` twice to exit.

## Human Players

To play a game between a human and an AI player, launch the server from within the running container.  Also, start the game:

```
uvicorn api:app --host 0.0.0.0 --port 8000 &
examples/ac.py --human 1
```

Then, open a browser window and navigate to:

```
http://myservername:8000
```

The terminal output will include the ID number for the human player.  Type the number into the browser then click `Connect`.  The top textbox gives player prompts; the bottom textbox is where the player enters responses.  Textboxes turn blue while waiting for the next prompt.

## License

This repo is released under the [Apache License Version 2.0](LICENSE), except for [jQuery](terminal/jquery-3.7.1.min.js) which is released under the [MIT License](https://github.com/jquery/jquery/blob/main/LICENSE.txt).
