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

## Human Players

To play a game between two human players, launch the server from within the running container.  Also, start the game:

```
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
examples/ac.py --human
```

Then, open two browser windows (one per human player) and navigate to:

```
http://myservername:8000
```

The terminal output will include the ID number for each human player.  Type the number into that player's browser then click `Connect`.  Top textbox gives player prompts; bottom textbox is where player enters response.  Textboxes turn blue while waiting for the next prompt.
