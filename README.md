# Snow Globe
***Open-Ended Wargames with Large Language Models***

Snow Globe, an ongoing applied research project and resulting software package, uses large language models (LLMs) for automated play of "open-ended" text-based wargames, such as seminar games and political wargames.  LLMs enable a light, flexible architecture in which player actions are not restricted to predefined options.  The system allows humans to play against or alongside AI agents with specific personas.  Every stage of the wargame from scenario preparation to post-game analysis can be optionally carried out by AI, humans, or a combination thereof.

## Setup

Build the Docker image and run a container.

```
./docker_setup.sh
```

## Demos

Once inside the running container (see Setup above), go to the examples folder:

```
cd examples
```

Then, simulate a tabletop exercise about an AI incident response.

```
./haiwire.py
```

Or, simulate a political wargame about a geopolitical crisis.

```
./ac.py
```
