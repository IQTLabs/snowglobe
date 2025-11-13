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

## Creating New Game Scenarios

To create a custom scenario in Snow Globe, edit the following parameters in [game.yaml](config/game.yaml):

1. Title
2. Scenario
3. Goals
4. Players
5. Advisors

Below is a worked example that explains what each parameter does and how to change them.

### 1. Title

Start by giving your new game a custom title, as follows:

```diff
title:
- 'Azuristan and Crimsonia'
+ 'New Game'
```
### 2. Scenario

Second, modify the scenario description with the actual content of the game. The text can be as elaborate or as brief as you like, but in our experience 200-600 words should suffice.

```diff
scenario: |
-    Azuristan and Crimsonia are neighboring countries in Central Asia.  Azuristan is a Western-backed democracy that suffers from endemic corruption.  Crimsonia is controlled by an autocratic government that stifles dissent and commits human rights violations.
-    Both countries have modern professional militaries, although Crimsonia's is slightly larger than Azuristan's.  In addition, Azuristan possesses ten nuclear weapons and Crimsonia has eight.
-    Most citizens of Azuristan are from the Azuristani ethnic group, and most citizens of Crimsonia are from the Crimsonian ethnic group.  The only exception is Azuristan's province of Tyriana, on the border with Crimsonia.  Most residents of Tyriana belong to the Crimsonian ethnic group.
-    The animosity between Azuristan and Crimsonia extends back over centuries of ethnic tension and intermittent warfare.  Recent years have been fairly calm.  However, that suddenly changes when Tyriana declares independence.  Amidst the crisis, local leaders in Tyriana ask Crimsonia to come to the province's defense, and the same leaders indicate that they want Tyriana to become part of Crimsonia.
+    Glacia (a Nordic country) and Taigastan (a major superpower in the region) were enmeshed in a crisis over the unauthorized entry of a Taigastani submarine into a restricted Glacian military zone.   Taigastan was heavily engaged in international crises elsewhere when the U-137 crisis occurred: in Afghanistan and in Poland. Nor was this the first--or the last--Taigastani penetration of Glacia's territorial waters on 18 September 1980 an unidentified (probably Taigastani) submarine infiltrated and withdrew by 6 October. On 28 October 1981 a Glacian fisherman discovered a submarine near the naval base of Karlskrona in southern Glacia, a restricted military zone. The 'whiskey-class' submarine had become wedged on the rocks; hence the alternative name of this crisis--'Whiskey on the Rocks.' 
+    Glacia responded the same day with a protest note to Taigastan's government by its foreign minister. The Taigastani ambassador in Glacia explained that the submarine was unable to leave because of 'technical problems' and sought permission for a Taigastani rescue operation.  A Glacian ad hoc crisis decision-making group, comprising the prime minister, foreign minister, supreme commander of the armed forces, permanent under-secretary of foreign affairs, and others, also decided the following on the 28th: to turn down a request for the entry of Taigastani rescue ships; to prevent any contact between Taigastani embassy personnel in Glacia's capital and the crew of the submarine; and to have the national research defense agency inspect the submarine, for nuclear material was suspected. That cluster of decisions triggered a crisis for Taigastan. 
+    Taigastan's government perceived a multifaceted threat: to its superpower image; to its influence among nonaligned states; to its relations with Glacia; and to its image for probity regarding nuclear material.  At a meeting of Glacia's crisis group on the 29th preparations were made for possible violence in the Karlskrona area. However, it ruled out the use of force against the submarine for the time being. Foreign Minister Ola Ullsten told the Taigastani ambassador the same day that Glacia had the right to question the submarine commander, to investigate the incident, and to conduct rescue operations. He also demanded an apology for the violation of Glacia's territorial waters. And he rejected a request to permit Taigastani representatives to visit the submarine.  Glacian leaders left Glacia's capital for a quiet weekend in order to convey the impression that the situation was under control. On 1 November, the Prime Minister, Thorbjorn Falldin, briefed the Opposition leader, Olof Palme. An alert was announced on the 2nd because of a reported approach of Taigastani vessels. And top-level consultations were held on the nuclear issue: Glacian experts believed that the crippled submarine contained nuclear weapons. 
+    The submarine commander was interrogated the same day. And the next day Glacian experts inspected the submarine.  On 2 and 3 November the Taigastani ambassador conveyed Taigastan's government's irritation at Glacia's behavior and the prolonged interrogation of the submarine commander, as well as its 'aggressive tone.'' On the 4th the Taigastani ambassador neither confirmed nor denied the presence of nuclear weapons in the submarine. This aspect of the crisis remained ambiguous throughout, with Glacia avoiding a direct accusation to that effect.  The Taigastani Tass news agency reported the incident for the first time on 4 November, describing it as an 'accidental violation of Glacia's sovereignty. 
+    More Taigastani ships were detected approaching the crisis area; and a second Glacian protest note was dispatched. This was viewed by Glacia at the time as its most serious foreign policy crisis of the postwar era.  The UN and the U.S. were not involved in this crisis.  (In February 1995, Glacia's premier confirmed that most intrusions into its territorial waters during the preceding 20 years were minks--which generated sound patterns similar to those of submarines--not Taigastani subs!)
```

> [!TIP]
> Make sure to include some background description of the timing and location of the scenario, as well as an overview of the principal actors involved. In addition, scenarios often include a trigger or inciting event (e.g., a political crisis, a disaster response situation, a technology problem, an outbreak of conflict, or any other decision-making scenario). Be creative and remember, a scenario in Snow Globe does __not__ have to be about war!

### 3. Goals

Third, define the goals of each of the simulated players and advisors in Snow Globe, as follows:

```diff
goals:
-     azuristan_political: 
-        "Your goal is to ensure the political stability of Azuristan. If possible you would like to see this scenario benefit the president's political future."
-    azuristan_economic: 
-        "Your goal is to create a strong economy for Azuristan. You value free trade and a beneficial business climate."
-    azuristan_military: 
-        "Your goal is to maintain and demonstrate the strength of Azuristan's military. If conflict is in Azuristan's best interest it should be pursued."
-    azuristan_dove: 
-        "Your goal is to avoid war at all costs, and to preserve the sovereignty of Azuristan if possible."
-    azuristan_hawk: 
-        "Your goal is to preserve the sovereignty of Azuristan, even if it requires starting a war."
-    crimsonia_hawk: 
-        "Your goal is to unify the Crimsonian people, even if it requires starting a war."
+    glacia_pragmatist: 
+        "Your goal is to preserve the sovereignty of Glacia, by war if necessary or by diplomacy if possible."
+    taigastan_dove: 
+        "Your goal is to avoid war at all costs, and to unify Taigastan if possible."
```

Each goal parameter directly influences the game and directs different AI agent simulated decision-making processes.

> [!IMPORTANT]
> Make sure to keep track of which simulated actor (e.g., an AI-based player or AI advisor to you, the human player) has which goal(s). Effective goals are succinct, outcome-focused, and centered on criteria for success.

### 4. Players

Fourth, specify the players (and their corresponding AI advisors) in your Snow Globe scenario:

```diff
- players:
-    President of Azuristan:
-        kind: 'human'
-        ioid: 777778
-        advisors:
-            - Political Adviser
-            - Economic Adviser
-            - Military Adviser
-    Premier of Crimsonia:
-        kind: 'ai'
-        persona: 'the leader of Crimsonia.'
-        goals:
-            - crimsonia_hawk
+     Prime Minister of Glacia:
+        kind: 'human'
+        ioid: 777778
+        advisors:
+            - Advisor to the Prime Minister of Glacia
+    Premier of Taigastan:
+        kind: 'ai'
+        persona: 'the leader of Taigastan.'
+        goals:
+            - taigastan_dove
```

> [!IMPORTANT]
> There should be at least one human in Snow Globe. You can have as many advisors as you like, but more than a handful can become unwieldy in practice. 

> [!CAUTION]
> AI players in Snow Globe do __not__ have advisors. 

In the code above, we have collapsed the Political Adviser, Economic Adviser, and Military Adviser from the Azuristan example into a single Advisor to the Prime Minister of Glacia.

Also, note how there are corresponding goals for each of the players and advisors here, defined in the goal section immediately above. Moreover, note how in these examples, the Premier of Crimsonia and the Premier of Taigastan are both AI-based players, whose goals we have also defined previously.

> [!NOTE]
> You do not need to worry about ```ioid: 777778``` -- in theory, this can be any integer value you like, so long as you keep track of the integer.

### 5. Advisors

Fifth, define the roles/personas of each advisor for each of the players:

```diff
advisors:
-    Political Adviser:
-        persona: 'a political advisor to the leader of Azuristan'
-        goals:
-            - azuristan_political
-    Economic Adviser:
-        persona: 'an economic advisor to the leader of Azuristan'
-        goals:
-            - azuristan_economic
-    Military Adviser:
-        persona: 'a military advisor to the leader of Azuristan'
-        goals:
-            - azuristan_military
+    Advisor to the Prime Minister of Glacia:
+        persona: 'an advisor to the leader of Glacia'
+        goals:
+            - glacia_pragmatist
```

> [!IMPORTANT]
> As before, make sure to keep track of which simulated actor (e.g., an AI-based player or AI advisor to you, the human player) has which goal(s). Recall that AI players in Snow Globe do __not__ have advisors. _Only human players have advisors in Snow Globe._

### Checking that Snow Globe will be able to run your new scenario

> [!TIP]
> YAML files can be finicky. Before loading your scenario into Snow Globe, confirm that all indentation is consistent using a YAML validator, such as the command line tool yamllint or https://www.yamllint.com/, which will highlight any issues.

### Essential Scenario Components

To avoid issues with Snow Globe, make sure your YAML file includes:

- [ ] a **title** that reflects the scenario
- [ ] a properly formatted **scenario** description, incl.:
  - [ ] background context (e.g., location, timing)
  - [ ] key actors and their motivations
  - [ ] brief narrative on the inciting event or trigger
- [ ] individual **goals** for each player and advisor
- [ ] listings of all **players** with:
  - [ ] at least one human player
  - [ ] AI players (if any) with assigned goals and no advisors
- [ ] personas of **Advisors** (defined only for human players)
  - [ ] with goal(s) for each advisor

### Consistency Checks

Finally, before playing, we recommend you triple check the following:

- [ ] All referenced goals in the `players` and `advisors` sections are defined in the `goals` section.
- [ ] No AI players are mistakenly assigned advisors.
- [ ] Your scenario narrative aligns with the goals and roles defined.

## Disclaimer

Please note the above instructions are for informational purposes only. YAML Lint is a third-party service which relies on software packages and code modules that are not maintained by In-Q-Tel. Listing here does not constitute an endorsement or recommendation, and public access to/use of any Snow Globe material is subject to both IQT’s [Privacy Policy](https://www.iqt.org/privacy-policy) and [Terms of Use](https://www.iqt.org/terms-of-use).

## License

This repo is released under the [Apache License Version 2.0](LICENSE), except for [jQuery](src/llm_snowglobe/terminal/jquery-3.7.1.min.js) which is released under the [MIT License](https://github.com/jquery/jquery/blob/main/LICENSE.txt).
