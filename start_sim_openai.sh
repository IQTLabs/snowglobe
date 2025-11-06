#! /bin/sh
# snowglobe_config && \
env AZURE_OPENAI_API_KEY="$(cat /run/secrets/openai_key)" snowglobe_simulation \
-c /home/snowglobe/.config/snowglobe/wotr_unmasked-openai.yaml \
-l /home/snowglobe/logs/openai/snowglobe.log --runs 20 \
--simulation-name WotR-Sim --simulation-mode