#! /bin/sh
# snowglobe_config;
env AZURE_OPENAI_API_KEY="$(cat /run/secrets/openai_key)" snowglobe_simulation -c /home/snowglobe/.config/snowglobe/WoTRM.yaml \
-l /home/snowglobe/logs/snowglobe.log --runs 3 \
--simulation-name WotR-Sim --simulation-mode