#! /bin/sh
snowglobe_config && \
snowglobe_simulation -c /home/snowglobe/.config/snowglobe/wotr_unmasked.yaml \
-l /home/snowglobe/logs/local/snowglobe.log --runs 5 \
--simulation-name WotR-Sim --simulation-mode