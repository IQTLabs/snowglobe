#! /bin/sh
snowglobe_config && \
snowglobe_simulation -c /home/snowglobe/.config/snowglobe/wotr_unmasked.yaml \
-l /home/snowglobe/logs/local/snowglobe.log --runs 3 \
--simulation-name WotR-Sim --simulation-mode