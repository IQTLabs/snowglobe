#!/usr/bin/env python3

#   Copyright 2024 IQT Labs LLC
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import os
import uvicorn
import platformdirs

def snowglobe_config(source='llamacpp', name='mistral-7b-openorca',
                     url='https://huggingface.co/TheBloke/Mistral-7B-OpenOrca-GGUF/resolve/main/mistral-7b-openorca.Q5_K_M.gguf'
                     ):
    cache_dir = platformdirs.user_cache_dir('snowglobe')
    config_dir = platformdirs.user_config_dir('snowglobe')
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(config_dir, exist_ok=True)
    print(cache_dir)
    print(config_dir)

def snowglobe_server(host='0.0.0.0', port=8000, log_level='warning'):
    uvicorn.run('api:app', host=host, port=port, log_level=log_level)
