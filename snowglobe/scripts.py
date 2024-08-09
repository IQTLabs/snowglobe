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
import yaml
import urllib
import urllib.request
import uvicorn
import platformdirs

def config(source='llamacpp', name='mistral-7b-openorca',
           url='https://huggingface.co/TheBloke/Mistral-7B-OpenOrca-GGUF/resolve/main/mistral-7b-openorca.Q5_K_M.gguf'
           ):
    cache_dir = platformdirs.user_cache_dir('snowglobe')
    config_dir = platformdirs.user_config_dir('snowglobe')
    model_file = os.path.basename(urllib.parse.urlparse(url).path)
    model_path = os.path.join(cache_dir, model_file)
    config_file = 'llms.yaml'
    config_path = os.path.join(config_dir, config_file)
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(config_dir, exist_ok=True)
    config_content = {'openai': {}, 'llamacpp': {}, 'huggingface': {}}
    config_content['openai']['gpt-3.5-turbo'] = ''
    config_content['openai']['gpt-4'] = ''
    config_content[source][name] = model_path
    if not os.path.exists(config_dir):
        os.makedirs(cofig_dir, exists_ok=True)
    with open(config_path, 'w') as config_file:
        yaml.dump(config_content, config_file,
                  default_flow_style=False, sort_keys=False)
    urllib.request.urlretrieve(url, model_path)

def server(host='0.0.0.0', port=8000, log_level='warning'):
    uvicorn.run('api:app', host=host, port=port, log_level=log_level)
