#!/usr/bin/env python3

#   Copyright 2023-2025 IQT Labs LLC
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
import platformdirs
import random
import sys
import transformers
import urllib.parse
import urllib.request
import yaml

import langchain_openai
import langchain_huggingface
import langchain_community.llms
import langchain_community.embeddings

import langchain_community.document_loaders
import langchain_community.document_loaders.text
import langchain_community.document_loaders.html

def settings():
    s = {}
    # Standard paths
    s["config_dir"] = platformdirs.user_config_dir("snowglobe")
    s["cache_dir"] = platformdirs.user_cache_dir("snowglobe")
    s["data_dir"] = platformdirs.user_data_dir("snowglobe", "snowglobe")
    s["menu_file"] = "llms.yaml"
    s["menu_path"] = os.path.join(s["config_dir"], s["menu_file"])

    # Default model
    s["default_source"] = "llamacpp"
    s["default_model"] = "mistral-7b-openorca"
    s["default_url"] = (
        "https://huggingface.co/TheBloke/Mistral-7B-OpenOrca-GGUF/resolve/main/mistral-7b-openorca.Q5_K_M.gguf"
    )
    s["default_file"] = os.path.basename(urllib.parse.urlparse(s["default_url"]).path)
    s["default_path"] = os.path.join(s["cache_dir"], s["default_file"])
    return s


def read_yaml(path):
    # Reads YAML file, expressing empty or nonexistent file as empty dictionary
    if os.path.exists(path):
        with open(path, "r") as obj:
            content = yaml.safe_load(obj)
        if content is None:
            content = {}
    else:
        content = {}
    return content


def write_yaml(data, path):
    # Writes YAML file, creating empty directory if needed
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as obj:
        yaml.dump(data, obj, default_flow_style=False, sort_keys=False)


def config(
    menu=None,
    source=None,
    model=None,
    url=None,
    path=None,
    update_menu=True,
    download_weights=True,
):
    s = settings()
    menu = menu if menu is not None else s["menu_path"]
    source = source if source is not None else s["default_source"]
    model = model if model is not None else s["default_model"]
    url = url if url is not None else s["default_url"]
    path = path if path is not None else s["default_path"]

    # Update menu of source+model options
    if update_menu:
        options = read_yaml(menu)
        if source not in options:
            options[source] = {}
        options[source][model] = path
        write_yaml(options, menu)

    # Download model weights
    if download_weights:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print("Downloading model weights... ", end="", flush=True)
        urllib.request.urlretrieve(url, path)
        print("Done", flush=True)

class LLM:
    def __init__(self, source=None, model=None, menu=None, gen=None, embed=None, verbosity=2):

        s = settings()
        self.menu = menu if menu is not None else s["menu_path"]
        self.source = source if source is not None else s["default_source"]
        self.model = model if model is not None else s["default_model"]
        self.gen = gen if gen is not None else True
        self.embed = embed if embed is not None else False
        self.verbosity = verbosity

        if self.source not in ["openai", "azure"]:
            options = read_yaml(self.menu)
            # When using the default model and standard menu path,
            # if the model is not found then auto-install it.
            if (
                (self.source not in options or self.model not in options[self.source])
                and self.source == s["default_source"]
                and self.model == s["default_model"]
                and self.menu == s["menu_path"]
            ):
                config()
                options = read_yaml(self.menu)
            self.model_path = options[self.source][self.model]
            self.model_path = os.path.expanduser(self.model_path)
            if not os.path.isabs(self.model_path):
                self.model_path = os.path.join(self.menu, self.model_path)
        elif self.source in ["azure"]:
            options = read_yaml(self.menu)
            self.azure_deployment = options[self.source][self.model]["deployment"]
            self.azure_endpoint = options[self.source][self.model]["endpoint"]
            self.azure_version = options[self.source][self.model]["version"]

        if self.source == "openai":

            # Model Source: OpenAI (Cloud)
            if self.gen:
                self.llm = langchain_openai.ChatOpenAI(
                    model_name=self.model,
                    streaming=True,
                )
            if self.embed:
                self.embeddings = langchain_openai.OpenAIEmbeddings()

        elif self.source == "azure":

            # Model Source: Azure OpenAI (Cloud)
            if self.gen:
                self.llm = langchain_openai.AzureChatOpenAI(
                    azure_deployment=self.azure_deployment,
                    azure_endpoint=self.azure_endpoint,
                    api_version=self.azure_version,
                    streaming=True,
                )
            if self.embed:
                self.embeddings = langchain_openai.AzureOpenAIEmbeddings(
                    azure_deployment=self.azure_deployment,
                    azure_endpoint=self.azure_endpoint,
                    api_version=self.azure_version,
                )

        elif self.source == "llamacpp":

            # Model Source: llama.cpp (Local)
            if self.gen:
                self.llm = langchain_community.llms.LlamaCpp(
                    model_path=self.model_path,
                    n_gpu_layers=-1,
                    seed=random.randint(0, sys.maxsize),
                    n_ctx=32768,
                    n_batch=512,
                    f16_kv=True,
                    max_tokens=1000,
                    verbose=False,
                )
            if self.embed:
                self.embeddings = langchain_community.embeddings.LlamaCppEmbeddings(
                    model_path=self.model_path,
                    n_gpu_layers=-1,
                    n_batch=512,
                    n_ctx=8192,
                    f16_kv=True,
                    verbose=False,
                )
            self.serial = True

        elif self.source == "huggingface":

            # Model Source: Hugging Face (Local)
            if self.gen:
                model = transformers.AutoModelForCausalLM.from_pretrained(
                    self.model_path, device_map="auto"
                )
                tokenizer = transformers.AutoTokenizer.from_pretrained(
                    self.model_path, device_map="auto"
                )
                tokenizer.pad_token = tokenizer.eos_token
                streamer = (
                    transformers.TextStreamer(
                        tokenizer, skip_prompt=True, skip_special_tokens=True
                    )
                    if self.verbosity >= 1
                    else None
                )
                pipeline = transformers.pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    device_map="auto",
                    max_new_tokens=2048,
                    repetition_penalty=1.05,
                    return_full_text=False,
                    streamer=streamer,
                    do_sample=True,
                )
                self.llm = langchain_huggingface.llms.HuggingFacePipeline(
                    pipeline=pipeline
                )
            if self.embed:
                self.embeddings = (
                    langchain_huggingface.embeddings.HuggingFaceEmbeddings(
                        model_name=self.model_path, show_progress=True
                    )
                )

        self.bound = {}
