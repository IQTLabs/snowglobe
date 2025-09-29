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

import langchain.prompts
import langchain.storage
import langchain.retrievers
import langchain_text_splitters
import langchain_core.documents
import langchain_community.llms
import langchain_community.embeddings
import langchain_community.document_loaders
import langchain_community.document_loaders.text
import langchain_community.document_loaders.html
import langchain.tools.retriever
import langchain_core.tools


def RAGTool(
    name,
    desc,
    llm,
    paths,
    doctype,
    chunk_size=None,
    chunk_overlap=None,
    count=None,
    level=1,
):
    # Loader
    loader_choices = {
        "text": langchain_community.document_loaders.text.TextLoader,
        "html": langchain_community.document_loaders.html.UnstructuredHTMLLoader,
        "xml": langchain_community.document_loaders.UnstructuredXMLLoader,
        "pdf": langchain_community.document_loaders.PyPDFLoader,
    }
    loader = loader_choices[doctype]

    # Docs & splits
    docs = [loader(path).load() for path in paths]
    docs = [doc for docsub in docs for doc in docsub]
    if chunk_size is not None:
        if chunk_overlap is None:
            chunk_overlap = 0
        splitter = langchain_text_splitters.RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        # if verbose >= 5:
        #     print("Doc count for %s before splitting: %i" % (name, len(docs)))
        docs = splitter.split_documents(docs)
        # if verbose >= 5:
        #     print("Doc count for %s after splitting: %i" % (name, len(docs)))

    # Vectors & retriever
    vectorstore = langchain_core.vectorstores.InMemoryVectorStore.from_documents(
        documents=docs, embedding=llm.embeddings
    )
    kwargs = {"search_kwargs": {"k": count}} if count is not None else {}
    retriever = vectorstore.as_retriever(**kwargs)

    # Tool
    tool = langchain.tools.retriever.create_retriever_tool(retriever, name, desc)
    tool.metadata = {"level": level}
    return tool
