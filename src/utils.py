from typing import List

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from dotenv import load_dotenv
from pprint import pprint
from time import time

load_dotenv()


def get_llm(
    azure_deployment="gpt-4o",
    api_version="2024-06-01",
    temperature=0,
    timeout=600,
    max_tokens=4096,
    max_retries=2,
):
    return AzureChatOpenAI(
        azure_deployment=azure_deployment,
        api_version=api_version,
        temperature=temperature,
        timeout=timeout,
        max_tokens=max_tokens,
        max_retries=max_retries,
    )


def get_embedder(
    openai_api_version="2023-05-15", azure_deployment="text-embedding-3-small"
):
    embeddings = AzureOpenAIEmbeddings(
        openai_api_version=openai_api_version, azure_deployment=azure_deployment
    )
    return embeddings


# create a time decorator


def timeit(func):
    """
    Decorator to measure the execution time of a function.
    """

    def wrapper(*args, **kwargs):
        start_time = time()
        result = func(*args, **kwargs)
        end_time = time()
        print(f"Execution time: {end_time - start_time:.2f} seconds")
        return result

    return wrapper
