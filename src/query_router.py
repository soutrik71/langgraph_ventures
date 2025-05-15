"""
Query router module
This module provides functionality to route queries to the appropriate source based on the nature of the query.
"""

from pydantic import BaseModel, Field
from typing import Literal, List, Optional, Dict, Any
from langchain_core.prompts import ChatPromptTemplate


class RouteQuery(BaseModel):
    source: Literal["vector_retriever", "web_search"] = Field(
        ...,
        description="The source using which the query needs to be answered, either vector_retriever or web_search.",
    )


system_prompt_router = """ You are an expert in routing queries to the appropriate source. Your task is to determine the best source for answering the given query.
Instructions for choosing the source:
1. If the query is related to financial markets, investments, or economic trends, route it to the vector_retriever.
2. If the query is related to general knowledge or current events, route it to the web_search.
"""

router_prompt_template = ChatPromptTemplate(
    messages=[
        {"role": "user", "content": system_prompt_router},
        {
            "role": "human",
            "content": """Given the following query, determine the best source for answering it:{query}""",
        },
    ],
    input_variables=["query"],
)


def get_question_router_chain(llm):
    """
    Create a query routing chain using the provided language model (llm).
    Args:
        llm: The language model to be used for routing queries.
    Returns:
        A structured query routing chain that takes the query as input and outputs the source to be used.

    """
    structured_llm_router = llm.with_structured_output(schema=RouteQuery)
    return router_prompt_template | structured_llm_router
