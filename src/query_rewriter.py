"""
Query Rewriter module
This module provides functionality to rewrite queries based on the original question and the retrieved documents.
"""

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate


class RewriteQuery(BaseModel):
    """
    A class to represent the rewriting of a query based on the original question and the retrieved documents.
    The rewritten query is generated to improve the relevance of the retrieved documents.
    """

    rewritten_query: str = Field(
        ...,
        description="The rewritten query based on the original question and the retrieved documents.",
    )


system_prompt_rewriter = """
Task:
Rewrite the user's original question to improve the relevance and specificity of the retrieved documents.

Instructions:
1. Carefully read the original question and the provided document keywords or context.
2. Rewrite the question to be more specific or relevant, using the context as guidance.
3. Do not add unrelated information or change the intent of the original question.
4. Return only the rewritten question.
"""

rewrite_prompt_template = ChatPromptTemplate(
    messages=[
        {
            "role": "user",
            "content": system_prompt_rewriter,
        },
        {
            "role": "human",
            "content": """Given the following original question {question} and document keyword {context}, rewrite the question to improve its relevance and specificity.""",
        },
    ],
    input_variables=["context", "question"],
    partial_variables={},
)


def get_query_rewrite_chain(llm):
    """
    Create a query rewriting chain using the provided language model (llm).
    Args:
        llm: The language model to be used for rewriting queries.
    Returns:
        A structured query rewriting chain that takes the original question and context as input
        and outputs a rewritten query.
    """
    structured_query_rewriter = llm.with_structured_output(schema=RewriteQuery)
    return rewrite_prompt_template | structured_query_rewriter
