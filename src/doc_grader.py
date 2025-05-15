"""
Document Grader Module
This module provides functionality to assess the relevance of retrieved documents
"""

from pydantic import BaseModel, Field
from typing import Literal
from langchain_core.prompts import ChatPromptTemplate


class GradingDocuments(BaseModel):
    """
    A class to represent the grading of documents.
    """

    grade_score: Literal["full_relevance", "partial_relevance", "no_relevance"] = Field(
        ...,
        description="The relevance score of the documents, which can be full_relevance, partial_relevance, or no_relevance based on the content.",
    )


system_prompt_grader = """
Task:
Assess the relevance of a retrieved document in answering the given user question.

Instructions:
1. Carefully read the user question and the retrieved document content.
2. Determine if the document contains information (keywords, facts, context, or semantics) directly related to the question.
3. Classify the document's relevance using one of the following categories:
   - full_relevance:
     The document contains key facts, figures, arguments, or contextual details that directly and comprehensively address the question.
   - partial_relevance:
     The document includes some relevant information but is incomplete, too general, or partially off-topic.
   - no_relevance:
     The document contains little to no meaningful connection to the question, or is entirely off-topic.
4. Base your decision on both semantic similarity and the presence of essential context or key information—not just superficial keyword overlap.
5. Do not include any additional explanation or commentary in the output.
6. Return only the relevance label using one of the following exact values: full_relevance, partial_relevance, no_relevance.

Input:
Question: <The user’s original question.>
Document Content: <The text content of the retrieved document.>
"""

grader_prompt_template = ChatPromptTemplate(
    messages=[
        {
            "role": "user",
            "content": system_prompt_grader,
        },
        {
            "role": "human",
            "content": """Given the following retrieved set of document{context} and user question {question}, grade the document as relevant or not""",
        },
    ],
    input_variables=["context", "question"],
    partial_variables={},
)


def get_doc_grader_chain(llm):
    """
    Create a document grading chain using the provided language model (llm).
    Args:
        llm: The language model to be used for grading documents.
    Returns:
        A structured document grading chain that takes the user question and document content as input
        and outputs the relevance score of the document.

    """
    structured_llm_grader = llm.with_structured_output(schema=GradingDocuments)
    return grader_prompt_template | structured_llm_grader
