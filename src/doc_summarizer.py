"""
Document Summarizer Module
This module provides functionality to summarize the content of retrieved documents
"""

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate


class SummarizedResponse(BaseModel):
    """
    A class to represent the summarization of a response based on the original question and the retrieved documents.
    The summarized response is generated to provide a concise answer to the user's question.
    """

    summary: str = Field(
        ...,
        description="The summarized response based on the original question and the retrieved documents with proper headings and subheadings and bullet points. The summary should have introduction, body, and conclusion witthout citations.",
    )
    citation: str = Field(
        ...,
        description="The citation for the summarized response, indicating the source of the information used in the summary.",
    )


system_prompt_summarizer = """
Task:
Generate a focused and accurate summary that answers the user’s original question using the retrieved documents and their citations.

Instructions:
1. Carefully read the original question, the full content of the retrieved documents, and the provided citation references.
2. Identify the most important and relevant information that directly responds to the user’s question.
   - Prioritize facts, insights, or arguments that align semantically and contextually with the question.
3. Write a concise and coherent summary that:
   - Clearly addresses the user's question.
   - Captures the essence of the supporting documents.
   - Is easy to understand and free from ambiguity.
4. Do not include information that is irrelevant, vague, or unrelated to the question.
5. Integrate citations into the summary to support factual statements, referencing the sources in the same order as they appear in the retrieved documents.
6. Maintain a neutral and informative tone—avoid speculation or unsupported claims.
7. Only return the final summary. Do not include commentary or formatting outside the summary text.
8. The summary should be structured with proper headings and subheadings and bullet points and in string format.
9. The summary should always have and introduction, body and conclusion in the response.

Input:
Original Question: <The user’s original question.>
Retrieved Documents: <Raw text content from the retrieved documents.>
Citations: <Citation identifiers matching the order and origin of the retrieved documents.>
"""


def get_doc_summarizer_chain(llm):
    """
    Create a document summarization chain using the provided language model (llm).
    Args:
        llm: The language model to be used for summarizing documents.
    Returns:
        A structured document summarization chain that takes the user question and document content as input
        and outputs a summarized response.

    """
    structured_llm_summarizer = llm.with_structured_output(schema=SummarizedResponse)
    summarizer_prompt_template = ChatPromptTemplate(
        messages=[
            {
                "role": "user",
                "content": system_prompt_summarizer,
            },
            {
                "role": "human",
                "content": """Given the following retrieved set of document{context}, citations {citation} and user question {question}, summarize the response to provide a concise answer to the user's question.""",
            },
        ],
        input_variables=["context", "question"],
        partial_variables={},
    )
    return summarizer_prompt_template | structured_llm_summarizer
