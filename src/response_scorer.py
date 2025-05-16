"""
Response Scoring Module
This module provides functionality to evaluate the quality and relevance of a summarized response based on the original question and the retrieved documents.
"""

from pydantic import BaseModel, Field
from typing import Literal
from langchain_core.prompts import ChatPromptTemplate


class CheckandGradeResponse(BaseModel):
    """
    A class to represent the grading of a response based on the original question and the summarized response.
    The grading is done to ensure the quality and relevance of the response.
    """

    hallucination: Literal["yes", "no"] = Field(
        ...,
        description="A binary score indicating whether the response contains hallucinations or not.",
    )

    response_score: float = Field(
        ...,
        description="A score indicating the quality of the response, ranging from 0 to 1 based on the relevance and precision of the information provided.",
    )


system_prompt_checker_grader = """
Task:
Evaluate whether the summarized response < Summarized Response> accurately and relevantly answers the original user question <Original Question>, based on the retrieved documents<Retrieved Documents>.

Instructions:
1. Carefully read the original user question, the summarized response, and the raw content of the retrieved documents.
2. Identify hallucinations in the summarized response:
   - A hallucination is any fabricated, inaccurate, or unsupported information not grounded in the retrieved documents.
   - When the facts and information in the summarized response do not match the content of the retrieved documents.
3. Set the Hallucination flag to:
   - 'yes' if any hallucinated content is found.
   - 'no' if the response is fully supported by the retrieved documents.
4. Assign a numerical score between 0.0 and 1.0 based on the following:
   - Relevance: How well the summarized response answers the original question.
   - Precision: How factually accurate the summary is according to the retrieved documents.
   - Score 1.0 for responses that are fully relevant and precise.
   - Score 0.0 for responses that are entirely irrelevant, incorrect, or hallucinated.
   - Use intermediate values (e.g., 0.3, 0.6, 0.8) to reflect partially relevant or accurate responses.
5. If hallucination is 'yes', the score must be 0.0.
6. If hallucination is 'no', assign a score between 0.0 and 1.0 based on relevance and accuracy.
7. Do not include any explanation or justification.
8. Return only the following output format:
   Hallucination: <yes/no>
   Score: <float value between 0.0 and 1.0>

Input:
Original Question: <The original user question.>
Summarized Response: <The summarized response generated from the question and retrieved documents.>
Retrieved Documents: <The raw content of the retrieved documents used to generate the summary.>
"""


def get_response_scorer_chain(llm):
    """
    Create a response grading chain using the provided language model (llm).
    Args:
        llm: The language model to be used for grading responses.
    Returns:
        A structured response grading chain that takes the original question, summarized response, and retrieved documents as input
        and outputs a grading score and hallucination flag.
    1. The grading score is a float value between 0.0 and 1.0.
    2. The hallucination flag is a binary value indicating whether the response contains hallucinations or not.
    """
    structured_llm_checker = llm.with_structured_output(schema=CheckandGradeResponse)
    checker_grader_prompt_template = ChatPromptTemplate(
        messages=[
            {
                "role": "user",
                "content": system_prompt_checker_grader,
            },
            {
                "role": "human",
                "content": """Given the following summarized response {response} , raw content of the retrieved documents {context} and user question {question}, evaluate whether the summarized response accurately and relevantly answers the original user question.""",
            },
        ],
        input_variables=["response", "question", "context"],
        partial_variables={},
    )
    return checker_grader_prompt_template | structured_llm_checker
