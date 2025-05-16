from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field


# Define the output schema
class ThemeKeywords(BaseModel):
    summary: str = Field(
        ...,
        description="A summary of the document content in 1-2 sentences which covers the most important aspects described in the document and is easy to understand.",
    )
    keywords: list[str] = Field(
        ..., description="A list of the most important keywords from the document."
    )


# Patch for Pydantic v1 compatibility
if not hasattr(ThemeKeywords, "model_json_schema"):
    ThemeKeywords.model_json_schema = ThemeKeywords.schema


def get_theme_keywords_chain(llm):
    """
    Returns a chain that extracts a summary and keywords from a document using the provided LLM.
    """
    parser = PydanticOutputParser(pydantic_object=ThemeKeywords)
    format_instructions = parser.get_format_instructions()

    theme_keywords_prompt = ChatPromptTemplate(
        messages=[
            {
                "role": "user",
                "content": """You are an expert in content analysis and keyword extraction. Your task is to analyze the provided page content and extract the main theme and important keywords.
Given the following page content, extract:
1. A summary of the document content in 1-2 sentences which covers the most important aspects described in the document and is easy to understand.
2. A list of the most important keywords from the document.

Page Content:
{page_content}

Return your answer  in the given format:
{format_instructions}
""",
            }
        ],
        input_variables=["page_content"],
        partial_variables={"format_instructions": format_instructions},
    )

    return theme_keywords_prompt | llm | parser
