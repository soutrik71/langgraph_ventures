from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from src.theme_generation import get_theme_keywords_chain
import asyncio

class DocumentProcessor:
    def __init__(self, llm):
        self.llm = llm
        self.theme_keywords_chain = get_theme_keywords_chain(llm)
        self.docs_list = []
        self.theme_keywords_results = []

    def load_documents(self, urls):
        """
        Input: List of URLs
        Output: List of loaded documents (self.docs_list)
        """
        docs = [WebBaseLoader(url).load() for url in urls]
        self.docs_list = [item for sublist in docs for item in sublist]
        print(f"Loaded {len(self.docs_list)} documents from {len(urls)} URLs.")
        return self.docs_list

    async def extract_theme_keywords(self):
        """
        Input: Uses self.docs_list
        Output: List of theme/keyword results (self.theme_keywords_results)
        """
        async def process(doc):
            return await self.theme_keywords_chain.ainvoke({"page_content": doc.page_content})

        self.theme_keywords_results = await asyncio.gather(*(process(doc) for doc in self.docs_list))
        return self.theme_keywords_results

    def enrich_metadata(self):
        """
        Input: Uses self.docs_list and self.theme_keywords_results
        Output: Updates self.docs_list in-place with 'summary' and 'keywords' in metadata
        """
        for i, doc in enumerate(self.docs_list):
            doc.metadata["summary"] = self.theme_keywords_results[i].summary
            doc.metadata["keywords"] = " ".join(self.theme_keywords_results[i].keywords)
        return self.docs_list

    def split_documents(self, chunk_size=1000, chunk_overlap=0):
        """
        Input: Uses self.docs_list
        Output: List of split document chunks
        """
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        doc_splits = text_splitter.split_documents(self.docs_list)
        print(f"Split into {len(doc_splits)} chunks.")
        return doc_splits

    async def process_all(self, urls, chunk_size=1000, chunk_overlap=0):
        """
        Public method to run the full pipeline:
        1. Load documents from URLs
        2. Extract theme keywords
        3. Enrich metadata
        4. Split documents

        Input: 
            urls: list of URLs
            chunk_size: int (default 1000)
            chunk_overlap: int (default 0)
        Output: 
            doc_splits: list of split document chunks
        """
        self.load_documents(urls)
        await self.extract_theme_keywords()
        self.enrich_metadata()
        return self.split_documents(chunk_size=chunk_size, chunk_overlap=chunk_overlap)