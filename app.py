from typing import List, Optional, Tuple, Union
from uuid import uuid4
from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from tqdm import tqdm
from upstash_vector import Index
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import streamlit as st


import os




os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
from exa_py import Exa
exa = Exa(os.getenv("Exa"))


def retrieve_context(query):
    results = exa.search(query, use_autoprompt=True, num_results=3)
    ids = [result.id for result in results.results]
    text = {"include_html_tags": False}
    contents = exa.get_contents(ids, text=text)
    context = [content.text for content in contents.results]
    # context = "\n\n\n".join(context_list)
    return context



class UpstashVectorStore:
    def __init__(self, index: Index, embeddings: Embeddings):
        self.index = index
        self.embeddings = embeddings

    def delete_vectors(self, ids: Union[str, List[str]], delete_all: bool = None,):
        if delete_all:
            self.index.rest()
        else:
            self.index.delete(ids)

    def add_documents(self, documents: List[str], ids: Optional[List[str]] = None, batch_size: int = 32):
        texts = []
        metadatas = []
        all_ids = []

        for document in tqdm(documents):
            text = document
            metadata = document
            metadata = {"context": text}
            texts.append(text)
            metadatas.append(metadata)



            if len(texts) >= batch_size:
                ids = [str(uuid4()) for _ in range(len(texts))]
                all_ids += ids
                embeddings = self.embeddings.embed_documents(texts, batch_size=250)
                self.index.upsert(
                    vectors = zip(ids, embeddings, metadatas)
                )
                texts = []
                metadatas = []

        if len(texts) > 0:
            ids = [str(uuid4()) for _ in range(len(texts))]
            all_ids += ids
            embeddings = self.embeddings.embed_documents(texts, batch_size=250)
            self.index.upsert(
                vectors = zip(ids, embeddings, metadatas)
            )

        n = len(all_ids)
        print(f"Sucessfully indexed {n} dense vectors to Upstash")
        return all_ids
    

    def similarity_search_with_score(self, query: str, k: int = 4
                                     ) -> List[Tuple[str, float]]:
        query_embedding = self.embeddings.embed_query(query)
        query_results = self.index.query(
            query_embedding, 
            top_k = k,
            include_metadata=True
        )

        output = []
        for query_result in query_results:
            score = query_result.score
            metadata = query_result.metadata
            #print(metadata)
            #print(query_result)
            context = metadata['context']
            doc = Document(
                page_content=context,
                metadata=metadata,
            )
            output.append((doc, score))
        #print(output)
        return output
        
    
    
def get_context(query, vector_store):
    results = vector_store.similarity_search_with_score(query)
    context = ""

    for doc, score in results:
        context += doc.page_content + "\n===\n"
    #print(context)    
    return context

def get_prompt(question, context):
    template = """
    Your task is to answer questions by using a given context.

    Don't invent anything that is outside of the context.
    Answer in at least 350 characters.

    %CONTEXT%
    {context}

    %Question%
    {question}

    Hint: Do not copy the context. Use your own words
    
    Answer:
    Just return the answer and no additions like content=
    """
    prompt = template.format(question=question, context=context)
    return prompt




from langchain_google_genai import ChatGoogleGenerativeAI
from google.generativeai.types.safety_types import HarmBlockThreshold, HarmCategory

llm = ChatGoogleGenerativeAI(model="gemini-pro")
#,safety_settings = {
#HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_NONE, HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE, HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH, HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE, HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
#})

from upstash_vector import Index

index = Index(url="https://normal-mammal-32964-eu1-vector.upstash.io", token=os.getenv("UPSTASH_TOKEN"))

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

upstash_vector_store = UpstashVectorStore(index=index, embeddings=embeddings)



def main():
    input_query = st.chat_input("Enter your query")
    response = ''
    if input_query:
        context = retrieve_context(input_query)

        ids = upstash_vector_store.add_documents(context, batch_size=25)
        answer_context = get_context(input_query, upstash_vector_store)
        prompt = get_prompt(input_query, answer_context)
        result = llm.invoke(prompt)
        st.write(f"{result.content}")


    



if __name__ == "__main__":
    main()
