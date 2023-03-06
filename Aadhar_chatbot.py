import logging
import json
import pandas as pd
from haystack.pipelines import FAQPipeline
from haystack.nodes import EmbeddingRetriever
from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import EmbeddingRetriever
from haystack.utils import print_answers


logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
logging.getLogger("haystack").setLevel(logging.INFO)
document_store = InMemoryDocumentStore()
retriever = EmbeddingRetriever(
    document_store=document_store,
    embedding_model='sentence-transformers/all-MiniLM-L6-v2'
)


def read_question_answers_list(file_name):
    with open(file_name) as f:
        data = json.load(f)
    return data['faq']

faq=read_question_answers_list('/Users/shirinwadood/Desktop/projects/Haystack/Aadhar_Faq.json')
df = pd.DataFrame(faq)
questions = list(df.question.values)
df["embedding"] = retriever.embed_queries(queries=questions).tolist()
df = df.rename(columns={"question": "content"})
docs_to_index = df.to_dict(orient="records")
document_store.write_documents(docs_to_index)
pipe = FAQPipeline(retriever=retriever)

# Run any question and change top_k to see more or less answers
aadhar_chatbot = pipe.run(query = 'How can i download aadhar', params={"Retriever": {"top_k": 1}})
print_answers(aadhar_chatbot, details="medium")
