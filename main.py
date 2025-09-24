import os
from dotenv import load_dotenv
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import json

# Load .env
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if api_key is None:
    raise ValueError("GOOGLE_API_KEY not found in .env")
os.environ["GOOGLE_API_KEY"] = api_key

# Load CSV subset
loader = CSVLoader(file_path="wiki_movie_plots_deduped.csv", encoding="utf-8")
docs = loader.load()[:500]  

# Simplify metadata
for doc in docs:
    doc.metadata = {"title": doc.metadata.get("Title", "")}

# Split long plots into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,    
    chunk_overlap=0,
    length_function=lambda x: len(x.split()) 
)
chunks = text_splitter.split_documents(docs)

# Embeddings and FAISS vector store
embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(chunks, embed_model)

# Gemini Chat LLM
llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash-lite")

# Prompt template
prompt_template = """
You are a helpful assistant. Use the following context to answer the question.

Context:
{context}

Question:
{question}

Provide the output as valid JSON with these exact keys:
- "answer": concise answer to the question
- "reasoning": brief explanation of how you derived the answer from the context

Return ONLY the JSON object, nothing else.
"""

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# Memory for multi turn conversation
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer"  
)

# Conversational Retrieval Chain
conv_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
    memory=memory,
    combine_docs_chain_kwargs={"prompt": PROMPT},
    return_source_documents=True
)

def parse_llm_response(response_text):
    """Parse the LLM response to extract JSON"""
    try:
        # Try to find JSON in the response
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}') + 1
        if start_idx != -1 and end_idx != 0:
            json_str = response_text[start_idx:end_idx]
            return json.loads(json_str)
        else:
            # Fallback: if no JSON found, return as answer
            return {"answer": response_text.strip(), "reasoning": "Response parsing failed"}
    except json.JSONDecodeError:
        return {"answer": response_text.strip(), "reasoning": "Invalid JSON format in response"}
    
# Interactive terminal loop
print("Mini-RAG Movie Plot QA (type 'exit' to quit)")

while True:

    query = input("\nEnter your question: ")
    if query.lower() in ("exit", "quit"):
        print("Exiting...")
        break

    result = conv_chain.invoke({"question": query})
    llm_output = parse_llm_response(result["answer"])

    # truncating context
    contexts = []
    for doc in result["source_documents"]:
        text = doc.page_content
        if len(text) > 300:
            text = text[:300] + "..."  
        contexts.append(text)
    
    output = {
        "answer": llm_output.get("answer", "No answer generated"),
        "contexts": contexts,
        "reasoning": llm_output.get("reasoning", "No reasoning provided")
    }

    print("Answer:\n", output["answer"], "\n")
    print("Contexts:")
    for ctx in output["contexts"]:
        print("-", ctx, "\n")
    print("Reasoning:\n", output["reasoning"], "\n")

