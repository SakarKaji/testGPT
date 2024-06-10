import os
import tempfile
import qdrant_client
import pytesseract
from PIL import Image
import streamlit as st
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex, Settings
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor, TimeWeightedPostprocessor
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore

class App:
    def reset_pipeline_generated(self):
        if 'pipeline_generated' in st.session_state:
            st.session_state['pipeline_generated'] = False

        if 'messages' in st.session_state:
            del st.session_state['messages']

    def upload_file(self):
        file = st.sidebar.file_uploader('Upload your document', on_change=self.reset_pipeline_generated)
        if file is not None:
            file_path = self.save_uploaded_file(file)
            if file_path:
                if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                    extracted_text = self.extract_text_from_image(file_path)
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp_file:
                        tmp_file.write(extracted_text.encode())
                        file_path = tmp_file.name
                loaded_file = SimpleDirectoryReader(input_files=[file_path]).load_data()
                return loaded_file
        return None

    def save_uploaded_file(self, uploaded_file):
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                return tmp_file.name
        except Exception as e:
            st.error(f"Error saving file: {e}")
            return None

    def extract_text_from_image(self, image_path):
        try:
            image = Image.open(image_path)
            text = pytesseract.image_to_string(image)
            return text
        except Exception as e:
            st.error(f"Error extracting text from image: {e}")
            return ""

    def node_parser(self):
        Settings.text_splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=20)
        return Settings.text_splitter

    def llm_model(self):
        Settings.llm = Ollama(model="phi3:mini", request_timeout=1000.0)
        return Settings.llm

    def embedding_model(self):
        with st.spinner("please wait") as status:
            Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")
            st.session_state['embed_model'] = Settings.embed_model
        return Settings.embed_model

    def response_synthesis_method(self):
        response_mode = "refine"
        return response_mode

    def vector_store(self):
        client = qdrant_client.QdrantClient(location=":memory:")
        vector_store = QdrantVectorStore(client=client, collection_name="sampledata")
        return vector_store

    def generate_rag_pipeline(self, file, llm, embed_model, node_parser, response_mode, vector_store):
        if vector_store is not None:
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
        else:
            storage_context = None

        vector_index = VectorStoreIndex.from_documents(
            documents=file,
            llm=llm,
            embed_model=embed_model,
            node_parser=node_parser,
            storage_context=storage_context,
            show_progress=True
        )
        if storage_context:
            vector_index.storage_context.persist(persist_dir="dir")

        query_engine = vector_index.as_query_engine(response_mode=response_mode, verbose=True, similarity_top_k=10, node_postprocessors=[
            TimeWeightedPostprocessor(
                time_decay=0.5, time_access_refresh=False, top_k=1
            )
        ])
        return query_engine

    def main(self):
        st.set_page_config(page_title="QuickChat")
        st.title("💬 QuickChat")
        st.caption('📝 Chat with your document')

        file = self.upload_file()
        node_parser = self.node_parser()
        llm = self.llm_model()
        embed_model = self.embedding_model()
        vector_store = self.vector_store()
        response_mode = self.response_synthesis_method()

        if file is not None:
            if st.sidebar.button("Generate RAG Pipeline"):
                with st.spinner():
                    query_engine = self.generate_rag_pipeline(file, llm, embed_model, node_parser, response_mode, vector_store)
                    st.session_state['query_engine'] = query_engine
                    st.session_state['pipeline_generated'] = True
                    st.success("RAG Pipeline Generated Successfully!")

            elif file is None:
                st.error("Please upload a file")

        if st.session_state.get('pipeline_generated', False):
            if 'messages' not in st.session_state:
                st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

            for msg in st.session_state.messages:
                st.chat_message(msg["role"]).write(msg['content'])

            if prompt := st.chat_input("Enter your query", key='query'):
                st.session_state.messages.append({'role': 'user', 'content': prompt})
                st.chat_message('user').write(prompt)
                st.markdown("")
                if 'query_engine' in st.session_state:
                    system_prompt = (
                        "You are an AI assistant specialized in providing information from the uploaded document. "
                        "Please ensure that your responses are strictly derived from the content of the document. "
                        "If the information is not found in the document, please indicate that explicitly."
                    )
                    query_with_prompt = f"{system_prompt}\nUser query: {prompt}"
                    Message = st.session_state['query_engine'].query(query_with_prompt)
                    msg = Message.response
                    st.session_state.messages.append({'role': 'assistant', 'content': msg})
                    st.chat_message("assistant").write(msg)
                else:
                    st.error("Query engine is not initialized")

if __name__ == '__main__':
    a = App()
    a.main()
