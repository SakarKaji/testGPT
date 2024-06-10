import os,tempfile,qdrant_client
import streamlit as st
from llama_index.core import SimpleDirectoryReader, StorageContext,VectorStoreIndex,Settings
from llama_index.core.node_parser import SentenceSplitter
# from llama_index.llms.openai import OpenAI
# from dotenv import load_dotenv
# from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
# from neo4j import GraphDatabase

class App:
   def reset_pipeline_generated(self):
      if 'pipeline_generated' in st.session_state:
        st.session_state['pipeline_generated'] = False

      if 'messages' in st.session_state:
         del st.session_state['messages']

   #Here, we upload the file
   def upload_file(self):
        file = st.sidebar.file_uploader('Upload your document',on_change=self.reset_pipeline_generated)
        if file is not None:
           file_path=self.save_uploaded_file(file)

           if file_path:
              loaded_file=SimpleDirectoryReader(input_files=[file_path]).load_data()
              return loaded_file
        return None
    
   #Here, we save the uploaded file
   def save_uploaded_file(self,uploaded_file):
        try:
          with tempfile.NamedTemporaryFile(delete=False,suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
              tmp_file.write(uploaded_file.getvalue())
              return tmp_file.name
        except Exception as e:
           st.error(f"Error saving file: {e}")
           return None
        
   #Next, we chunk the file
   def node_parser(self):
      Settings.text_splitter=SentenceSplitter(chunk_size=1024,chunk_overlap=20)
      return Settings.text_splitter
      
   #Here, we define our LLM model to use
   def llm_model(self):
      #For OpenAi 
      # load_dotenv()
      # openai_api_key=os.getenv('OPENAI_API_KEY')
      #  st.write(f"Aip key is {openai_api_key}")
      # llm=OpenAI(temperature=0.1,model="gpt-3.5-turbo-1106",api_key=openai_api_key)

      #For Ollama
      Settings.llm=Ollama(model="llama3",request_timeout=300.0,temperature=0.2,top_p=0.5,top_k=40)
      return Settings.llm
       
   #Here, we select the embedding model
   def embedding_model(self):
      with st.spinner("please wait") as status:
         # #For OpenAI
         # embed_model=OpenAIEmbedding()

         #For Ollama
         Settings.embed_model=OllamaEmbedding(model_name="snowflake-arctic-embed")

         st.session_state['embed_model'] = Settings.embed_model
         # st.markdown(F"Embedding Model: {embed_model.model_name}")
      return Settings.embed_model
      
   #Here, we define response synthesis method
   def response_synthesis_method(self):
      response_mode="refine"
      return response_mode
   
   #Here, we make a vector database
   def vector_store(self):
      client=qdrant_client.QdrantClient(location=":memory:")
      vector_store=QdrantVectorStore(client=client,collection_name="sampledata")
      return vector_store
   
   #Now, creating the rag pipeline
   def generate_rag_pipeline(self,file,llm,embed_model,node_parser,response_mode,vector_store):

      if vector_store is not None:
        #Set storage context if vector store is not None
        storage_context=StorageContext.from_defaults(vector_store=vector_store)
      else:
        storage_context=None
      #Create the service context
    #   service_context=ServiceContext.from_defaults(llm=llm, embed_model=embed_model, node_parser=node_parser)
    
      #Create the vector index
      vector_index=VectorStoreIndex.from_documents(documents=file,llm=llm,embed_model=embed_model,node_parser=node_parser,storage_context=storage_context,show_progress=True)
      if storage_context:
        vector_index.storage_context.persist(persist_dir="dir")

#       system_prompt = (
#    "You are an AI assistant specialized in providing information from the uploaded document. "
#    "Please ensure that your responses are strictly derived from the content of the document. "
#    "If the information is not found in the document, please indicate that explicitly."
# )
      #Create the query engine
      query_engine=vector_index.as_query_engine(response_mode=response_mode,verbose=True)
      return query_engine

   def main(self):
      st.set_page_config(page_title="QuickChat")
      st.title("💬 QuickChat")
      st.caption('📝 Chat with your document')

      file=self.upload_file()
      node_parser=self.node_parser()
      # if file is not None:
      #      if node_parser:
      #         nodes=node_parser.get_nodes_from_documents(file)
      #       #   st.write(nodes[0])
      #       #   st.write(len(nodes))
        
      llm=self.llm_model()
      embed_model=self.embedding_model()
      vectore_store=self.vector_store()
      response_mode=self.response_synthesis_method()
        
      #Generate RAG Pipeline Button  
      if file is not None:
         if st.sidebar.button("Generate RAG Pipeline"):
            with st.spinner():
               query_engine=self.generate_rag_pipeline(file,llm,embed_model,node_parser,response_mode,vectore_store)
               st.session_state['query_engine'] = query_engine
               st.session_state['pipeline_generated'] = True
               st.success("RAG Pipeline Generated Successfully!")

         elif file is None:
            st.error("Please upload a file")

      #After Generating the RAG pipeline
      if st.session_state.get('pipeline_generated',False):
        if 'messages' not in st.session_state:
            st.session_state["messages"]=[{"role": "assistant", "content": "How can I help you?"}]

        for msg in st.session_state.messages:
           st.chat_message(msg["role"]).write(msg['content'])

        if prompt := st.chat_input("Enter your query", key='query'):
            st.session_state.messages.append({'role':'user','content':prompt})
            st.chat_message('user').write(prompt)
            st.markdown("")
            if 'query_engine' in st.session_state:
               system_prompt = (
   "You are an AI assistant specialized in providing information from the uploaded document. "
   "Please ensure that your responses are strictly derived from the content of the document. "
   "If the information is not found in the document, please indicate that explicitly."
)              
               query_with_prompt=f"{system_prompt}\nUser query:{prompt}"
               Message=st.session_state['query_engine'].query(query_with_prompt)
               msg=Message.response
               st.session_state.messages.append({'role':'assistant','content':msg})
               st.chat_message("assistant").write(msg)   
            else:
               st.error("Query engine is not initialised")

if __name__=='__main__':
    a=App()
    a.main()