from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
# from apillama3.llama3api.model_loader import llm_model
from llama3api.model_loader import llm_model 



from langchain import HuggingFacePipeline, PromptTemplate
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from transformers import AutoTokenizer, TextStreamer, pipeline
from transformers import AutoModelForCausalLM

class ChatbotModel:
    def __init__(self):
        self.model = llm_model
        # self.model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
        # self.model = AutoModelForCausalLM.from_pretrained(
        #     self.model_id,
        #     torch_dtype=torch.bfloat16,
        #     device_map="auto",
        # )
    
    def generate_response(self, user_input):
        messages = [
            {"role": "system", "content": "You are the knowledgeable chatbot to answer the user query perfectly"},
            {"role": "user", "content": user_input},
        ]
        
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device)

        eos_token_id = self.tokenizer.eos_token_id

        outputs = self.model.generate(
            input_ids,
            max_new_tokens=500,
            eos_token_id=eos_token_id,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        response = outputs[0][input_ids.shape[-1]:]
        result = self.tokenizer.decode(response, skip_special_tokens=True)
        
        # Clean up to free GPU memory
        del input_ids
        del outputs
        torch.cuda.empty_cache()
        
        return result


    def initialize_components(self,file_path,question,user_prompt=None,botname="Chatbot:"):
            loader = PyPDFLoader(file_path)
            data = loader.load()
        
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"device": "cuda"}
            )
        
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=64)
            texts = text_splitter.split_documents(data)
        
            db = Chroma.from_documents(texts, embeddings)
        
        
            streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
            text_pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=1024,
                temperature=0.5,
                top_p=0.95,
                repetition_penalty=1.15,
                streamer=streamer,
            )
            llm = HuggingFacePipeline(pipeline=text_pipeline, model_kwargs={"temperature": 0.7})
            
            # Default user prompt if none is provided
            if user_prompt is None:
                user_prompt = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
                If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""
    
            context_template = f"\nContext: {{context}}\nUser: {{question}}\n{botname}"

        # Combine user-defined prompt with the context template
            combined_prompt = f"{user_prompt}{context_template}"
        
            prompt = PromptTemplate(template=combined_prompt, input_variables=["context", "question"])
        
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=db.as_retriever(search_kwargs={"k": 2}),
                return_source_documents=True,
                chain_type_kwargs={"prompt": prompt},
            )
        
                    # Execute a query (you can customize this part as needed)
            query_result = qa_chain(question)

            response_content = query_result['result']
            answer_prefix = botname
            answer_start_index = response_content.find(answer_prefix)
            if answer_start_index != -1:
                answer = response_content[answer_start_index + len(answer_prefix):].strip()
                print(answer)
                return answer
            else:
                print("No answer found in the response.")
                return response_content


    







    