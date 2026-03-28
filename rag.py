from KnowledgeBase.rag.vector_stores import VectorStoreService
import config_data as config
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableWithMessageHistory, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from file_history_store import get_history

def print_prompt(prompt):
    print("="*20)
    print(prompt.to_string())
    print("="*20)
    return prompt

class RagService():
    def __init__(self):
        self.vector_service = VectorStoreService(embedding=OllamaEmbeddings(model=config.embed_model_name))
        self.prompt_template = ChatPromptTemplate.from_messages([
           ("system", "以我提供的已知参考资料为主, 简洁和专业地回答用户的问题. 参考资料:{context}"),
           ("system", "并且我提供用户的历史会话记录如下:"),
           MessagesPlaceholder("history"),
           ("user", "请回答用户提问:{input}")
        ])
        self.chat_model = ChatOllama(model=config.chat_model_name)
        self.chain = self.__get_chain()

    def __get_chain(self):
        retriever = self.vector_service.get_retriever()

        def format_document(docs):
            if not docs:
                return ""
            formated_str = ""
            for doc in docs:
                formated_str += f"文档内容:{doc.page_content}\n文档元数据:{doc.metadata}\n"
            return formated_str

        def format_for_retriever(value):
            return value["input"]
        
        def format_for_template(value):
            new_value = {}
            new_value["input"] = value["input"]["input"]
            new_value["context"] = value["context"]
            new_value["history"] = value["input"]["history"]
            return new_value
        
        chain = (
            {
                "input": RunnablePassthrough(),
                "context": RunnableLambda(format_for_retriever) | retriever | format_document,
            } | RunnableLambda(format_for_template) | self.prompt_template | print_prompt | self.chat_model | StrOutputParser()
        )

        conversation_chain = RunnableWithMessageHistory(
            chain,
            get_history,
            input_messages_key="input",
            history_messages_key="history",
        )
        return conversation_chain
    
if __name__ == "__main__":
    # session_id配置
    session_config = {
        "configurable":{
            "session_id": "user_001",
        }
    }
    res = RagService().chain.invoke({"input": "陕西旅游的核心运营项目是什么?"}, session_config)
    print(res)