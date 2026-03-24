import os
import config_data as config
import hashlib
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from datetime import datetime

def check_md5(md5_str: str):
    # 检查传入的md5字符串是否已经被处理过了
    if not os.path.exists(config.md5_path):
        open(config.md5_path, "w", encoding="utf-8").close()
        return False
    else:
        with open(config.md5_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if line == md5_str:
                    return True
        return False

def save_md5(md5_str: str):
    # 将传入的md5字符串记录到文件内保存
    with open(config.md5_path, "a", encoding="utf-8") as f:
        f.write(md5_str + "\n")

def get_string_md5(input_str: str, encodings="utf-8"):
    # 计算传入字符串的md5值
    md5_obj = hashlib.md5()
    md5_obj.update(input_str.encode(encodings))
    return md5_obj.hexdigest() # 返回md5值的16进制字符串表示

class KnowledgeBaseService(object):
    def __init__(self):
        os.makedirs(config.persist_directory, exist_ok=True)
        self.chrome = Chroma(
            collection_name=config.collection_name,
            embedding_function=OllamaEmbeddings(model="qwen3-embedding:0.6b"),
            persist_directory=config.persist_directory,
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap, 
            separators=config.separators,
            length_function=len,
        )

    def upload_by_string(self, data:str, filename):
        md5_str = get_string_md5(data)
        if check_md5(md5_str):
            return "[跳过]该数据已经被处理过"
        
        if len(data) < config.max_split_char_number:
            knowledge_chunks: list[str] = self.splitter.split_text(data)
        else:
            knowledge_chunks = [data]
        metadata = {
            "filename": filename,
            "create_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "operator": config.operator,
        }
        self.chrome.add_texts(knowledge_chunks, metadatas=[metadata for _ in knowledge_chunks])

        save_md5(md5_str)
        return "[成功]内容已经成功上传至知识库"
            

if __name__ == "__main__":  
    r1 = "周杰伦的《我很忙》是一部关于繁忙生活的电影，它展示了人在繁忙的生活中如何保持平衡和专注。"
    service= KnowledgeBaseService()
    result = service.upload_by_string(r1, "r1.txt")
    print(result)
