operator = "ayanami"

md5_path = "./configs/md5.txt"

# chroma数据库配置
persist_directory = "./chroma_db"
collection_name = "rag"

# 文本分割配置策略
chunk_size = 1000
chunk_overlap = 100
separators = ["\n\n", "\n", "。", ".", "!", "！", "?", "？", " ", ""]    
max_split_char_number = 1000

# 检索器配置
similarity_threshold = 1

# embedding模型配置
embed_model_name = "qwen3-embedding:0.6b"

# chat模型配置
chat_model_name = "qwen3:4b"