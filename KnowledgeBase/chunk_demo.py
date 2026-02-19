from tqdm import tqdm
from dotenv import load_dotenv
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
import pdfplumber
import fitz  # PyMuPDF
import io
from PIL import Image
import base64
import numpy as np
import onnxruntime as ort
from typing import List
from transformers import AutoTokenizer
from langchain_core.embeddings import Embeddings


# 加载 .env 文件中的环境变量
load_dotenv("../env")
# 获取名为'VAR'的环境变量
API_KEY = os.environ.get("API_KEY")
BASE_URL = os.environ.get("BASE_URL")
LM_MODEL = os.environ.get("LM_MODEL")

VL_API_KEY = os.environ.get("VL_API_KEY")
VL_BASE_URL = os.environ.get("VL_BASE_URL")
VL_MODEL = os.environ.get("VL_MODEL")

# 加载模型
# 阿里百炼部分模型只接受流式输入，为保证后续逻辑通用性，在模型加载时作调整
if LM_MODEL not in ["qwq_plus"]:
    llm = ChatOpenAI(
        api_key=API_KEY, base_url=BASE_URL, model=LM_MODEL, temperature=0.7
    )
else:
    llm = ChatOpenAI(
        api_key=API_KEY,
        base_url=BASE_URL,
        model=LM_MODEL,
        streaming=True,
        temperature=0.7,
    )
vl_llm = ChatOpenAI(
    api_key=VL_API_KEY, base_url=VL_BASE_URL, model=VL_MODEL, temperature=0.01
)
# 文本切分器
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)

def get_file_list(root_dir):
    """返回包含所有子文件的完整路径列表"""
    file_list = []
    for root, dirs, files in os.walk(root_dir):
        for filename in files:
            file_path = os.path.join(root, filename)
            file_list.append(file_path)
    return file_list

class ProcessPDF:
    def __init__(self, pdf_path: str):
        '''解析pdf中的文本、表格、图像信息
        '''
        self.pdf_path = pdf_path
        self.texts = self.extract_text()
        self.tabels = self.extract_tables()
        images_info = self.extract_images()
        self.images = [k["image"] for k in images_info]

    def extract_text(self):
        """提取全部文本内容（含基础排版信息）"""
        full_text = []
        with pdfplumber.open(self.pdf_path) as pdf:
            for page in pdf.pages:
                # 提取当前页文本（保留Y坐标顺序）
                text = page.extract_text(x_tolerance=1, y_tolerance=3)
                full_text.append(f"=== Page {page.page_number} ===")
                full_text.append(text)
        return "\n".join(full_text)

    def extract_tables(self):
        """结构化提取表格数据"""
        tables = []

        with pdfplumber.open(self.pdf_path) as pdf:
            for page in pdf.pages:
                for table in page.extract_tables(
                    {"vertical_strategy": "lines", "horizontal_strategy": "text"}
                ):
                    # 转换为二维列表
                    # print(table)
                    try:
                        cleaned_table = [
                            [cell.replace("\n", " ") for cell in row] for row in table
                        ]
                        tables.append(cleaned_table)
                    except:
                        cleaned_table = []
                    # df = pd.DataFrame(table[1:], columns=table[0])
                    # print(df.head())

        return "\n".join(["|".join(k) for l in tables for k in l])

    def extract_images(self, target_size=(250, 300)):
        """提取并保存所有图片"""
        doc = fitz.open(self.pdf_path)
        # os.makedirs(output_dir, exist_ok=True)

        image_info = []
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            image_list = page.get_images(full=True)

            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]

                # 转换为Pillow对象
                img_pil = Image.open(io.BytesIO(image_bytes))
                img_pil.thumbnail(target_size, Image.Resampling.LANCZOS)
                # width, height = img_pil.size
                # 转换为字节流
                with io.BytesIO() as byte_stream:
                    img_pil.save(byte_stream,
                                 quality=80,
                                 format="WEBP" if img_pil.mode == "RGBA" else "JPEG",
                                 optimize=True)
                    img_pil = byte_stream.getvalue()

                # 保存元数据
                meta = {
                    "page": page_num + 1,
                    "index": img_index,
                    "format": base_image["ext"],
                    "image": "data:image/jpeg;base64,{}".format(
                        base64.urlsafe_b64encode(img_pil).decode("utf-8")
                    ),
                }
                image_info.append(meta)

                # # 保存文件
                # filename = f"page{page_num+1}_img{img_index}.{base_image['ext']}"
                # img_pil.save(os.path.join(output_dir, filename))

        return image_info

def parse_path(pdf_path):
    pdf_info = ProcessPDF(pdf_path)
    images_desc = []

    if len(pdf_info.images) > 0:
        meassage_list = []
        # 将图片送入视觉-语言模型进行将解析输出文本
        for image in pdf_info.images:
            meassage_list.append(
                [
                    {
                        "role": "system",
                        "content": [
                            {"type": "text", "text": "帮我描述一下这张图片的内容"}
                        ],
                    },
                    {
                        "role": "user",
                        "content": [{"type": "image_url", "image_url": image}],
                    },
                ]
            )
        try:
            images_desc = [llm_res.content for llm_res in vl_llm.batch(meassage_list)]
        except:
            images_desc = []
    if len(images_desc) < 1:
        # 拼接文本+表格
        return f"# text:\n{pdf_info.texts}\n\n# table:\n{pdf_info.tabels}\n\n"
    else:
        # 拼接文本+表格+图像描述
        return f"# text:\n{pdf_info.texts}\n\n# table:\n{pdf_info.tabels}\n\n# iamges desc info:\n{'====='.join(images_desc)}\n\n"


def parse_rich_pdf_list(pdf_path):
    """解析图文表格结合的复杂pdf"""
    pdf_path_list = get_file_list(pdf_path)
    texts = []
    for p in tqdm(pdf_path_list):
        texts.append(parse_path(p))
    return texts



# 自定义一个 Embeddings 类来加载和使用 ONNX 模型
class ONNXEmbeddings(Embeddings):
    def __init__(self, model_path):
        # 初始化 ONNX 会话
        self.session = ort.InferenceSession(f"{model_path}/model.onnx")
        # 获取输入和输出名称
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.tokenizer_function = AutoTokenizer.from_pretrained(
            model_path, use_fast=True
        )
        self.dim = 1024

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            inputs = self.tokenizer_function(text, return_tensors="np")

            # 运行 ONNX 模型进行推理
            outputs = self.session.run(
                [self.output_name],
                {
                    self.input_name: inputs["input_ids"].astype(np.int64),
                    "attention_mask": inputs["attention_mask"].astype(np.int64),
                },
            )
            # print(len(outputs))
            # print(outputs[0].shape)
            # 获取嵌入句向量
            embedding = np.average(outputs[0][0], axis=0)
            # print(embedding.shape)
            embedding = embedding / np.linalg.norm(embedding)
            embeddings.append(embedding.tolist())

        return embeddings

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

if __name__=='__main__':
    import pickle
    text_list=parse_rich_pdf_list("/home/yang/文档/project/MyNote/pdf")
    # 转换单个字符串为多个 Document
    pickle.dump(text_list,open('pdf_text_list.pk','wb'))
    documents = text_splitter.create_documents(text_list)

    bge_model_path='/home/yang/文档/project/MyNote/work_paln/bge-m3-onnx'
    embedding_model=ONNXEmbeddings(bge_model_path)
    embedding_dim=len(embedding_model.embed_query('你好'))
    from pymilvus import MilvusClient
    # 实例化milvus客户端
    client = MilvusClient(
        uri="http://localhost:19530",
        token="root:Milvus"
    )
    collection_name = "lm_doc"
    if client.has_collection(collection_name):
        client.drop_collection(collection_name)
    # 设置index
    index_params = MilvusClient.prepare_index_params()
    metric_type='IP'
    index_params.add_index(
                field_name="vector", # Name of the vector field to be indexed
                index_type="IVF_FLAT", # Type of the index to create
                index_name="vector_index", # Name of the index to create
                metric_type=metric_type, # Metric type used to measure similarity
                params={
                    "nlist": 128, # Number of clusters for the index
                } # Index building params
            )
    # 创建collection
    client.create_collection(
        collection_name=collection_name,
        dimension=embedding_dim,
        metric_type=metric_type,  # Inner product distance
        consistency_level="Strong",  # Strong consistency level
        index_params=index_params
    )
    # 文档向量化
    embedding_doc = embedding_model.embed_documents([doc.page_content for doc in documents])
    data=[]
    for i, line in enumerate(tqdm(documents, desc="Creating embeddings")):
        data.append({"id": i, "vector": embedding_doc[i], "text": embedding_doc[i].page_content})
    print("开始插入数据")
    client.insert(collection_name=collection_name, data=data)