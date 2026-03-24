"""
使用streamlit实现在线上传文件功能
streamlit run app_file_uploader.py
"""
import streamlit as st
from knowledgebase import KnowledgeBaseService
import time

st.title("文件上传服务")

uploader_file = st.file_uploader("上传文件", type=["pdf", "docx", "txt"], accept_multiple_files=False)

if "service" not in st.session_state:
    st.session_state["service"] = KnowledgeBaseService()

if uploader_file is not None:
    file_name = uploader_file.name
    file_size = uploader_file.size / 1024 # 单位为kb
    file_type = uploader_file.type

    st.subheader("文件名: {file_name}")
    st.write(f"文件大小: {file_size:.2f} kb | 文件类型: {file_type}")

    text = uploader_file.getvalue().decode("utf-8")
    st.text_area("文件内容预览", text[:200], height=300)

    with st.spinner("正在载入知识库..."):
        time.sleep(1)
        result = st.session_state["service"].upload_by_string(text, file_name)
        st.success(result)
