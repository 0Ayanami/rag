"""
使用streamlit实现在线上传文件功能
"""
import streamlit as st

st.title("文件上传服务")

uploader_file = st.file_uploader("上传文件", type=["pdf", "docx", "txt"], accept_multiple_files=False)

if uploader_file is not None:
    file_name = uploader_file.name
    file_size = uploader_file.size / 1024 # 单位为kb
    file_type = uploader_file.type

    st.subheader("文件名: {file_name}")
    st.write(f"文件大小: {file_size:.2f} kb | 文件类型: {file_type}")

    text = uploader_file.getvalue().decode("utf-8")
    st.text_area("文件内容预览", text, height=300)
