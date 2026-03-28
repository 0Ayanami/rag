import streamlit as st
import time
import config_data as config
from rag import RagService

st.title("智能金融问答系统")
st.divider()

query = st.chat_input("请输入您的问题")

if "rag" not in st.session_state:
    st.session_state["rag"] = RagService()

if "message" not in st.session_state:
    st.session_state["message"] = [{"role": "assistant", 
                                    "content": "您好,我是您的智能金融问答系统, 我可以回答您关于金融的问题."}]

for message in st.session_state["message"]:
    st.chat_message(message["role"]).write(message["content"])

if query:
    st.chat_message("user").write(query)
    st.session_state["message"].append({"role": "user", "content": query})
    
    ai_res_list = []
    with st.spinner("正在思考中..."):
        res = st.session_state["rag"].chain.stream({"input": query}, config.session_config)

        def capture(generator, cache_list):
            for chunk in generator:
                cache_list.append(chunk)
                yield chunk
        st.chat_message("assistant").write_stream(capture(res, ai_res_list))
        st.session_state["message"].append({"role": "assistant", "content": "".join(ai_res_list)})
