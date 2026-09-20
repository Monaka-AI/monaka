import streamlit as st
import glob
import os
import io 
import uuid

from monaka_exec import MonakaExec, WORK_DIR, RESL_DIR, CONF_DIR

st.write("# ファイルアップロード")
name = st.text_input("ファイル名", value="")
fle = st.file_uploader("入力ファイル", type="txt")
st.write("※ファイルの最大サイズは200MBです。それ以上の場合は、ファイルを分割して実行してください。")


exc = st.button("保存")

if exc and fle is not None:
    with st.status("Uploading data..."):
        infile_name = os.path.join(WORK_DIR, name)
        st.write("creating working file ...")
        with open(infile_name, "w") as f:
            c = 0
            for line in io.TextIOWrapper(fle, encoding="utf-8"):
                f.write(line)
                c += 1
                if c % 1000 == 0:
                    st.write(f"Wrote {c} lines ...")
                