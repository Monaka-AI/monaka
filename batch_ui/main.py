import streamlit as st
import glob
import os
import zipfile

from monaka_exec import MonakaExec, WORK_DIR, RESL_DIR, CONF_DIR, ZIPF_DIR
st.set_page_config(page_title="バッチ実行WebUI", layout="wide")

if 'exec' not in st.session_state:
    st.session_state.exec = MonakaExec()
    print("create obj")

mexec = st.session_state.exec

st.write("# バッチ実行UI")
st.write("## 実行")
conf = st.selectbox("実行設定", [os.path.basename(name) for name in glob.glob(os.path.join(CONF_DIR, "*"))])
fle = st.selectbox("対象ファイル", [os.path.basename(name) for name in glob.glob(os.path.join(WORK_DIR, "*"))])

exc = st.button("実行")

if exc:
    with st.status("Executing ..."):
        infile_name = os.path.join(WORK_DIR, fle)
        
    #mexec = MonakaExec()
    mexec.run(os.path.join(CONF_DIR, conf), infile_name)

st.write("## ステータス")
show_status = st.button("表示")
if show_status:
    status = mexec.status()

    r1, r2, r3, r4, r5, r6 = st.columns(6)
    with r1:
        st.write("id")
    with r2:
        st.write("開始")
    with r3:
        st.write("ステータス")
    with r4:
        st.write("実行設定")
    with r5:
        st.write("対象ファイル")
    with r6:
        st.write("結果取得")

    for id_, d in status.items():
        r1, r2, r3, r4, r5, r6 = st.columns(6)
        with r1:
            st.write(id_)
        with r2:
            st.write(d['start'])
        with r3:
            st.write(d['status'])
        with r4:
            st.write(os.path.basename(d['config']))
        with r5:
            st.write(d['input'])
        with r6:
            if d['status'] in ["finished", "closed"]:
                out_exists = os.path.exists(os.path.join(RESL_DIR, id_, "output"))
                target = os.path.join(ZIPF_DIR, f"{id_}.zip")
                if not os.path.exists(target):
                    with zipfile.ZipFile(target,'w') as myzip:
                        for folder, subfolders, files in os.walk(os.path.join(RESL_DIR, id_)):
                            ind = folder.index(id_)
                            print(folder, folder[ind:])
                            myzip.write(folder, folder[ind:])
                            for file in files:
                                myzip.write(os.path.join(folder, file), os.path.join(folder[ind:], file))

                data = open(target, 'rb')
                download = st.download_button("download", data, f"{id_}.zip", "application/zip", key=f"b_{id_}", disabled=not out_exists)
            else:
                download = st.button("download", disabled=True, key=f"b_{id_}")
