import streamlit as st
import os
import glob
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONF_DIR = os.path.join(BASE_DIR, "config")

select = st.selectbox("操作", ["作成・追加", "変更"])

if "作成" in select:
    name = st.text_input("名前")
    params = dict()
elif "変更" in select:
    name = st.selectbox("名前", [os.path.basename(n) for n in glob.glob(os.path.join(CONF_DIR, "*"))])

    conf_file = os.path.join(CONF_DIR, name)
    if os.path.exists(conf_file):
        with open(conf_file) as f:
            params = json.load(f)
    else:
        params = dict()

exec_name = st.text_input("実行コマンド", params.get("exec", ""))

keys = ["追加"]
keys.extend(params.keys())

key = st.selectbox("パラメタ", keys)

param = st.text_input("パラメタ名", key if key != "追加" else "")
value = st.text_input("値", value=params.get(key, ""))

add_b = st.button("パラメタ追加・変更")

if add_b:
    params[param] = value

uppdate_b = st.button("Config保存・更新")

if uppdate_b:
    params["exec"] = exec_name
    conf_file = os.path.join(CONF_DIR, name)

    with open(conf_file, 'w') as f:
        json.dump(params, f)
    
    st.toast(f"config: {name} 保存しました")


