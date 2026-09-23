"""bccwj_util.py

BCCWJ関連のファイルを処理するためのツール群をまとめたモジュール
"""

import typer
import json
import csv
import stanza

from typing import List

app = typer.Typer(pretty_exceptions_show_locals=False)

BCPEXPORT_LIST = [
    "corpusName(S)",
"file(S)",
"start(S)",
"end(S)",
"boundary(S)",
"orthToken(S)",
"pronToken(S)",
"reading(S)",
"lemma(S)",
"originalText(S)",
"pos(S)",
"sysCType(S)",
"cForm(S)",
"apply(S)",
"additionalInfo(S)",
"lid(S)",
"meaning(S)",
"UpdUser(S)",
"UpdDate(S)",
"order(S)",
"note(S)",
"open(S)",
"close(S)",
"wType(S)",
"fix(S)",
"variable(S)",
"formBase(S)",
"lemmaID(S)",
"usage(S)",
"sentenceId(S)",
"s_memo(S)",
"origChar(S)",
"pSampleID(S)",
"pStart(S)",
"orthBase(S)",
"file(L)",
"l_orthToken(L)",
"l_pos(L)",
"l_cType(L)",
"l_cForm(L)",
"l_reading(L)",
"l_lemma(L)",
"luw(L)",
"memo(L)",
"UpdUser(L)",
"UpdDate(L)",
"l_start(L)",
"l_end(L)",
"bunsetsu1(L)",
"bunsetsu2(L)",
"corpusName(L)",
"diffSuw(L)",
"l_lemmaNew(L)",
"l_readingNew(L)",
"l_orthBase(L)",
"l_formBase(L)",
"l_pronToken(L)",
"l_wType(L)",
"l_originalText(L)",
"complex(L)",
"l_meaning(L)",
"l_kanaToken(L)",
"l_formOrthBase(L)",
"l_origChar(L)",
"note(L)",
"pSampleID(L)",
"pStart(L)",
"rn"
]

@app.command()
def diff(gold: str, pred: str):
    """長単位解析前後で変わらない要素に変化があるかを調べる

    Args:
        gold (str): 入力前、あるいは正解のBCCWJ用TSV
        pred (str): 長単位解析後のBCCWJ用出力TSV
    """
    with open(gold) as f1:
        with open(pred) as f2:
            grd = csv.reader(f1, delimiter="\t")
            prd = csv.reader(f2, delimiter="\t")
            i = 0
            for rp in prd:
                p_text = "_".join(rp[:5])
                rg = next(grd)
                i += 1
                g_text = "_".join(rg[:5])
                if p_text == g_text:
                    continue
                while p_text != g_text:
                    print(f"line: {i} {g_text} {p_text}")
                    i+=1
                    rg = next(grd)
                    g_text = "_".join(rg[:5])
                    

@app.command()
def export_tsv(base: str, target: str):
    """JSON-L形式のデータをBCCWJ用のTSV形式に変換する
        * JSON-L形式のデータへの変換には、utii.py chj2jsonl を用いる
        * 学習用JSON-Lデータは 上記のJSON-Lデータを util.py chjjsonl2luwjson で変換したもの
        * 結果は標準出力に出る

    Args:
        base (str): JSON-L形式のBCCWJデータ
        target (str): 学習用JSON-Lデータ
    """
    based = dict()
    with open(base) as f:
        for line in f:
            js = json.loads(line)
            based[js["sentence"]] = js

    with open(target) as f:
        for line in f:
            js = json.loads(line)
            name = js["sentence"]
            d = based[name]
            for token in d["tokens"]:
                out = {"corpusName(S)": d["corpusName(S)"], "file(S)": d["file(S)"]}
                out.update(token)
                print("\t".join(out[k] for k in BCPEXPORT_LIST))

@app.command()
def evaluate(gold: str, pred: str, corrects :List[str]=["luw(L)"], errors :List[str]=["l_orthToken(L)"], fields: List[str]=["bunsetsu1(L)", "luw(L)", "l_orthToken(L)", "l_reading(L)", "l_pos(L)", "l_cType(L)", "l_cForm(L)"]):
    """精度評価スクリプト
        * 結果は標準出力

    Args:
        gold (str): 正解のBCCWJ用TSV
        pred (str): 解析結果のBCCWJ用TSV
        corrects (List[str], optional): 正解として表示するフィールド. Defaults to ["luw(L)"].
        errors (List[str], optional): 不正解として表示するフィールド. Defaults to ["l_orthToken(L)"].
        fields (List[str], optional): 評価対象のフィールド. Defaults to ["bunsetsu1(L)", "luw(L)", "l_orthToken(L)", "l_reading(L)", "l_pos(L)", "l_cType(L)", "l_cForm(L)"].
    """
    output = dict()
    with open(gold) as f1:
        grd = csv.reader(f1, delimiter="\t")
        with open(pred) as f2:
            prd = csv.reader(f2, delimiter="\t")
            for rg, rp in zip(grd, prd):
                crrct = False
                error = False
                for field in fields:
                    i = BCPEXPORT_LIST.index(field)
                    d = output.get(field, {"a":0 , "c": 0})
                    d["a"] +=1
                    if field in ["bunsetsu1(L)", "luw(L)"]:
                        if 'B' in rg[i]:
                            if 'B' in rp[i]:
                                if field in corrects:
                                    crrct = True
                                d["c"] += 1
                            else:
                                if field in errors:
                                    error = True
                        else:
                            if 'B' not in rp[i]:
                                if field in corrects:
                                    crrct = True
                                d["c"] += 1
                            else:
                                if field in errors:
                                    error = True

                    elif rg[i] == rp[i]:
                        d["c"] += 1
                        if field in corrects:
                            crrct = True
                    elif len(rg[i]) == 0 and rp[i] == '*':
                        d["c"] += 1
                        if field in corrects:
                            crrct = True
                    else:
                        if field in errors:
                            error = True

                    output[field] = d
                if crrct and error:
                    print("gold:", "\t".join([rg[BCPEXPORT_LIST.index(name)] for name in corrects + errors ]))
                    print("pred:", "\t".join([rp[BCPEXPORT_LIST.index(name)] for name in corrects + errors ]))
    
    for field, d in output.items():
        print(f"{field} count: {d['a']} correct: {d['c']} acc: {d['c']/d['a']}")


if __name__ == "__main__":
    app()