# -*- coding: utf-8 -*-
"""解析や解析結果の処理に関するコード群

Encoder:
    解析結果を所望のフォーマットに変更するコード群

Decoder:
    入力データを使用可能なデータに変換するコード群

Predictor:
    深層学習の推論処理に関するクラス群
"""

import os
import io
import re
import sys
import csv
import glob
import json
import torch
import ipadic
import fugashi
import numpy as np

from typing import List, Dict, Any, Optional
from registrable import Registrable

from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from conllu.models import TokenList, Token
from ufal.chu_liu_edmonds import chu_liu_edmonds

from monaka.model import LUWParserModel, init_device, is_master
from monaka.dataset import LUWJsonLDataset, LemmaJsonDataset, ChunkDepJsonLDataset
from monaka.metric import MetricReporter, SpanBasedMetricReporter
from monaka.mylogging import logger

BASE_DIR = os.path.abspath(os.path.dirname(__file__)) # monaka dir
RESC_DIR = os.path.join(BASE_DIR, "resource") # monaka/resource dir


class Decoder(Registrable):
    """入力データを使用可能なデータに変換するコードの基底クラス
    """

    def __init__(self) -> None:
        super().__init__()


    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.decode(*args, **kwds)

    def luw_pos(self, text: str, pos_level: int) -> str:
        pos: List = list()
        for token in text.split("_"):
            pos.extend([t for t in token.split("-") if len(t) > 0])
        if pos_level is not None and pos_level > -1:
            return "-".join(pos[:pos_level])
        return "-".join(pos)
    
    def decode(self, tokens: List[str], pos: List[str], labels: List[str], **kwargs) -> Dict:
        """入力データを使用可能なデータに変換する処理
        
        Args:
            tokens (List[str]):
                短単位語の列
            pos (List[str]):
                短単位の品詞の列
            labels (List[str]):
                推論対象のラベルの列（1短単位あたり1ラベル）
        Returns:
            Dict 出力は辞書形式 LUWは開始位置の場合はPOS-tag名、そうでない場合は"*"。文節は開始位置は"B"そうでなければ"I"。
                解析対象のfieldを含んでいれば良い。
                kwargsにメタ情報を追記でき、それらを辞書に加えることを想定している:

                    {
                        "luw": ["POS-tag or *"],
            
                        "chunk": ["B", "I"]
                    }
        """
        raise NotImplementedError


class DepDecoder(Registrable):
    """係り受け解析の入力データを使用可能なデータに変換するコードの基底クラス
    """

    def __init__(self) -> None:
        super().__init__()


    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.decode(*args, **kwds)

    
    def decode(self, sent: str, **kwargs) -> Dict:
        """入力データを使用可能なデータに変換する処理
       
        Args:
            sent (str):
                入力1行のテキスト

        Returns:
            Dict:
                :py:class:`~monaka.dataset.ChunkDepJsonLDataset` 準拠のデータ
        """
        raise NotImplementedError


@DepDecoder.register("jsonl")
class DepJsonL(DepDecoder):
    """係り受け解析用のJSON-L データをモデル入力用に変換するコード

    :py:class:`~Registrable`で呼び出す際は"jsonl"
    """

    def decode(self, sent, **kwargs):
        return json.loads(sent)


@Decoder.register("LUW-Bunsetsu")
class LUWChunkDecoder(Decoder):
    """長単位品詞・境界と文節境界をすべて一つのラベルにする方式

    :py:class:`~Registrable`で呼び出す際は"LUW-Bunsetsu"
    """

    @staticmethod
    def is_space(luw_pos: str) -> bool:
        """空白の判定

        Args:
            luw_pos (str): 長単位品詞

        Returns:
            bool: 空白である(True)
        """
        if luw_pos.startswith("補助記号"):
            return True
        elif luw_pos.startswith("空白"):
            return True
        return False
    
    @staticmethod
    def is_passthrough(pos: str) -> bool:
        """そのまま出力するか

        Args:
            pos (str): 短単位品詞

        Returns:
            bool: そのまま出力(True)
        """
        if '漢文' in pos:
            return True
        return False

    def decode(self, tokens: List[str], pos: List[str], labels: List[str], pos_level:int = -1, **kwargs) -> Dict:
        """
        labelsが以下の形式の場合に利用する
        (B or I)(B or I)(LUW品詞)

        Args:
            tokens (List[str]):
                短単位語の列
            pos (List[str]):
                短単位の品詞の列
            labels (List[str]):
                推論対象のラベルの列（1短単位あたり1ラベル）
            pos_level (int):
                推論する長単位品詞の階層 名詞-固有名詞-人名 の様な場合は3階層ある

        Returns:
            Dict:
                :py:class:`~monaka.dataset.LUWJsonLDataset` で読み込めるデータ形式に変換
                
        """
        luw = list()
        chunk = list()
        begin_with_space = True
        for l, p in zip(labels, pos):
            if self.is_passthrough(p): #SUWをそのまま出力するシリーズ
                chunk.append('B')
                luw.append(p)
                continue

            lpos = self.luw_pos(l[2:], pos_level)
            if begin_with_space:
                chunk.append("I")

                if l in ["unk", "pad"]:
                    luw.append("*")
                else:
                    luw.append(lpos)

                begin_with_space = self.is_space(lpos)
                continue
            
            if l in ["unk", "pad"]:
                chunk.append("B")
                luw.append("*")
            else:
                chunk.append(l[0])
                if l[1] == "B":
                    luw.append(lpos)
                else:
                    luw.append("*")
        if len(chunk) > 0: # 銭湯は必ずB
            chunk[0] = 'B'
        res = {
            "tokens": tokens,
            "pos": pos,
            "luw": luw[:len(tokens)],
            "chunk": chunk[:len(tokens)]
        }
        res.update(kwargs)

        return res
    

@Decoder.register("comainu")
class ComainuDecoder(Decoder):
    """Comainu準拠のラベルにする方式

    :py:class:`~Registrable`で呼び出す際は"comainu"
    """

    def decode(self, tokens: List[str], pos: List[str], labels: List[str], pos_level:int = -1, **kwargs) -> Dict:
        """
        labelsが以下の形式の場合に利用する Comainu方式
        (B or I)(B or I)(a)

        Args:
            tokens (List[str]):
                短単位語の列
            pos (List[str]):
                短単位の品詞の列
            labels (List[str]):
                推論対象のラベルの列（1短単位あたり1ラベル）
            pos_level (int):
                推論する長単位品詞の階層 名詞-固有名詞-人名 の様な場合は3階層ある

        Returns:
            Dict:
                :py:class:`~monaka.dataset.LUWJsonLDataset` で読み込めるデータ形式に変換
        """
        luw = list()
        chunk = list()
        prv = -1
        for l, p in zip(labels, pos):
            if l in ["unk", "pad"]:
                chunk.append("B")
                luw.append(self.luw_pos(p, pos_level))
            else:
                chunk.append(l[0])
                if l[1] == "B":
                    luw.append(p)
                    prv = len(luw) -1
                elif l[1:] == "Ia":
                    if prv > 0:
                        luw[prv] = p
                    luw.append("*")
                else:
                    luw.append("*")
        res = {
            "tokens": tokens,
            "pos": pos,
            "luw": luw,
            "chunk": chunk
        }
        res.update(kwargs)

        return res


class Encoder(Registrable):
    """解析結果を所望のフォーマットに変更するコード
    """

    def __init__(self, **kwargs) -> None:
        super().__init__()

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.encode(*args, **kwds)

    def encode(self, tokens: List[str], pos: List[str], **kwargs) -> Any:
        """
        モデルが出力する形式を受け取って、所望の出力形式に変換する

        Args:
            tokens (List[str]):
                短単位語の列
            pos (List[str]):
                短単位の品詞の列

        Returns:
            Any: 変換結果
        """
        raise NotImplementedError
    

class DepEncoder(Registrable):
    """係り受け解析結果を所望のフォーマットに変更するコード
    """

    def __init__(self, **kwargs) -> None:
        super().__init__()

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.encode(*args, **kwds)

    def encode(self, original: Dict, **kwargs) -> Any:
        """
        モデルが出力する形式を受け取って、所望の出力形式に変換する

        Args:
            original (Dict):
                モデル推論結果の辞書データ

        Returns:
            Any:
                変換後の形式
        """
        raise NotImplementedError


@DepEncoder.register("jsonl")
class DepPathThrough(DepEncoder):
    """そのまま(JSON-L)で出力する係り受け解析用のデコーダ

    :py:class:`~Registrable`で呼び出す際は"jsonl"
    """

    def encode(self, original, **kwargs):
        return json.dumps(original, ensure_ascii=False)
    

@DepEncoder.register("ud")
class UDDepEncoder(DepEncoder):
    """UD形式に変換する係り受け解析用のデコーダ

    :py:class:`~Registrable`で呼び出す際は"ud"
    """

    def encode(self, original, **kwargs) -> str:
        """
        モデルが出力する形式を受け取って、UD形式(conllu形式)に変換する

        Args:
            original (Dict):
                モデル推論結果の辞書データ

        Returns:
            str:
                UD形式のテキスト
        """
        if 'lemma' not in original or original['lemma'][0] == '_':
            original['lemma'] = original['tokens']

        if 'upos' not in original or original['upos'][0] == '_':
            original['upos'] = original['pos']

        if 'misc' not in original or original['misc'][0] == '_':
            prv = -1
            feats = list()
            for b in original['bid']:
                if prv != b:
                    feats.append({'BunsetuBILabel': 'B', 'SpaceAfter': 'No'})
                    prv = b
                else:
                    feats.append({'BunsetuBILabel': 'I', 'SpaceAfter': 'No'})
            original['misc'] = feats

        token_list = [{"id": i+1, "form": t, "lemma": l, "upos": u, "xpos":p, "feats": "_", "head": 0, "deprel": r, "deps": "_", "misc": {}} 
            for i,(t,p,r,l,u) in enumerate(zip(original['tokens'], original['pos'], original['rel'], original['lemma'], original['upos']))]

        for t, f in zip(token_list, original['misc']):
            t['misc'] = f

        bid = np.array(original['bid'])
        rel = np.array(original['rel'])
        indices = np.arange(len(bid))
        hids = list()
        for i in range(np.max(bid)+1):
            ind = np.where(bid == i)
            hid = indices[ind][np.where(rel[ind] == 'shead')][0]
            hids.append(hid)
            for j in ind[0]:
                if j == hid:
                    continue
                token_list[j]['head'] = hid + 1
        
        for dep in original['dependency']:
            id_ = dep['id']
            i = hids[id_]
            h = dep['head']
            if  id_ == h or dep['rel'] == 'root':
                token_list[i]['head'] = 0
                token_list[i]['deprel'] = 'root'
            elif h < len(hids):
                #print(i, h)
                #print(len(token_list), len(hids))
                token_list[i]['head'] = hids[h] + 1
                token_list[i]['deprel'] = dep['rel'] if dep['rel'] != 'unk' else 'nmod'
            else:
                token_list[i]['head'] = hids[-1] + 1
                token_list[i]['deprel'] = dep['rel']if dep['rel'] != 'unk' else 'nmod'
    
        for t in token_list[1:]:
            if t['deprel'] == 'fixed':
                t['head'] = t['id'] -1
            if t['deprel'] == 'unk':
                t['deprel'] = 'nmod'

        tlist = TokenList([Token(**t) for t in token_list])
        tlist.metadata['sent_id'] = original['sent_id']
        tlist.metadata['text'] = original['text']

        return tlist.serialize()[:-1]


@DepEncoder.register("cabocha")
class CabochaDepEncoder(DepEncoder):
    """CaboCha形式に変換する係り受け解析用のデコーダ

    :py:class:`~Registrable`で呼び出す際は"cabocha"
    """

    def encode(self, original, **kwargs):
        """
        モデルが出力する形式を受け取って、CaboCha形式に変換する

        Args:
            original (Dict):
                モデル推論結果の辞書データ

        Returns:
            str:
                CaboCha形式のテキスト
        """
        
        pbid = -1
        out = list()
        out.append(f'#! DOCATTR	<sent_id># sent_id = {original["sent_id"]}</sent_id>')
        deps = original['dependency']
        for bid, token, feat in zip(original['bid'], original['tokens'], original['unidic']):
            if bid != pbid:
                pbid = bid
                head = deps[bid]['head']
                if head == bid or head >= len(deps):
                    head = -1
                out.append(f'* {bid} {head}D')
            feat = [f.replace(',', '，') for f in feat]
            out.append(f"{token.strip()}\t{','.join(feat)}")
        out.append('EOS')
        return '\n'.join(out)
            

def append_spans(data: Dict) -> Dict:
    """スパン情報を付与する

    Args:
        data (Dict): 入力データの1レコード

    Returns:
        Dict: 
            入力データに以下を追加して返す:
                短単位スパン(suw_span): (begin, end)のリスト
                長単位スパン(luw_span): (begin, end)のリスト
                文節のスパン(chunk_span): (begin, end)のリスト
                長単位のトリプル(luw_triples): (begin, end, 品詞)のリスト
    """
    start = 0
    data["suw_span"] = []
    for token in data["tokens"]:
        data["suw_span"].append((start, start+len(token)))
        start += len(token)

    data["luw_span"] = []
    data["luw_triples"] = []
    data["chunk_span"] = []

    luw_start = -1
    luw_end = -1
    luw_type = None
    chunk_start = -1
    chunk_end = -1
    for span, luw, chunk in zip(data["suw_span"], data["luw"], data["chunk"]):
        if luw_start < 0:
            luw_start = span[0]
            luw_end = span[1]
            chunk_start = span[0]
            chunk_end = span[1]
            luw_type = luw
            continue
        if "*" in luw:
            luw_end = span[1]
        else:
            data["luw_span"].append((luw_start, luw_end))
            data["luw_triples"].append((luw_start, luw_end, luw_type))
            luw_start = span[0]
            luw_end = span[1]
            luw_type = luw
        
        if "I" in chunk:
            chunk_end = span[1]
        else:
            data["chunk_span"].append((chunk_start, chunk_end))
            chunk_start = span[0]
            chunk_end = span[1]

    data["chunk_span"].append((chunk_start, chunk_end))
    data["luw_span"].append((luw_start, luw_end))
    data["luw_triples"].append((luw_start, luw_end, luw_type))

    return data


@Encoder.register("jsonl")
class PassThrough(Encoder):
    """そのまま(JSON-L)で出力するデコーダ

    :py:class:`~Registrable`で呼び出す際は"jsonl"
    """

    def encode(self, tokens: List[str], pos: List[str], **kwargs) -> str:
        """
        モデルが出力する形式を受け取って、そのままJSON-L形式に変換する

        Args:
            tokens (List[str]):
                短単位語の列
            pos (List[str]):
                短単位の品詞の列

        Returns:
            str: 変換結果のJSON(1行)
        """
        r = {"tokens": tokens, "pos": pos}
        r.update(kwargs)
        append_spans(r)
        return json.dumps(r, ensure_ascii=False)


@Encoder.register("bunsetsu-split")
class BunsetsuSplitter(Encoder):
    """文節区切りをするデコーダ

    :py:class:`~Registrable`で呼び出す際は"bunsetsu-split"
    """

    def encode(self, tokens: List[str], pos: List[str], chunk: List[str], **kwargs) -> str:
        """
        モデルが出力する形式を受け取って、文節区切りのテキストに変換する

        Args:
            tokens (List[str]):
                短単位語の列
            pos (List[str]):
                短単位の品詞の列
            chunk (List[str]):
                文節のBIタグのリスト

        Returns:
            str: 文節区切りのテキスト
        """
        c_tokens = list()
        prv = ""
        for token, c in zip(tokens, chunk):
            if c == "B":
                if len(prv) > 0:
                    c_tokens.append(prv)
                prv = token
            else:
                prv += token

        if len(prv) > 0:
            c_tokens.append(prv)

        return " ".join(c_tokens)
    

@Encoder.register("csv")
class CSVEncoder(Encoder):
    """CSV出力デコーダ

    :py:class:`~Registrable`で呼び出す際は"csv"
    """

    def encode(self, tokens: List[str], pos: List[str], chunk: List[str], features: List[List[str]], **kwargs) -> str:
        """
        モデルが出力する形式を受け取って、CSVテキストに変換する

        Args:
            tokens (List[str]):
                短単位語の列
            pos (List[str]):
                短単位の品詞の列
            chunk (List[str]):
                文節のBIタグのリスト

        Returns:
            str: 短単位, 品詞, X, 長単位境界, 文節境界
        """
        output = io.StringIO()
        writer = csv.writer(output)
        if "luw" in kwargs:
            lpos = kwargs["luw"]
        else:
            lpos = pos

        for token, p, l, c, f in zip(tokens, pos, lpos, chunk, features):
            writer.writerow((token, p, f[5], l, c))

        return output.getvalue()
    
@Encoder.register("mecab")
class MeCabEncoder(Encoder):
    """MeCab準拠のデコーダ

    :py:class:`~Registrable`で呼び出す際は"mecab"

    Args:
        node_format (str, optional): 
            短単位レベルの出力のフォーマット文. Defaults to '%m\t%f[9]\t%f[6]\t%f[7]\t%F-[0,1,2,3]\t%f[4]\t%f[5]\t%f[13]\t%f[27]\t%f[28]\n'.
        bos_format (str, optional): 
            文頭に付与する表現. Defaults to ''.
        eos_format (str, optional): 
            文末に付与する表現. Defaults to '\n'.
        unk_format (str, optional): 
            未知語のフォーマット文. Defaults to '%m\t%m\t%m\t%m\tUNK\t%f[4]\t%f[5]\t\n'.
        eon_format (str, optional): 
            トークン終わりに付与する表現. Defaults to ''.
    """
    
    def __init__(self, node_format: str='%m\t%f[9]\t%f[6]\t%f[7]\t%F-[0,1,2,3]\t%f[4]\t%f[5]\t%f[13]\t%f[27]\t%f[28]\n', bos_format: str='', eos_format: str='\n', unk_format: str='%m\t%m\t%m\t%m\tUNK\t%f[4]\t%f[5]\t\n', eon_format: str='', **kwargs) -> None:
        super().__init__(**kwargs)
        self.node_format = node_format
        self.bos_format = bos_format
        self.eos_format = eos_format
        self.unk_format = unk_format
        self.eon_format = eon_format

        self.space_matcher = re.compile(r'^(\s+)')
        self.f_matcher = re.compile(r'\%f\[([\d\,]+)\]')
        self.fc_matcher = re.compile(r'\%F(.+)\[([\d\,]+)\]')

    @staticmethod
    def is_yougen(pos: str) -> bool:
        """用言であるかを判定する

        Args:
            pos (str): 短単位品詞

        Returns:
            bool: 用言である(True)
        """
        if '動詞' in pos: #動詞 助動詞
            return True
        if '形容詞' in pos:
            return True
        if '形状詞' in pos:
            return True
        return False
    
    def format(self, fstring: str, sentence: str, m_type, token: str, p, l, c, feat: List[str], start: int)-> str:
        """
        フォーマットは以下

        Examples:
            %s	形態素種類 (0: 通常, 1: 未知語, 2:文頭, 3:文末)

            %S	入力文

            %L	入力文の長さ

            %m	形態素の表層文字列

            %M	形態素の表層文字列, ただし空白文字も含めて出力 (%pS を参照のこと)

            %h	素性の内部 ID

            %%	% そのもの

            %c	単語生起コスト

            %H	素性 (品詞, 活用, 読み) 等を CSV で表現したもの

            %t	文字種 id

            %P	周辺確率 (-l2 オプションを指定したときのみ有効)

            %pi	形態素に付与されるユニークなID

            %pS	もし形態素が空白文字列で始まる場合は, その空白文字列を表示 %pS%m と %M は同一

            %ps	開始位置

            %pe	終了位置

            %pC	1つ前の形態素との連接コスト

            %pw	%c と同じ

            %pc	連接コスト + 単語生起コスト (文頭から累積)

            %pn	連接コスト + 単語生起コスト (その形態素単独, %pw + %pC)

            %pb	最適パスの場合 *, それ以外は ' '

            %pP	周辺確率 (-l2 オプションを指定したときのみ有効)

            %pA	blpha, forward log 確率 (-l2 オプションを指定したときのみ有効)

            %pB	beta, backward log 確率 (-l2 オプションを指定したときのみ有効)

            %pl	形態素の表層文字列としての長さ, strlen (%m) と同一

            %pL	形態素の表層文字列としての長さ, ただし空白文字列も含む, strlen(%M) と同一

            %phl	左文脈 id

            %phr	右文脈 id

            %b  文節情報

            %l  長単位境界

            %lP 長単位品詞

            %lO 長単位OrthToken

            %lR 長単位読み

            %lT 長単位活用型

            %lF 長単位活用形

            %lL 長単位語彙素

            %f[N]	csv で表記された素性の N番目の要素

            %f[N1,N2,N3...]	N1,N2,N3番目の素性を, "," を デリミタとして表示

            %FC[N1,N2,N3...]	N1,N2,N3番目の素性を, C を デリミタとして表示.

            ただし, 要素が 空の場合は以降表示が省略される. (例)F-[0,1,2]

            "¥0 ¥a ¥b ¥t ¥n ¥v ¥f ¥r ¥¥"	通常の エスケープ文字列

            "¥s"	' ' (半角スペース)

            設定ファイルに記述するときに使用
        """
        f = list()
        f.extend(feat)
        #f.extend((l, c))

        output = fstring.replace('\\0', '\0')
        output = output.replace('\\a', '\a')
        output = output.replace('\\b', '\b')
        output = output.replace('\\t', '\t')
        output = output.replace('\\n', '\n')   
        output = output.replace('\\v', '\v')
        output = output.replace('\\f', '\f')
        output = output.replace('\\r', '\r')
        output = output.replace('\\\\', '\\')

        output = output.replace('%S', sentence)
        output = output.replace('%s', str(m_type))
        output = output.replace('%L', str(len(sentence)))
        output = output.replace('%m', token.strip())
        output = output.replace('%M', token)
        output = output.replace('%h', '0')
        output = output.replace('%%', '%')
        output = output.replace('%c', '1')
        output = output.replace('%H', ','.join(f))
        output = output.replace('%t', '0')
        output = output.replace('%P', '0')
        output = output.replace('%pi', '0')
        # %pS
        m = self.space_matcher.match(token)
        if m:
            output = output.replace('%pS', m.group(1))
        else:
            output = output.replace('%pS', '')

        output = output.replace('%ps', str(start))
        output = output.replace('%pe', str(start + len(token)))
        output = output.replace('%pC', '0')
        output = output.replace('%pw', '0')
        output = output.replace('%pc', '0')
        output = output.replace('%pn', '0')
        output = output.replace('%pC', '*')
        output = output.replace('%pP', '0')
        output = output.replace('%pB', '0')
        output = output.replace('%pl', str(len(token.strip())))
        output = output.replace('%pL', str(len(token)))
        output = output.replace('%phl', '0')
        output = output.replace('%phr', '0')
        output = output.replace('%b', c)
        output = output.replace('%lP', l.get('l_pos',''))
        output = output.replace('%lR', l.get('l_reading', ''))
        output = output.replace('%lO', l.get('l_orthToken', ''))
        output = output.replace('%lT', l.get('l_cType',''))
        output = output.replace('%lF', l.get('l_cForm', ''))
        output = output.replace('%lL', l.get('l_lemma', ''))
        output = output.replace('%l', l.get('LUW', ''))
        output = output.replace('\s', ' ')
        
        for m in self.f_matcher.finditer(output):
            txt = m.group(0)
            index = [int(v) for v in m.group(1).split(',')]
            vals = [f[i] if i < len(f) else '*' for i in index]
            output = output.replace(txt, ','.join(vals), 1)

        for m in self.fc_matcher.finditer(output):
            txt = m.group(0)
            sep = m.group(1)
            index = [int(v) for v in m.group(2).split(',')]
            vals = [f[i] for i in index if i < len(f) and f[i] not in ('', '*')]
            output = output.replace(txt, sep.join(vals), 1)

        return output

    def encode(self, tokens: List[str], pos: List[str], chunk: List[str], features: List[List[str]], **kwargs) -> str:
        """
        モデルが出力する形式を受け取って、MeCabの形式で出力する

        Args:
            tokens (List[str]):
                短単位語の列
            pos (List[str]):
                短単位の品詞の列
            chunk (List[str]):
                文節のBIタグのリスト
            features (List[List[str]]):
                辞書のフィールドデータ

        Returns:
            str: MeCab準拠のテキスト
        """
        output = ''
        if "luw" in kwargs:
            lpos = kwargs["luw"]
        else:
            lpos = pos

        output += self.format(self.bos_format, kwargs.get('sentence', ''), 0, '', '', {}, '', [], 0)
        if len(features) > 0:
            if len(features[0]) < 15: # ipadic:
                mapping = {
                    "reading": 7,
                    "lemma": 6,
                    "cType": 4,
                    "cForm": 5,
                    "orthToken": 6
                }
                max_mapping = 9
            else: #UniDic
                mapping = {
                    "reading": 9,
                    "lemma": 7,
                    "cType": 4,
                    "cForm": 5,
                    "orthToken": 8
                }
                max_mapping = 10
        lfeats = list()
        start = -1
        count = 0
        pos_ = None
        for l, f, t in zip(lpos, features, tokens):
            f = list(f)
            while len(f) < max_mapping:
                f.append(t)
            out = dict()
            if start < 0 or '*' not in l: #長単位先頭
                out["LUW"] = 'B'
                out["l_orthToken"] = f[mapping.get('orthToken', 0)]
                out["l_reading"] = f[mapping.get('reading', 0)]
                out["l_pos"] = l

                pos_ = l
                # 用言のみ活用情報を追記
                if self.is_yougen(pos_):
                    out["l_cType"] = f[mapping.get('cType', 0)]
                    out["l_cForm"] = f[mapping.get('cForm', 0)]
                else:
                    out["l_cType"] = ""
                    out["l_cForm"] = ""
                start = count
            else: #長単位途中
                out["LUW"] = 'I'
                lfeats[start]["l_orthToken"] += f[mapping.get('orthToken', 0)]# 長単位先頭のトークンに追記
                out["l_orthToken"] = "*"
                lfeats[start]["l_reading"] += f[mapping.get('reading', 0)] # 長単位先頭のトークンに追記
                out["l_reading"] = "*"
                out["l_pos"] = "*"
                # 用言のみ活用情報を追記
                if self.is_yougen(pos_):
                    lfeats[start]["l_cType"] = f[mapping.get('cType', 0)] # 長単位先頭のトークンを上書き
                    lfeats[start]["l_cForm"] = f[mapping.get('cForm', 0)] # 長単位先頭のトークンを上書き
                out["l_cType"] = "*"
                out["l_cForm"] = "*"

            lfeats.append(out)
            count += 1

        start = 0
        for i, (token, p, l, c, f) in enumerate(zip(tokens, pos, lfeats, chunk, features)):
            m_type = 0
            if i == 0:
                m_type = 2
            elif i == len(tokens) -1:
                m_type = 3
            output += self.format(self.node_format, kwargs.get('sentence', ''), m_type, token, p, l, c, f, start)
            start += len(token)

        output += self.format(self.eos_format, kwargs.get('sentence', ''), 0, '', '', {}, '', [], len(tokens))
        return output
    
    
@Encoder.register("cabocha")
class CaboChaEncoder(Encoder):
    """CaboCha形式に変換するデコーダ

    :py:class:`~Registrable`で呼び出す際は"cabocha"
    """

    def encode(self, tokens: List[str], pos: List[str], chunk: List[str],  features: List[List[str]], **kwargs) -> str:
        """
        モデルが出力する形式を受け取って、長単位区切りのテキストに変換する

        Args:
            tokens (List[str]):
                短単位語の列
            pos (List[str]):
                短単位の品詞の列
            chunk (List[str]):
                文節のBIタグのリスト
            features (List[List[str]]):
                辞書のフィールドデータ

        Returns:
            str: CaboCha形式のテキスト
        """

        if "luw" in kwargs:
            lpos = kwargs["luw"]
        else:
            lpos = pos

        chunk_count = 0
        outputs = list()

        luws = list()
        luw_read = list()
        luw = ""
        luw_r = ""
        for token, p, feat in zip(tokens, lpos, features):
            if p == '*':
                luw += token
                luw_r += feat[9] if len(feat) > 10 else ''
            else:
                if len(luw) > 0:
                    luws.append(luw)
                    luw_read.append(luw_r)
                luw = token
                luw_r = feat[9] if len(feat) > 10 else ''
        if len(luw) > 0:
            luws.append(luw)
            luw_read.append(luw_r)

        luw_c = 0
        for token, c, p, feat in zip(tokens, chunk, lpos, features):
            if chunk_count < 1:
                outputs.append("* 0 -1D")
                chunk_count += 1
                c = 'B'
            elif 'B' in c:
                outputs.append(f"* {chunk_count} -1D")
                chunk_count += 1
            pos_tokens = ['*' for _ in range(4)]
            pos_tokens.extend(['' for _ in range(4)])
            if p == '*':
                if c == 'B':
                    luw = token
                    for i, f in enumerate(feat[:4]):
                        pos_tokens[i] = f
                else:
                    luw = ""
                    pos_tokens = pos_tokens[1:]
            else:
                luw = luws[luw_c]
                luw_c += 1
                for i, t in enumerate(p.split('-')):
                    pos_tokens[i] = t
            c = '' if 'I' in c else 'B'

            feat = [f.replace(',', '，') for f in feat]
            if len(feat) < 29:
                L = 29 -len(feat)
                feat.extend(['' for _ in range(L)])

            outputs.append(f"{token.strip()}\t{','.join(feat)}\t{luw}\t{','.join(pos_tokens)}\t{c}")
        outputs.append('EOS')
        return '\n'.join(outputs)
    

@Encoder.register("luw-split")
class LUWSplitter(Encoder):
    """長単位区切りをするデコーダ

    :py:class:`~Registrable`で呼び出す際は"luw-split"
    """

    def encode(self, tokens: List[str], pos: List[str], chunk: List[str], **kwargs) -> str:
        """
        モデルが出力する形式を受け取って、長単位区切りのテキストに変換する

        Args:
            tokens (List[str]):
                短単位語の列
            pos (List[str]):
                短単位の品詞の列
            chunk (List[str]):
                文節のBIタグのリスト

        Returns:
            str: 長単位区切りのテキスト
        """
        c_tokens = list()
        prv = ""
        if "luw" in kwargs:
            lpos = kwargs["luw"]
        else:
            lpos = pos

        for token, c in zip(tokens, lpos):
            if c != "*":
                if len(prv) > 0:
                    c_tokens.append(prv)
                prv = token

            else:
                prv += token

        if len(prv) > 0:
            c_tokens.append(prv)
            
        return " ".join(c_tokens)

@Encoder.register("bccwj")
class BCCWJComainu(Encoder):
    FIELDS = [
        "file(S)",
        "start(S)",
        "end(S)",
        "boundary(S)",
        "orthToken(S)",
        "reading(S)",
        "lemma(S)",
        "meaning(S)",
        "pos(S)",
        "cType(S)",
        "cForm(S)",
        "usage(S)",
        "pronToken(S)",
        "pronBase(S)",
        "kana(S)",
        "kanaBase(S)",
        "form(S)",
        "formBase(S)",
        "formOrthBase(S)",
        "formOrth(S)",
        "orthBase(S)",
        "wType(S)",
        "charEncloserOpen(S)",
        "charEncloserClose(S)",
        "originalText(S)",
        "order",
        "BOB",
        "LUW",
        "l_orthToken",
        "l_reading",
        "l_lemma",
        "l_pos",
        "l_cType",
        "l_cForm",
        "depend",
        "MID",
        "m"
    ]
    """出力のTSVの列の順"""

    @staticmethod
    def is_yougen(pos: str) -> bool:
        """用言であるかを判定する

        Args:
            pos (str): 短単位品詞

        Returns:
            bool: 用言である(True)
        """
        if '動詞' in pos: #動詞 助動詞
            return True
        if '形容詞' in pos:
            return True
        if '形状詞' in pos:
            return True
        return False
    
    def encode(self, tokens: List[str], pos: List[str], chunk: List[str], **kwargs) -> str:
        """
        モデルが出力する形式を受け取って、BCCWJの形式で出力する

        Args:
            tokens (List[str]):
                短単位語の列
            pos (List[str]):
                短単位の品詞の列
            chunk (List[str]):
                文節のBIタグのリスト

        Returns:
            str: TSV形式のテキスト
        """
        if "luw" in kwargs:
            lpos = kwargs["luw"]
        else:
            lpos = pos

        meta = kwargs.get("meta")
        outs = list()
        start = -1
        count = 0
        pos_ = None
        for p, l, c, m in zip(pos, lpos, chunk, meta):
            out = [m.get(name, "") for name in self.FIELDS]
            out[self.FIELDS.index("BOB")] = c
            if start < 0 or '*' not in l: #長単位先頭
                out[self.FIELDS.index("LUW")] = 'B'
                out[self.FIELDS.index("l_orthToken")] = m['orthToken(S)']
                out[self.FIELDS.index("l_reading")] = m['reading(S)']
                out[self.FIELDS.index("l_pos")] = l

                pos_ = l
                # 用言のみ活用情報を追記
                if self.is_yougen(pos_):
                    out[self.FIELDS.index("l_cType")] = m['cType(S)']
                    out[self.FIELDS.index("l_cForm")] = m['cForm(S)']
                else:
                    out[self.FIELDS.index("l_cType")] = ""
                    out[self.FIELDS.index("l_cForm")] = ""
                start = count
            else: #長単位途中
                out[self.FIELDS.index("LUW")] = 'I'
                outs[start][self.FIELDS.index("l_orthToken")] += m['orthToken(S)'] # 長単位先頭のトークンに追記
                out[self.FIELDS.index("l_orthToken")] = "*"
                outs[start][self.FIELDS.index("l_reading")] += m['reading(S)'] # 長単位先頭のトークンに追記
                out[self.FIELDS.index("l_reading")] = "*"
                out[self.FIELDS.index("l_pos")] = "*"
                # 用言のみ活用情報を追記
                if self.is_yougen(pos_):
                    outs[start][self.FIELDS.index("l_cType")] = m['cType(S)'] # 長単位先頭のトークンを上書き
                    outs[start][self.FIELDS.index("l_cForm")] = m['cForm(S)'] # 長単位先頭のトークンを上書き¥
                out[self.FIELDS.index("l_cType")] = "*"
                out[self.FIELDS.index("l_cForm")] = "*"

            outs.append(out)
            count += 1
        return "\n".join(['\t'.join(o) for o in outs])


@Encoder.register("bcpexport")
class BCPExport(Encoder):
    """BCCWJ2の開発のための入出力フォーマットに準拠した出力
    
    """
    FIELDS = [
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
    """出力のTSVの列の順"""

    @staticmethod
    def is_yougen(pos: str) -> bool:
        """用言であるかを判定する

        Args:
            pos (str): 短単位品詞

        Returns:
            bool: 用言である(True)
        """
        if '動詞' in pos: #動詞 助動詞
            return True
        if '形容詞' in pos:
            return True
        if '形状詞' in pos:
            return True
        return False

    def encode(self, tokens: List[str], pos: List[str], chunk: List[str], **kwargs) -> str:
        """
        モデルが出力する形式を受け取って、BCCWJ2の形式で出力する

        Args:
            tokens (List[str]):
                短単位語の列
            pos (List[str]):
                短単位の品詞の列
            chunk (List[str]):
                文節のBIタグのリスト

        Returns:
            str: TSV形式のテキスト
        """
        if "luw" in kwargs:
            lpos = kwargs["luw"]
        else:
            lpos = pos

        meta = kwargs.get("meta")
        outs = list()
        start = -1
        count = 0
        pos_ = None
        for p, l, c, m in zip(pos, lpos, chunk, meta):
            out = [m.get(name, "") for name in self.FIELDS]
            out[self.FIELDS.index("bunsetsu1(L)")] = c
            if start < 0 or '*' not in l: #長単位先頭
                out[self.FIELDS.index("luw(L)")] = 'B'
                out[self.FIELDS.index("l_orthToken(L)")] = m['orthToken(S)']
                out[self.FIELDS.index("l_reading(L)")] = m['reading(S)']
                out[self.FIELDS.index("l_pos(L)")] = l
                pos_ = l
                # 用言のみ活用情報を追記
                if self.is_yougen(pos_):
                    out[self.FIELDS.index("l_cType(L)")] = m['sysCType(S)']
                    out[self.FIELDS.index("l_cForm(L)")] = m['cForm(S)']
                else:
                    out[self.FIELDS.index("l_cType(L)")] = ""
                    out[self.FIELDS.index("l_cForm(L)")] = ""

                start = count
            else: #長単位途中
                out[self.FIELDS.index("luw(L)")] = 'I'
                outs[start][self.FIELDS.index("l_orthToken(L)")] += m['orthToken(S)'] # 長単位先頭のトークンに追記
                out[self.FIELDS.index("l_orthToken(L)")] = "*"
                outs[start][self.FIELDS.index("l_reading(L)")] += m['reading(S)'] # 長単位先頭のトークンに追記
                out[self.FIELDS.index("l_reading(L)")] = "*"
                out[self.FIELDS.index("l_pos(L)")] = "*"
                # 用言のみ活用情報を追記
                if self.is_yougen(pos_):
                    outs[start][self.FIELDS.index("l_cType(L)")] = m['sysCType(S)'] # 長単位先頭のトークンを上書き
                    outs[start][self.FIELDS.index("l_cForm(L)")] = m['cForm(S)'] # 長単位先頭のトークンを上書き
                out[self.FIELDS.index("l_cType(L)")] = "*"
                out[self.FIELDS.index("l_cForm(L)")] = "*"

            outs.append(out)
            count += 1
        return "\n".join(['\t'.join(o) for o in outs])


class SUWTokenizer(Registrable):

    def __init__(self, **kwargs) -> None:
        super().__init__()

    def tokenize(self, sentence: str) -> Dict:
        """
        以下のフォーマットを必ず含む結果を返す。（他にフィールドがあっても良い)
        {
            "sentence": "",
            "tokens": [],
            "pos": []
        }
        """
        raise NotImplementedError
    

@SUWTokenizer.register("mecab")
class MecabSUWTokenizer(SUWTokenizer):

    def __init__(self, 
        dic: Optional[str] = "gendai", dic_path: Optional[str]=None) -> None:
        super().__init__()
        if dic_path is None:
            dic_path = RESC_DIR
        dicdir = os.path.join(dic_path, dic)
        mecabrc = os.path.join(RESC_DIR, "mecabrc")
        mecab_option = f"-r {mecabrc} -d {dicdir}"
        if dic == 'ipadic':
            self.mecab = fugashi.GenericTagger(ipadic.MECAB_ARGS)
        else:
            self.mecab = fugashi.GenericTagger(mecab_option)

    @staticmethod
    def get_pos(feature):
        res = list()
        for feat in feature[:5]:
            if "*" not in feat:
                res.append(feat)
        return "-".join(res)


    def tokenize(self, sentence: str) -> Dict:
        res = {
            "sentence": sentence,
            "tokens": [],
            "pos": [],
            "features": []
        }
        for word in self.mecab(sentence):
            res["tokens"].append(word.surface)
            res["pos"].append(self.get_pos(word.feature))
            res["features"].append(word.feature)
        return res


class Predictor:
    """長単位品詞・境界と文節境界のための推論器

    Args:
        model_dir (str): 学習済みモデルのあるフォルダへのパス
    """

    def __init__(self, model_dir: str) -> None:
        self.model_dir = model_dir
        
        with open(os.path.join(model_dir, "config.json")) as f:
            self.config = json.load(f)

        self.config["dataeset_options"]["label_file"] = os.path.join(model_dir, "labels.json")

        posfile = os.path.join(model_dir, "pos.json")
        if os.path.exists(posfile):
            self.config["dataeset_options"]["pos_file"] = posfile

        self.model = LUWParserModel.by_name(self.config["model_name"]).from_config(self.config["model_config"], **self.config["dataeset_options"])
        self.model.load_state_dict(torch.load(self.find_best_pt(model_dir)), strict=False)
        self.model.eval()

        self.decoder = Decoder.by_name(self.config["model_config"]["decoder"])()

        self.dataeset_options = self.config['dataeset_options']

        with open(self.dataeset_options["label_file"]) as f:
            self.label_dic = json.load(f)

        self.inv_label_dic = {v:k for k, v in self.label_dic.items()}


    @staticmethod
    def find_best_pt(model_dir: str) -> str:
        """ベストデータポイントを探す

        Args:
            model_dir (str): 学習済みモデルのあるフォルダへのパス

        Returns:
            str: ベストモデルポイントへのパス
        """
        candidates = list()
        longest = -1
        idx = -1
        for i, fname in enumerate(glob.glob(os.path.join(model_dir, "best_at_*.pt"))):
            candidates.append(fname)
            base = os.path.basename(fname)
            epoch = base.split("_")[2]
            epoch = int(epoch.split(".")[0])
            if longest < epoch:
                longest = epoch
                idx = i

        return candidates[idx]
    
    def extract_labels(self, word_ids, labels):
        """テキストラベルからIDに変換

        Args:
            word_ids (List[int]): None出ない場合は、サブワードに対応する短単位語の文頭からの番号
            labels (List[str]): テキストラベル列 

        Returns:
            List[int]: テキストラベルに対応するIDの列
        """
        res = list()
        if word_ids is None:
            return [self.inv_label_dic.get(l, "unk") for l in labels]
        prv = -1
        for wid, l in zip(word_ids, labels):
            if wid is not None and wid >= 0:
                if wid == prv:
                    continue
                res.append(self.inv_label_dic.get(l, "unk"))
                prv = wid
        return res

    def predict(self, input: List[str], suw_tokenizer: str, suw_tokenizer_option: dict, encoder_name: str, batch_size: int = 8, device: str="cpu", **kwargs):
        """モデル推論の実行

        Args:
            input (List[str]): 
                文のリスト(複数文を処理する)
            suw_tokenizer (str): 
                短単位語解析をするトークナイザ :py:class:`~monaka.tokenizer.Tokenizer` のサブクラス名。Registrableで付けた名前を利用する。
                
                例: :py:class:`~monaka.tokenizer.AutoLMTakenizer` の場合は、"auto"
            suw_tokenizer_option (dict):
                トークナイザの初期化に必要な引数。用いるTokenizerの引数を参照して設定。
            encoder_name (str): 
                出力先のフォーマットを指定する :py:class:`~monaka.predictor.Encoder` のサブクラス名。Registrableで付けた名前を利用する。
            batch_size (int, optional): 
                バッチ数. Defaults to 8.
            device (str, optional): 
                用いるGPU番号か"cpu"を指定. Defaults to "cpu".

        Yields:
            Any: 指定した :py:class:`~monaka.predictor.Encoder` が出力する結果
        """
        encoder = Encoder.by_name(encoder_name)(**kwargs)
        tokenizer = SUWTokenizer.by_name(suw_tokenizer)(**suw_tokenizer_option)

        data = [tokenizer.tokenize(sent) for sent in input]

        self.dataeset_options['store_all'] = True
        dataset = LUWJsonLDataset(data, **self.dataeset_options)
        dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=LUWJsonLDataset.collate_function)


        init_device(device)
        try:
            device = int(device)
        except:
            pass
        self.model.to(device)

        with torch.no_grad():
            for data in dataloader:
                #word_ids = [sbw.word_ids() for sbw in data["subwords"]]
                subwords = pad_sequence(data["input_ids"], batch_first=True, padding_value=dataset.pad_token_id).to(device)
                word_ids = pad_sequence([torch.LongTensor(js.word_ids()) for js in data["subwords"]], batch_first=True, padding_value=-1).to(device)
                pos_ids = pad_sequence(data["pos_ids"], batch_first=True, padding_value=1).to(device) if "pos_ids" in data else None

                out = self.model(subwords, word_ids, pos_ids)
                pred = torch.argmax(out, dim=-1) # batch, len, 
                tops = torch.topk(out, len(self.label_dic), dim=-1)

                pred_np = pred.detach().cpu().numpy()
                tops_np = tops.indices.detach().cpu().numpy()
                prv_tokens = None
                prv_pos = None
                prv_labels = None
                for prd, wids, sentence, tokens, pos, meta, fold, top in zip(pred_np, word_ids, data["sentence"], data["tokens"], data["pos"], data.get("features", {}), data["fold"], tops_np):
                    if not dataset.label_for_all_subwords:
                        labels = self.extract_labels(None, prd)
                    else:
                        labels = self.extract_labels(wids, prd)
                    if fold < 0:
                        res = self.decoder.decode(tokens, pos, labels)
                        res = self.apply_single_suw_rule(res, top)
                        res["sentence"] = sentence
                        res["features"] = meta
                        res["meta"] = meta
                        out = encoder.encode(**res)
                        yield out
                    elif fold == 0:
                        logger.warning(f"fold: 0 {''.join(tokens)}")
                        prv_tokens = tokens
                        prv_pos = pos
                        prv_labels = labels
                    else: #fold == 1
                        logger.warning(f"fold: 1 {''.join(tokens)}")
                        prv_tokens.extend(tokens)
                        prv_pos.extend(pos)
                        prv_labels.extend(labels)
                        logger.warning(f"unfolding {''.join(prv_tokens)}")
                        res = self.decoder.decode(prv_tokens, prv_pos, prv_labels)
                        res = self.apply_single_suw_rule(res, top)
                        res["sentence"] = sentence
                        res["features"] = meta
                        res["meta"] = meta
                        out = encoder.encode(**res)
                        yield out


    def predict_raw(self, input: str, encoder_name: str, batch_size: int = 8, device: str="cpu"):
        """モデル推論の実行（短単位解析なし）

        Args:
            input (str): 
                :py:class:`~monaka.dataset.LUWJsonLDataset` が読み取れるJSONテキスト
            encoder_name (str): 
                出力先のフォーマットを指定する :py:class:`~monaka.predictor.Encoder` のサブクラス名。Registrableで付けた名前を利用する。
            batch_size (int, optional): 
                バッチ数. Defaults to 8.
            device (str, optional): 
                用いるGPU番号か"cpu"を指定. Defaults to "cpu".

        Yields:
            Any: 指定した :py:class:`~monaka.predictor.Encoder` が出力する結果
        """
        encoder = Encoder.by_name(encoder_name)()

        dataset = LUWJsonLDataset(input, **self.dataeset_options)
        dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=LUWJsonLDataset.collate_function)


        init_device(device)
        try:
            device = int(device)
        except:
            pass
        self.model.to(device)


        with torch.no_grad():
            for data in dataloader:
                #word_ids = [sbw.word_ids() for sbw in data["subwords"]]
                subwords = pad_sequence(data["input_ids"], batch_first=True, padding_value=dataset.pad_token_id).to(device)
                word_ids = pad_sequence([torch.LongTensor(js.word_ids()) for js in data["subwords"]], batch_first=True, padding_value=-1).to(device)
                pos_ids = pad_sequence(data["pos_ids"], batch_first=True, padding_value=1).to(device) if "pos_ids" in data else None

                out = self.model(subwords, word_ids, pos_ids)
                pred = torch.argmax(out, dim=-1) # batch, len, 
                tops = torch.topk(out, len(self.label_dic), dim=-1)

                pred_np = pred.detach().cpu().numpy()
                tops_np = tops.indices.detach().cpu().numpy()
                prv_tokens = None
                prv_pos = None
                prv_labels = None
                for prd, wids, sentence, tokens, pos, meta, fold, top in zip(pred_np, word_ids, data["sentence"], data["tokens"], data["pos"], data.get("features", {}), data["fold"], tops_np):
                    if not dataset.label_for_all_subwords:
                        labels = self.extract_labels(None, prd)
                    else:
                        labels = self.extract_labels(wids, prd)
                    if fold < 0:
                        res = self.decoder.decode(tokens, pos, labels)
                        res = self.apply_single_suw_rule(res, top)
                        res["sentence"] = sentence
                        res["features"] = meta
                        res["meta"] = meta
                        out = encoder.encode(**res)
                        yield out
                    elif fold == 0:
                        logger.warning(f"fold: 0 {''.join(tokens)}")
                        prv_tokens = tokens
                        prv_pos = pos
                        prv_labels = labels
                    else: #fold == 1
                        logger.warning(f"fold: 1 {''.join(tokens)}")
                        prv_tokens.extend(tokens)
                        prv_pos.extend(pos)
                        prv_labels.extend(labels)
                        logger.warning(f"unfolding {''.join(prv_tokens)}")
                        res = self.decoder.decode(prv_tokens, prv_pos, prv_labels)
                        res = self.apply_single_suw_rule(res, top)
                        res["sentence"] = sentence
                        res["features"] = meta
                        res["meta"] = meta
                        out = encoder.encode(**res)
                        yield out

    def apply_single_suw_rule(self, decoder_out, top):
        """簡易のルールベース処理を行う

        Args:
            decoder_out (Dict): :py:class:`~monaka.predictor.Decoder` が出力するデータ 学習モデルによる違いを吸収して統一したフォーマットにするための処理。
            top (List[List[int]]): 推論時の尤度順に並べた推論結果

        Returns:
            Dict: ルール処理後のdecoder_out
        """
        singles = list(range(len(decoder_out["luw"])))
        for i, l in enumerate(decoder_out["luw"]):
            if '*' in l:
                singles[i] = -1
                if i > 0:
                    singles[i-1] = -1

        for s, pos, luw, t in zip(singles, decoder_out['pos'], decoder_out['luw'], top):
            if s < 0:
                continue
            if '可能' not in pos:
                if luw in pos: ## MeCabの品詞に活用型が含まれるので、長単位品詞がMeCabと一致していればそれを使う。
                    decoder_out['luw'][s] = luw
                else: # 品詞が異なる場合は、やむをえずMeCabを使う。(副作用ありなので、要相談)
                    decoder_out['luw'][s] = pos

            elif '名詞-普通名詞-助数詞可能' in pos:
                decoder_out['luw'][s] = '名詞-普通名詞-一般'
            elif '動詞-非自立可能' in pos:
                decoder_out['luw'][s] = '動詞-一般'
            elif '形容詞-非自立可能' in pos:
                decoder_out['luw'][s] = '形容詞-一般'
            elif '名詞-普通名詞-サ変可能' in pos:
                decoder_out['luw'][s] = '名詞-普通名詞-一般'
            elif '名詞-普通名詞-形状詞可能' in pos or '名詞-普通名詞-サ変形状詞可能' in pos:
                for i in t:
                    label = self.inv_label_dic[i]
                    if '形状詞-一般' in label:
                        decoder_out['luw'][s] = '形状詞-一般'
                        break
                    elif '名詞-普通名詞-一般' in label:
                        decoder_out['luw'][s] = '名詞-普通名詞-一般'
                        break
            elif '名詞-普通名詞-副詞可能' in pos :
                for i in t:
                    label = self.inv_label_dic[i]
                    if '副詞' in label:
                        decoder_out['luw'][s] = '副詞'
                        break
                    elif '名詞-普通名詞-一般' in label:
                        decoder_out['luw'][s] = '名詞-普通名詞-一般'
                        break
        return decoder_out

    
    def evaluate(self, inputfile: str, batch_size: int = 8, device: str="cpu", targets: List[str]=("luw", "chunk"), pos_level: int = -1, format_: str="pretty",  outputfile: str=None):
        """推論し、評価を実行

        Args:
            inputfile (str): 
                評価対象のファイルへのパス
            batch_size (int, optional): 
                バッチ数. Defaults to 8.
            device (str, optional): 
                用いるGPU番号か"cpu"を指定. Defaults to "cpu".
            targets (List[str], optional): 
                評価対象の項目 "luw"は長単位、"chunk"は文節. Defaults to ("luw", "chunk").
            pos_level (int, optional): 
                品詞の評価の際の評価対象の品詞階層. Defaults to -1.
            format_ (str, optional): 
                評価結果の出力フォーマット "pretty"か"json". Defaults to "pretty".
            outputfile (str, optional): 
                評価結果の出力先のパス. Defaults to None.
        """
        dataset = LUWJsonLDataset(inputfile, **self.dataeset_options)
        dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=LUWJsonLDataset.collate_function)
        if outputfile is not None:
            output = open(outputfile, "w")

        init_device(device)
        try:
            device = int(device)
        except:
            pass
        self.model.to(device)

        reporters = {name: MetricReporter(name) for name in targets}
        span_reporters = {f"{name}_span": SpanBasedMetricReporter(f"{name}_span") for name in targets}
        span_reporters["luw_triples"] = SpanBasedMetricReporter("luw_triples")


        for data in dataloader:
            subwords = pad_sequence(data["input_ids"], batch_first=True, padding_value=dataset.pad_token_id).to(device)
            word_ids = pad_sequence([torch.LongTensor(js.word_ids()) for js in data["subwords"]], batch_first=True, padding_value=-1).to(device)
            pos_ids = pad_sequence(data["pos_ids"], batch_first=True, padding_value=1).to(device) if "pos_ids" in data else None

            out = self.model(subwords, word_ids, pos_ids)
            pred = torch.argmax(out, dim=-1) # batch, len, 

            pred_np = pred.detach().cpu().numpy()
            for prd, wids, sentence, tokens, pos, gold in zip(pred_np, word_ids, data["sentence"], data["tokens"], data["pos"], data["labels"]):
                if not dataset.label_for_all_subwords:
                    labels = self.extract_labels(None, prd)
                else:
                    labels = self.extract_labels(wids, prd)
                
                res = self.decoder.decode(tokens, pos, labels, pos_level)
                res = append_spans(res)
                gres = self.decoder.decode(tokens, pos, gold, pos_level)
                gres = append_spans(gres)

                if outputfile is not None:
                    print(json.dumps(res, ensure_ascii=False), file=output)

                if np.random.random() < 0.02:
                    print("gold", gres, file=sys.stderr)
                    print("pred", res, file=sys.stderr)
                for target in targets:
                    rep = reporters[target]
                    rep.update(gres[target], res[target])

                    starget = f"{target}_span"
                    rep = span_reporters[starget]
                    rep.update(gres[starget], res[starget])
                span_reporters["luw_triples"].update(gres["luw_triples"], res["luw_triples"])

        if "pretty" in format_:
            for rep in reporters.values():
                rep.pretty()
            
            for rep in span_reporters.values():
                rep.pretty()
        else:
            res = dict()
            for k, rep in reporters.items():
                res[k] = rep.to_json()

            for k, rep in span_reporters.items():
                res[k] = rep.to_json()

            print(json.dumps(res, indent=True))

        if outputfile is not None:
            output.close()


class EnsemblePredictor:
    """平均アンサンブルを用いた長単位品詞・境界と文節境界のための推論器

    Args:
        model_dirs (List[str]): 学習済みモデルのあるフォルダへのパス(複数可。複数の場合はアンサンブルされる)
        device (str, optional): 
            用いるGPU番号か"cpu"を指定. Defaults to "cpu".
    """

    def __init__(self, model_dirs: List[str], device: str="cpu") -> None:
        self.model_dirs = model_dirs
        
        with open(os.path.join(model_dirs[0], "config.json")) as f:
            self.config = json.load(f)

        self.config["dataeset_options"]["label_file"] = os.path.join(model_dirs[0], "labels.json")

        posfile = os.path.join(model_dirs[0], "pos.json")
        if os.path.exists(posfile):
            self.config["dataeset_options"]["pos_file"] = posfile

        self.models = list()
        for model_dir in model_dirs:
            model = LUWParserModel.by_name(self.config["model_name"]).from_config(self.config["model_config"], **self.config["dataeset_options"])
            model.load_state_dict(torch.load(self.find_best_pt(model_dir)), strict=False)
            model.eval()
            self.models.append(model)

        self.decoder = Decoder.by_name(self.config["model_config"]["decoder"])()

        self.dataeset_options = self.config['dataeset_options']
        self.dataeset_options["fold_sentence"] = True

        with open(self.dataeset_options["label_file"]) as f:
            self.label_dic = json.load(f)

        self.inv_label_dic = {v:k for k, v in self.label_dic.items()}

        init_device(device)
        try:
            device = int(device)
        except:
            pass
        for model in self.models:
            model.to(device)
        self.device = device


    @staticmethod
    def find_best_pt(model_dir: str) -> str:
        """ベストデータポイントを探す

        Args:
            model_dir (str): 学習済みモデルのあるフォルダへのパス

        Returns:
            str: ベストモデルポイントへのパス
        """
        candidates = list()
        longest = -1
        idx = -1
        for i, fname in enumerate(glob.glob(os.path.join(model_dir, "best_at_*.pt"))):
            candidates.append(fname)
            base = os.path.basename(fname)
            epoch = base.split("_")[2]
            epoch = int(epoch.split(".")[0])
            if longest < epoch:
                longest = epoch
                idx = i

        return candidates[idx]
    
    def extract_labels(self, word_ids, labels):
        """テキストラベルからIDに変換

        Args:
            word_ids (List[int]): None出ない場合は、サブワードに対応する短単位語の文頭からの番号
            labels (List[str]): テキストラベル列 

        Returns:
            List[int]: テキストラベルに対応するIDの列
        """
        res = list()
        if word_ids is None:
            return [self.inv_label_dic.get(l, "unk") for l in labels]
        prv = -1
        for wid, l in zip(word_ids, labels):
            if wid is not None and wid >= 0:
                if wid == prv:
                    continue
                res.append(self.inv_label_dic.get(l, "unk"))
                prv = wid
        return res

    def predict(self, input: List[str], suw_tokenizer: str, suw_tokenizer_option: dict, encoder_name: str, batch_size: int = 8, **kwargs):
        """モデル推論の実行
        
        Args:
            input (List[str]): 
                文のリスト(複数文を処理する)
            suw_tokenizer (str): 
                短単位語解析をするトークナイザ :py:class:`~monaka.tokenizer.Tokenizer` のサブクラス名。Registrableで付けた名前を利用する。
                
                例: :py:class:`~monaka.tokenizer.AutoLMTokenizer` の場合は、"auto"
            suw_tokenizer_option (dict):
                トークナイザの初期化に必要な引数。用いるTokenizerの引数を参照して設定。
            encoder_name (str): 
                出力先のフォーマットを指定する :py:class:`~monaka.predictor.Encoder` のサブクラス名。Registrableで付けた名前を利用する。
            batch_size (int, optional): 
                バッチ数. Defaults to 8.

        Yields:
            Any: 指定した :py:class:`~monaka.predictor.Encoder` が出力する結果
        """
        encoder = Encoder.by_name(encoder_name)(**kwargs)
        tokenizer = SUWTokenizer.by_name(suw_tokenizer)(**suw_tokenizer_option)

        data = [tokenizer.tokenize(sent) for sent in input]

        self.dataeset_options['store_all'] = True
        dataset = LUWJsonLDataset(data, **self.dataeset_options)
        dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=LUWJsonLDataset.collate_function)

        with torch.no_grad():
            for data in dataloader:
                #word_ids = [sbw.word_ids() for sbw in data["subwords"]]
                subwords = pad_sequence(data["input_ids"], batch_first=True, padding_value=dataset.pad_token_id).to(self.device)
                word_ids = pad_sequence([torch.LongTensor(js.word_ids()) for js in data["subwords"]], batch_first=True, padding_value=-1).to(self.device)
                pos_ids = pad_sequence(data["pos_ids"], batch_first=True, padding_value=1).to(self.device) if "pos_ids" in data else None

                # average ensemble
                out = 0.
                for model in self.models:
                    out = out + model(subwords, word_ids, pos_ids)

                pred = torch.argmax(out, dim=-1) # batch, len, 
                tops = torch.topk(out, len(self.label_dic), dim=-1)

                pred_np = pred.detach().cpu().numpy()
                tops_np = tops.indices.detach().cpu().numpy()
                prv_tokens = None
                prv_pos = None
                prv_labels = None
                for prd, wids, sentence, tokens, pos, meta, fold, top in zip(pred_np, word_ids, data["sentence"], data["tokens"], data["pos"], data.get("features", {}), data["fold"], tops_np):
                    if not dataset.label_for_all_subwords:
                        labels = self.extract_labels(None, prd)
                    else:
                        labels = self.extract_labels(wids, prd)
                    if fold < 0:
                        res = self.decoder.decode(tokens, pos, labels)
                        res = self.apply_single_suw_rule(res, top)
                        res["sentence"] = sentence
                        res["features"] = meta
                        res["meta"] = meta
                        out = encoder.encode(**res)
                        yield out
                    elif fold == 0:
                        logger.warning(f"fold: 0 {''.join(tokens)}")
                        prv_tokens = tokens
                        prv_pos = pos
                        prv_labels = labels
                    else: #fold == 1
                        logger.warning(f"fold: 1 {''.join(tokens)}")
                        prv_tokens.extend(tokens)
                        prv_pos.extend(pos)
                        prv_labels.extend(labels)
                        logger.warning(f"unfolding {''.join(prv_tokens)}")
                        res = self.decoder.decode(prv_tokens, prv_pos, prv_labels)
                        res = self.apply_single_suw_rule(res, top)
                        res["sentence"] = sentence
                        res["features"] = meta
                        res["meta"] = meta
                        out = encoder.encode(**res)
                        yield out

    def apply_single_suw_rule(self, decoder_out, top):
        """簡易のルールベース処理を行う

        Args:
            decoder_out (Dict): :py:class:`~monaka.predictor.Decoder` が出力するデータ 学習モデルによる違いを吸収して統一したフォーマットにするための処理。
            top (List[List[int]]): 推論時の尤度順に並べた推論結果

        Returns:
            Dict: ルール処理後のdecoder_out
        """
        singles = list(range(len(decoder_out["luw"])))
        for i, l in enumerate(decoder_out["luw"]):
            if '*' in l:
                singles[i] = -1
                if i > 0:
                    singles[i-1] = -1

        for s, pos, luw, t in zip(singles, decoder_out['pos'], decoder_out['luw'], top):
            if s < 0:
                continue
            if '可能' not in pos:
                if luw in pos: ## MeCabの品詞に活用型が含まれるので、長単位品詞がMeCabと一致していればそれを使う。
                    decoder_out['luw'][s] = luw
                else: # 品詞が異なる場合は、やむをえずMeCabを使う。(副作用ありなので、要相談)
                    decoder_out['luw'][s] = pos

            elif '名詞-普通名詞-助数詞可能' in pos:
                decoder_out['luw'][s] = '名詞-普通名詞-一般'
            elif '動詞-非自立可能' in pos:
                decoder_out['luw'][s] = '動詞-一般'
            elif '形容詞-非自立可能' in pos:
                decoder_out['luw'][s] = '形容詞-一般'
            elif '名詞-普通名詞-サ変可能' in pos:
                decoder_out['luw'][s] = '名詞-普通名詞-一般'
            elif '名詞-普通名詞-形状詞可能' in pos or '名詞-普通名詞-サ変形状詞可能' in pos:
                for i in t:
                    label = self.inv_label_dic[i]
                    if '形状詞-一般' in label:
                        decoder_out['luw'][s] = '形状詞-一般'
                        break
                    elif '名詞-普通名詞-一般' in label:
                        decoder_out['luw'][s] = '名詞-普通名詞-一般'
                        break
            elif '名詞-普通名詞-副詞可能' in pos :
                for i in t:
                    label = self.inv_label_dic[i]
                    if '副詞' in label:
                        decoder_out['luw'][s] = '副詞'
                        break
                    elif '名詞-普通名詞-一般' in label:
                        decoder_out['luw'][s] = '名詞-普通名詞-一般'
                        break
        return decoder_out
        

    def predict_raw(self, input, encoder_name: str, batch_size: int = 8):
        """モデル推論の実行（短単位解析なし）

        Args:
            input (str): 
                :py:class:`~monaka.dataset.LUWJsonLDataset` が読み取れるJSONテキスト
            encoder_name (str): 
                出力先のフォーマットを指定する :py:class:`~monaka.predictor.Encoder` のサブクラス名。Registrableで付けた名前を利用する。
            batch_size (int, optional): 
                バッチ数. Defaults to 8.

        Yields:
            Any: 指定した :py:class:`~monaka.predictor.Encoder` が出力する結果
        """
        encoder = Encoder.by_name(encoder_name)()

        self.dataeset_options['store_all'] = True
        dataset = LUWJsonLDataset(input, **self.dataeset_options)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=LUWJsonLDataset.collate_function)


        with torch.no_grad():
            for data in dataloader:
                #word_ids = [sbw.word_ids() for sbw in data["subwords"]]
                subwords = pad_sequence(data["input_ids"], batch_first=True, padding_value=dataset.pad_token_id).to(self.device)
                word_ids = pad_sequence([torch.LongTensor(js.word_ids()) for js in data["subwords"]], batch_first=True, padding_value=-1).to(self.device)
                pos_ids = pad_sequence(data["pos_ids"], batch_first=True, padding_value=1).to(self.device) if "pos_ids" in data else None

                # average ensemble
                out = 0.
                for model in self.models:
                    out = out + model(subwords, word_ids, pos_ids)

                pred = torch.argmax(out, dim=-1) # batch, len, 
                tops = torch.topk(out, len(self.label_dic), dim=-1)

                pred_np = pred.detach().cpu().numpy()
                tops_np = tops.indices.detach().cpu().numpy()
                prv_tokens = None
                prv_pos = None
                prv_labels = None
                for prd, wids, sentence, tokens, pos, meta, fold, top in zip(pred_np, word_ids, data["sentence"], data["tokens"], data["pos"], data.get("meta", data["pos"]), data["fold"], tops_np):
                    if not dataset.label_for_all_subwords:
                        labels = self.extract_labels(None, prd)
                    else:
                        labels = self.extract_labels(wids, prd)
                    #logger.warning(labels)
                    if fold < 0:
                        res = self.decoder.decode(tokens, pos, labels)
                        res = self.apply_single_suw_rule(res, top)
                        res["sentence"] = sentence
                        res["features"] = meta
                        res["meta"] = meta
                        out = encoder.encode(**res)
                        yield out
                    elif fold == 0:
                        logger.warning(f"fold: 0 {''.join(tokens)}")
                        prv_tokens = tokens
                        prv_pos = pos
                        prv_labels = labels
                    else: #fold == 1
                        logger.warning(f"fold: 1 {''.join(tokens)}")
                        prv_tokens.extend(tokens)
                        prv_pos.extend(pos)
                        prv_labels.extend(labels)
                        logger.warning(f"unfolding {''.join(prv_tokens)}")
                        res = self.decoder.decode(prv_tokens, prv_pos, prv_labels)
                        res = self.apply_single_suw_rule(res, top)
                        res["sentence"] = sentence
                        res["meta"] = meta
                        out = encoder.encode(**res)
                        yield out


from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoConfig, Seq2SeqTrainer, Seq2SeqTrainingArguments
from torch.utils.data import DataLoader

class LemmaPredictor:
    """語彙素推定のモデル推論を行う

    Args:
        model_dirs (List[str]): 学習済みモデルのあるフォルダへのパス(複数可。複数の場合はアンサンブルされる)
        device (str, optional): 
            用いるGPU番号か"cpu"を指定. Defaults to "cpu".
    """

    def __init__(self, model_dir: str, device='cpu') -> None:
        self.model_dir = model_dir
        self.device = device
        with open(os.path.join(model_dir, "config.json")) as f:
            self.config = json.load(f)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(os.path.join(model_dir, "last-checkpoint")).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['model_name'])
        self.special_tokens = list(self.tokenizer.all_special_tokens)
        self.special_tokens.extend([" ", "▁", "_"])
        self.trainer = Seq2SeqTrainer(self.model, Seq2SeqTrainingArguments(".", per_device_eval_batch_size=8, predict_with_generate=True))

    def predict(self, input: Dict, use_pos: bool=False) -> str:
        """語彙素推定の実行

        Args:
            input (Dict): :py:class:`~monaka.dataset.LemmaJsonDataset` で読み取り可能な辞書型のJSONデータ
            use_pos (bool, optional): 
                特定の品詞は短単位語のものをそのまま使うか. Defaults to False.

        Returns:
            str: 推論された語彙素
        """
        dataset = LemmaJsonDataset(input, **self.config['dataset_options'])
        #print(dataset[0])
        #print(dataset[0]['input_ids'].size())
        #preds = self.trainer.predict(test_dataset=dataset)
        #preds_ = [np.argmax(prd, axis=-1) for prd in preds]
        #decoded_preds = self.tokenizer.batch_decode(preds_[0], skip_special_tokens=True)
        #return decoded_preds[0]

        #print(input)
        data = dataset[0]
        #print(data['input_ids'].size())
        inp = self.tokenizer.decode(data['input_ids'])
        #print(inp)
        if '<unk>' in inp:
            return ''.join([v['lemma'] for v in data.get('suw', [{'lemma': ''}])])
        
        if use_pos:
            pos:str = data.get('pos', '')
            if pos.startswith(('接続詞', '感動詞', '副詞')):
                # 特定の品詞の場合はSUWを直接使う
                return ''.join([v['lemma'] for v in data.get('suw', [{'lemma': ''}])])
            
        outputs = self.model.generate(data['input_ids'].unsqueeze(0).to(self.device), 
                                      attention_mask=data['attention_mask'].to(self.device),
                                      do_sample=False)
        #print(outputs)
        #print(self.tokenizer.decode(outputs[0], skip_special_tokens=False))
        tokens = self.tokenizer.convert_ids_to_tokens(outputs[0], skip_special_tokens=False)
        output = []
        for t in tokens:
            if t not in self.special_tokens:
                output.append(t)
            else:
                if len(output) == 0:
                    continue
                else:
                    break
        o ="".join(output)
        #print(o.replace("▁", ""))
        return o.replace("▁", "")
    
    def tokens2output(self, tokens: List[str]) -> str:
        """トークン出力を整理してテキストで返す

        Args:
            tokens (List[str]): 推論されたトークン列

        Returns:
            str: 復元された語彙素
        """
        output = []
        for t in tokens:
            if t not in self.special_tokens:
                output.append(t)
            else:
                if len(output) == 0:
                    continue
                else:
                    break
        o ="".join(output)
        #print(o.replace("▁", ""))
        return o.replace("▁", "")
    
    def batch_predict(self, inputs: Dict) -> List[str]:
        """推論のバッチ実行

        Args:
            input (Dict): :py:class:`~monaka.dataset.LemmaJsonDataset` で読み取り可能な辞書型のJSONデータ

        Returns:
            List[str]: 推論された語彙素の列
        """
        dataset = LemmaJsonDataset(input, **self.config['dataset_options'])
        loader = DataLoader(dataset=dataset, batch_size=self.config['batch_size'], shuffle=False)
        #print(data['input_ids'].size())
        res = list()
        for data in loader:
            #print(self.tokenizer.decode(data['input_ids']))
            outputs = self.model.generate(data['input_ids'].to(self.device), 
                                        attention_mask=data['attention_mask'].to(self.device),
                                        do_sample=False)
            #print(outputs)
            #print(self.tokenizer.decode(outputs[0], skip_special_tokens=False))
            for output in outputs:
                res.append(self.tokens2output(self.tokenizer.convert_ids_to_tokens(output, skip_special_tokens=False)))
    
    def evaluate(self, jsonfile: str) -> Dict:
        """評価の実行

        Args:
            jsonfile (str): 評価対象のファイルへのパス

        Returns:
            Dict: 評価結果の辞書型データ count: 評価数 acc: 正解率 diff: 正解との差分がある結果
        """
        dataset = LemmaJsonDataset(jsonfile, **self.config['dataset_options'])
        loader = DataLoader(dataset, batch_size=1, shuffle=False)

        c = 0
        a = 0
        diff = list()
        for feat in loader:
            a += 1
            pred = self.predict(feat['input'][0])
            label = feat['target'][0]
            if pred == label:
                c += 1
            else:
                diff.append(f"input: {feat['input']}, correct: {label}, pred: {pred}")
        return {
            "count": a,
            "acc": c/a,
            "diff": diff
        }


class DepPredictor:
    """係り受け解析の推論の実行

    Args:
        model_dir (str): 学習済みモデルのあるフォルダへのパス
    """

    def __init__(self, model_dir: str) -> None:

        self.model_dir = model_dir
        
        with open(os.path.join(model_dir, "config.json")) as f:
            self.config = json.load(f)

        self.config["dataeset_options"]["label_file"] = os.path.join(model_dir, "labels.json")
        self.config["dataeset_options"]["rel_file"] = os.path.join(model_dir, "rels.json")
        self.config["dataeset_options"]["wlsp_file"] = os.path.join(model_dir, "wlsp_dic.json")

        posfile = os.path.join(model_dir, "pos.json")
        if os.path.exists(posfile):
            self.config["dataeset_options"]["pos_file"] = posfile

        self.model = LUWParserModel.by_name(self.config["model_name"]).from_config(self.config["model_config"], **self.config["dataeset_options"])
        self.model.load_state_dict(torch.load(os.path.join(model_dir, "best.pt")), strict=False)
        self.model.eval()

        #self.decoder = Decoder.by_name(self.config["model_config"]["decoder"])()

        self.dataeset_options = self.config['dataeset_options']

        with open(self.dataeset_options["label_file"]) as f:
            self.label_dic = json.load(f)

        self.inv_label_dic = {v:k for k, v in self.label_dic.items()}

        with open(self.dataeset_options["rel_file"]) as f:
            self.rel_dic = json.load(f)
    
        self.inv_rel_dic = {v:k for k, v in self.rel_dic.items()}

    def predict(self, input: List[str], dep_decoder_name: str, encoder_name: str, batch_size: int = 8, device: str="cpu", left2right: bool=False, **kwargs):
        """モデル推論の実行

        Args:
            input (List[str]): 
                :py:class:`~monaka.dataset.LUWJsonLDataset` が読み取れるJSONテキスト
            dep_decoder_name (str): 
                入力元のフォーマットを指定する :py:class:`~monaka.predictor.DepDecoder` のサブクラス名。Registrableで付けた名前を利用する。
            encoder_name (str): 
                出力先のフォーマットを指定する :py:class:`~monaka.predictor.DepEncoder` のサブクラス名。Registrableで付けた名前を利用する。
            batch_size (int, optional): 
                バッチ数. Defaults to 8.
            device (str, optional): 
                用いるGPU番号か"cpu"を指定. Defaults to "cpu".
            left2right (bool, optional): 
                推論する係り受けを文の先頭から文の終わり方向に限定する. Defaults to False.

        Yields:
            Any: 指定した :py:class:`~monaka.predictor.DepEncoder` が出力する結果
        """
        encoder = DepEncoder.by_name(encoder_name)(**kwargs)
        decoder = DepDecoder.by_name(dep_decoder_name)(**kwargs)

        data = [decoder(sent) for sent in input]

        self.dataeset_options['store_all'] = True
        dataset = ChunkDepJsonLDataset(data, **self.dataeset_options)
        dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=ChunkDepJsonLDataset.collate_function)


        init_device(device)
        try:
            device = int(device)
        except:
            pass
        self.model.to(device)

        self.model.eval()

        #print(self.inv_label_dic)
        with torch.no_grad():
            for data in dataloader:
                    subwords = pad_sequence(data["input_ids"], batch_first=True, padding_value=dataset.pad_token_id).to(device) if 'input_ids' in data else None
                    if 'input_ids' in data:
                        word_ids = pad_sequence([torch.LongTensor(js.word_ids()) for js in data["subwords"]], batch_first=True, padding_value=-1).to(device)
                    else:
                        word_ids = pad_sequence([torch.LongTensor([i for i in range(len(tokens))]) for tokens in data['tokens']]  , batch_first=True, padding_value=-1).to(device)
                    #word_ids = pad_sequence([torch.LongTensor(js.word_ids()) for js in data["subwords"]], batch_first=True, padding_value=-1).to(device)
                    chunk_ids = pad_sequence(data["chunk_ids"], batch_first=True, padding_value=-1).to(device)
                    #dep_ids = pad_sequence(data["dep_ids"], batch_first=True, padding_value=-1).to(device)
                    #word_rel_ids = pad_sequence(data["word_rel_ids"], batch_first=True, padding_value=1).to(device)
                    #dep_rel_ids = pad_sequence(data["dep_rel_ids"], batch_first=True, padding_value=1).to(device)
                    pos_ids = pad_sequence(data["pos_ids"], batch_first=True, padding_value=1).to(device) if "pos_ids" in data else None
                    wlsp_ids = pad_sequence(data["wlsp_ids"], batch_first=True, padding_value=1).to(device) if "wlsp_ids" in data else None
                    #wmask = word_rel_ids.ne(1)
                    #dmask = dep_ids.ne(-1)
                    #rmask = dep_rel_ids.ne(1)

                    dep_out, deprel_out, word_out  = self.model(subwords, word_ids, chunk_ids, pos_ids, wlsp_ids)
                    #deprel_out = deprel_out.permute((0,2,3,1)) # [batch, chunk_class, chunk_len, chunk_len] -> [batch, chunk_len, chunk_len, chunk_class]
                    #dep_pred = torch.argmax(dep_out, dim=-1)
                    softmax = torch.nn.functional.softmax
                    deprel_out[:, self.rel_dic['root'], :, :] = -1000.
                    rel_pred = torch.argmax(deprel_out, dim=1) #[batch, chunk_len, chunk_len]
                    shead_ids = [i for k, i in self.label_dic.items() if 'shead' in k]
                    wrd_shead_np = softmax(word_out[:, :, shead_ids], dim=-1).detach().cpu().numpy()
                    word_out2 = word_out.detach()
                    word_out2[:, :, shead_ids] = -1000.
                    wrd_pred = torch.argmax(word_out2, dim=-1)

                    dep_pred_np = softmax(dep_out, dim=-1).detach().cpu().numpy()
                    rel_pred_np = rel_pred.detach().cpu().numpy()
                    wrd_pred_np = wrd_pred.detach().cpu().numpy()
                    nulls = ['_' for _ in data['tokens']]

                    for depp, relp, wrdp, shp, bnst, pos, tokens, bid, sentid, text, upos, misc, lemma, undc in zip(dep_pred_np, rel_pred_np, wrd_pred_np, wrd_shead_np, 
                            data['bunsetsu'], data['pos'], data['tokens'], data['bid'], data['sent_id'], data['text'], 
                            data.get('upos', nulls), data.get('misc', nulls), data.get('lemma', nulls), data.get('unidic', nulls)):
                        res = {
                            "sent_id": sentid,
                            "text": text,
                            "tokens": tokens,
                            "bid": bid,
                            "bunsetsu": bnst,
                            "pos": pos,
                            "upos": upos,
                            "misc": misc,
                            "lemma": lemma,
                            "unidic": undc
                        }
                        roots = np.array([depp[i,i] for i in range(len(bnst))])
                        ndepp = np.hstack((roots.reshape(len(bnst), 1), depp[:len(bnst), :len(bnst)]))
                        ndepp = np.vstack((np.hstack(([0], roots)).reshape(1, len(bnst)+1), ndepp))
                        #for i in range(ndepp.shape[0]):
                        #    ndepp[i, i] = -1000.
                        heads, _ = chu_liu_edmonds(ndepp)
                        #print(heads, len(heads), len(bnst))
                        #print(heads.index(0))
                        root = heads.index(0) -1
                        if left2right:
                            x = list()
                            y = list()
                            for i in range(len(bnst)):
                                for j in range(0, root):
                                    x.append(j)
                                    y.append(i)
                                for j in range(root+1, len(bnst)):
                                    x.append(i)
                                    y.append(j)
                            ndepp[(x, y)] = -1000.
                            heads, _ = chu_liu_edmonds(ndepp)
                        #roots = np.array([depp[i,i] if np.argmax(depp[i, :len(bnst)]) == i else -1000. for i in range(len(bnst))])
                        #root = np.argmax(roots)
                        #for i in range(depp.shape[0]):
                        #    depp[i, i] = -1000.
                        #dep = np.argmax(depp, axis=-1)
                        res['rel'] = [self.inv_label_dic.get(v, 'nmod') for v,_ in zip(wrdp, tokens)]
                        res['dependency'] = [{"id": i, "head": int(u) -1 if u > 0 else root, "rel": self.inv_rel_dic.get(v[u-1], 'nmod')} for i, (u,v,_) in enumerate(zip(heads[1:], relp, bnst))]
                        for d in res['dependency']:
                            if d['id'] == d['head']:
                                d['head'] = root
                        #res['dependency'][root]['head'] = res['dependency'][root]['id']
                        res['dependency'][root]['rel'] = "root"
                        bid = np.array(bid)
                        indices = np.arange(len(bid))
                        #for i in range(np.max(bid)+1):
                        #    ind = np.where(bid == i)
                            #print(ind)
                        #    j = np.argmax(shp[ind])
                        #    k = indices[ind][j]
                        #    res['rel'][k] = 'shead'
                        yield encoder(res)