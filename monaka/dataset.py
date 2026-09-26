# -*- coding: utf-8 -*-
"""dataset.py

学習・推論に用いるデータセット(PyTorchで扱える形式)に変換するクラス群
"""
import copy
import json
from pathlib import Path
from collections import namedtuple

import numpy as np
import torch
import torch.distributed as dist
from typing import Union, List, Dict, Optional, Any
from monaka.tokenizer import Tokenizer
from monaka.mylogging import logger
from transformers import  AutoTokenizer

class ChunkDepJsonLDataset(torch.utils.data.Dataset):
    """文節係り受け学習・推論用のデータセット

    JsonL形式のデータセット 各行は以下:
        {
            "sent_id": str      # 文ID

            "text": str,        # 文そのもの

            "bunsetsu": [str, ] # 文節区切りのテキスト

            "bid": [int, ]      # 短単位語の文節ID

            "pos": [str ]       # 形態論情報(短単位)

            "tokens": [str, ]   # 短単位のリスト

            "rel": [str, ]      # 単語単位の係り受けラベル

            "dependency": [{id, head, rel}] # 文節係り受け情報
        }

    Attributes:
        tokenizer (~monaka.tokenizer.Tokenizer): 言語モデル固有のトークナイザ
        pad_token_id (int): [PAD]トークンのID番号
        max_length (int): 最大文長
        chunk_max_length (int): 最大文節数
        store_all (bool): 全データを保持するかのフラグ
        wlsp_file (str): 分類語彙表データのファイル
        jsonlfiles: 読み込むJSON-Lファイルのリスト(JSON-Lを対象としない場合でも内部的にJSON-Lと同等のデータ形式で保持)
        logger (logging.Logger): ログ出力用のロガー
        
    Args:
        jsonlfiles (Union[str, List[str]]): 読み込むJSON-Lファイルのリスト(JSON-Lを対象としない場合でも内部的にJSON-Lと同等のデータ形式で保持)
        label_file (str): 
            全ラベルを記載したjsonファイル label: id 形式。未知ラベルは常に unk: 0 train_cli.py create-vocab で作成する
        pos_file (str): 
            短単位品詞IDを記載したJSONファイル label: id 形式。未知ラベルは常に unk: 0 pos_as_tokensの時は利用されない。 train_cli.py create-vocab で作成する。
        rel_file (str): 係り受け関係ラベルとそのIDが記載されたJSONファイル train_cli.py create-vocab で作成する
        lm_tokenizer (str): 
            適切なTransoformesのTokenizerをラップしたTokenizer名
        lm_tokenizer_config (Dict): 言語モデルのトークナイザに渡す設定データ
        wlsp_file (str, optional): 分類語彙表データへのパス. Defaults to None.
        max_length (int, optional): 最大文長. Defaults to 1024.
        chunk_max_length (int, optional): 最大文節数. Defaults to 128.
        logger (~logging.Logger, optional): ログ出力用のロガー. Defaults to :py:const:`~monaka.mylogging.logger`.
        store_all (bool, optional): 全ての情報を保持するかのフラグ. Defaults to False (保持しない).
    """

    def __init__(self, jsonlfiles: Union[str, List[str]], label_file: str, pos_file: str, rel_file: str, lm_tokenizer: str, lm_tokenizer_config: Dict, wlsp_file:str=None, max_length: int=1024,  chunk_max_length: int=128, logger=logger, store_all: bool=False, 
                 **kwargs):
        """文節係り受け学習用のデータセットのinit

        Args:
            jsonlfiles (Union[str, List[str]]): 読み込むJSON-Lファイルのリスト(JSON-Lを対象としない場合でも内部的にJSON-Lと同等のデータ形式で保持)
            label_file (str): 
                全ラベルを記載したjsonファイル label: id 形式。未知ラベルは常に unk: 0 train_cli.py create-vocab で作成する
            pos_file (str): 
                短単位品詞IDを記載したJSONファイル label: id 形式。未知ラベルは常に unk: 0 pos_as_tokensの時は利用されない。 train_cli.py create-vocab で作成する。
            rel_file (str): 係り受け関係ラベルとそのIDが記載されたJSONファイル train_cli.py create-vocab で作成する
            lm_tokenizer (str): 
                適切なTransoformesのTokenizerをラップしたTokenizer名
            lm_tokenizer_config (Dict): 言語モデルのトークナイザに渡す設定データ
            wlsp_file (str, optional): 分類語彙表データへのパス. Defaults to None.
            max_length (int, optional): 最大文長. Defaults to 1024.
            chunk_max_length (int, optional): 最大文節数. Defaults to 128.
            logger (~logging.Logger, optional): ログ出力用のロガー. Defaults to :py:const:`~monaka.mylogging.logger`.
            store_all (bool, optional): 全ての情報を保持するかのフラグ. Defaults to False (保持しない).
        """
        self.sentences = list()
        if lm_tokenizer is not None:
            self.tokenizer = Tokenizer.by_name(lm_tokenizer)(**lm_tokenizer_config)
            self.pad_token_id = self.tokenizer.pad_token_id
        else:
            self.tokenizer = None
            self.pad_token_id = 1

        self.max_length = max_length
        self.chunk_max_length = chunk_max_length
        self.jsonlfiles = jsonlfiles
        self.logger = logger
        self.store_all = store_all
        self.wlsp_file = wlsp_file

        with open(label_file) as f:
            self.label_dic = json.load(f)

        with open(rel_file) as f:
            self.rel_dic = json.load(f)

        if pos_file is not None :
            with open(pos_file) as f:
                self.pos_dic = json.load(f)
        else:
            self.pos_dic = None

        if wlsp_file is not None:
            with open(wlsp_file) as f:
                self.wlsp_dic = json.load(f)
        else:
            self.wlsp_dic = None
        
        if isinstance(jsonlfiles, str):
            self.logger.info(f"loading {jsonlfiles}")
            self.load(jsonlfiles)
        elif isinstance(jsonlfiles, list):
            for fname in jsonlfiles:
                if isinstance(fname, str):
                    self.logger.info(f"loading {fname}")
                    self.load(fname)
                elif isinstance(fname, dict):
                    self.load_dict(fname)
        
        self.logger.info(f"total {len(self.sentences)} sentences loaded.")
        super().__init__()

    @staticmethod
    def collate_function(data: List[Dict]):
        """Batchのデータを作成する関数

        Args:
            data (List[Dict]): Batch内の各データ点のデータ

        Returns:
            Dict: DictのキーごとにBatch内のデータをリストにしたもの
        """
        #targets = ["input_ids", "label_ids", "pos_ids"]
        res = dict()
        #for target in targets:
        #    if target not in data[0]:
        #        continue
        #    res[target] = [d[target] for d in data]
        for k in data[0].keys():
            res[k] = [d.get(k) for d in data]
        return res

    def load(self, jsonlfile: str):
        """JSON-Lファイルの読み込み

        Args:
            jsonlfile (str): ファイルへのパス
        """
        with open(jsonlfile) as f:
            for line in f:
                js = json.loads(line)
                self.load_dict(js)

    def load_dict(self, js: dict):
        """JSON-Lの1行に対する読み込み

        Args:
            js (dict): JSON-L1行に相当するデータ
        """
        js['skip'] = False

        if "rel" in js and len(js["rel"]) != len(js["tokens"]):
            self.logger.warning(f'skip loading {js["sentence"]} because of pos {len(js["pos"])} and token {len(js["tokens"])} length unmatch')
            if self.store_all:
                js['skip'] = True
                self.sentences.append(js)
            return
        
        if len(js["tokens"]) == 0:
            self.logger.warning(f'skip loading {js["sentence"]} because there is no token.')
            if self.store_all:
                js['skip'] = True
                self.sentences.append(js)
            return

        if self.tokenizer:
            js["subwords"] = self.to_token_ids(js["tokens"])
            js["input_ids"] = torch.LongTensor(js["subwords"]["input_ids"])

        js["word_rel_ids"] = self.to_label_ids(js["rel"]) if "rel" in js else None

        js['chunk_ids'] = torch.LongTensor(js['bid'])
        js["dep_ids"] = torch.LongTensor([d['head'] for d in js['dependency']]) if "dependency" in js else None
        js["dep_rel_ids"] = self.to_rel_ids([d['head'] for d in js['dependency']], [d['rel'] for d in js['dependency']]) if "dependency" in js else None

        if self.pos_dic:
            js["pos_ids"] = self.to_pos_ids(js["pos"]) 

        if self.wlsp_dic:
            wids = [self.pos_dic.get(k, 0) for k in js["wlsp"]]
            if len(wids) > self.max_length:
                wids = wids[:self.max_length]
            js["wlsp_ids"] = torch.LongTensor(wids)

        if self.tokenizer is not None and len(js["subwords"].word_ids()) == 0:
            self.logger.warning(f"no words: {js['tokens']}")
            if self.store_all:
                js['skip'] = True
                self.sentences.append(js)
            return

        if self.tokenizer is not None and len(js["pos"]) != np.max(js["subwords"].word_ids()) + 1:
            self.logger.warning(f'unmatch length {len(js["pos"])} {np.max(js["subwords"].word_ids()) + 1}, {js["tokens"]} {js["subwords"]} {js["subwords"].word_ids()}')
        self.sentences.append(js)


    def to_label_ids(self, labels: List[str], word_ids: Optional[List[int]]=None) -> torch.LongTensor:
        """トークンや短単位語の単位でラベルに対するPytorch Tensorを出力する

        Args:
            labels (List[str]): ラベル系列
            word_ids (Optional[List[int]], optional): Noneの場合は短単位ごと、word_idsが与えられるとサブワード単位でラベル付与. Defaults to None.

        Returns:
            torch.LongTensor: ラベルに対応する pytorch Tensor
        """
        labels_ = [self.label_dic.get(k, 0) for k in labels]
        if word_ids is not None:
            prv = -1
            labels = list()
            for idx in word_ids:
                if idx != prv:
                    labels.append(labels_[idx])
                    prv = idx
                else:
                    labels.append(1)
        else:
            labels = labels_
        if len(labels) > self.max_length:
            labels = labels[:self.max_length]
        return torch.LongTensor(labels)
    

    def to_rel_ids(self, heads: List[int], rels: List[str]) -> torch.LongTensor:
        """関係ラベルに対するPytorch Tensorを出力する

        Args:
            heads (List[int]): 係り先の短単位語の位置
            rels (List[str]): 対応する関係ラベル

        Returns:
            torch.LongTensor: ラベルに対応する pytorch Tensor
        """
        rels_ = [self.rel_dic.get(k, 0) for k in rels]
        outs = torch.LongTensor([[self.pad_token_id for _ in range(self.chunk_max_length)] for _ in range(self.chunk_max_length)])
        for i, (hid, r) in enumerate(zip(heads, rels_)):
            if i >= self.chunk_max_length or hid > self.chunk_max_length:
                continue
            outs[i, hid] = r
        return outs
    

    def to_pos_ids(self, labels: List[str], word_ids: Optional[List[int]]=None) -> torch.LongTensor:
        """品詞を表すpytorch tensorに変換する

        Args:
            labels (List[str]): ラベル系列
            word_ids (Optional[List[int]], optional): Noneの場合は短単位ごと、word_idsが与えられるとサブワード単位でラベル付与. Defaults to None.

        Returns:
            torch.LongTensor: 品詞に対応する pytorch Tensor
        """
        labels_ = [self.pos_dic.get(k, 0) for k in labels]
        if word_ids is not None:
            prv = -1
            labels = list()
            for idx in word_ids:
                if idx != prv:
                    labels.append(labels_[idx])
                    prv = idx
                else:
                    labels.append(1) # padding index = 1
        else:
            labels = labels_
        if len(labels) > self.max_length:
            labels = labels[:self.max_length]
        return torch.LongTensor(labels)
    
    def to_token_ids(self, tokens: List[str]) -> Any:
        """言語モデルのトークナイザによって入力トークンを処理する

        Args:
            tokens (List[str]): トークン列

        Returns:
            Any: トークナイザの出力そのまま
        """
        targets = [self.replace_token(t) for t in tokens]

        return self.tokenizer.tokenize(targets, max_length=self.max_length)

    @staticmethod
    def replace_token(token: str) -> str:
        """うまくtokenizeできない句点記号を置換する

        Args:
            token (str): トークン

        Returns:
            str: 問題のある記号は処理してほかはそのまま
        """
        if token in ["．", "："]:
            return "。"
        if token in ["，", "；"]:
            return "、"
        if token in ["？"]:
            return "?"
        if token in ["！"]:
            return "!"
        if token in ["（"]:
            return "("
        if token in ["）"]:
            return ")"
        return token

    def __repr__(self):
        s = f"{self.__class__.__name__}("
        s += f"n_sentences={len(self.sentences)}"

        return s

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        return self.sentences[index]


class LUWJsonLDataset(torch.utils.data.Dataset):
    r"""長単位解析用のデータセット

    JsonL形式のデータセット 各行は以下:
        {
            "sentence": str,   # 文そのもの

            "tokens": [str, ]  # 短単位のリスト

            "pos": [str, ]     # 形態論情報(短単位)

            "lemma": [str,]    # 語彙素原形(必須フィールドではない)

            "labels": [str, ]  # ラベル。短単位ごとに付与
        }

    Args:
        jsonlfiles (str or list[str]):
            読み込むJSON-Lファイルのリスト(JSON-Lを対象としない場合でも内部的にJSON-Lと同等のデータ形式で保持)
        label_file (str):
            全ラベルを記載したjsonファイル label: id 形式。未知ラベルは常に unk: 0 train_cli.py create-vocab で作成する
        pos_file (str):
            短単位品詞IDを記載したJSONファイル label: id 形式。未知ラベルは常に unk: 0 pos_as_tokensの時は利用されない。 train_cli.py create-vocab で作成する。
        lm_tokenizer (str):
            適切なTransoformesのTokenizerをラップしたTokenizer名
        lm_tokenizer_config (dict):
            言語モデルのトークナイザに渡す設定データ
        max_length (int):
            データの最大長
        pos_as_tokens (bool):
            形態論情報を短単位の後に付与するかどうか default=False
        label_for_all_subwords (bool):
            サブワード単位でラベル付けをする。default False
        kwargs (dict):
            Keyword arguments that will be passed into :meth:`transform.load` together with `data`
            to control the loading behaviour.

    Attributes:
        sentences (list[Dict]): 読み込んだデータを保持したもの
        tokenizer (~monaka.tokenizer.Tokenizer): 言語モデル固有のトークナイザ
        pad_token_id (int): [PAD]トークンのID番号
        pos_as_tokens (bool): 品詞情報を言語モデルに直接入力するかどうか 
        max_length (int): 最大文長
        fold_sentence (bool): 推論時に最大長(max_length)を超える文を「折り曲げて」格納して、複数回の推論で一文を処理するかどうか (Falseにすると先頭からmax_lengthまでを処理)
        store_all (bool): 全データを保持するかのフラグ
        wlsp_file (str): 分類語彙表データのファイル
        jsonlfiles: 読み込むJSON-Lファイルのリスト(JSON-Lを対象としない場合でも内部的にJSON-Lと同等のデータ形式で保持)
        logger (logging.Logger): ログ出力用のロガー

    Args:
        jsonlfiles (Union[str, List[str]]): 
            読み込み対象のファイル名（のリスト)
        label_file (str): _description_
            全ラベルを記載したjsonファイル label: id 形式。未知ラベルは常に unk: 0
        pos_file (str): 
            短単位品詞IDを記載したJSONファイル label: id 形式。未知ラベルは常に unk: 0 pos_as_tokensの時は利用されない。 train_cli.py create-vocab で作成する。
        lm_tokenizer (str): 
            適切なTransoformesのTokenizerをラップしたTokenizer名
        lm_tokenizer_config (Dict): 
            言語モデルのトークナイザに渡す設定データ
        max_length (int, optional): 
            言語モデルのトークナイザに渡す設定データ. Defaults to 1024.
        pos_as_tokens (bool, optional): 
            形態論情報を短単位の後に付与するかどうか. Defaults to False.
        label_for_all_subwords (bool, optional): 
            サブワード単位でラベル付けをする。. Defaults to False.
        logger (logging.Logger, optional): 
            ログ出力用のロガー. Defaults to :py:const:`~monaka.mylogging.logger`
        store_all (bool, optional): 
            全ての情報を保持するかのフラグ. Defaults to False (保持しない).
        fold_sentence (bool, optional): 
            推論時に最大長(max_length)を超える文を「折り曲げて」格納して、複数回の推論で一文を処理するかどうか (Falseにすると先頭からmax_lengthまでを処理). Defaults to False.
    """

    def __init__(self, jsonlfiles: Union[str, List[str]], label_file: str, pos_file: str, lm_tokenizer: str, lm_tokenizer_config: Dict, max_length: int=1024, pos_as_tokens: bool=False, 
                 label_for_all_subwords: bool=False, logger=logger, store_all: bool=False, fold_sentence:bool=False,
                 **kwargs):
        """長単位解析用のデータセットのinit

        Args:
            jsonlfiles (Union[str, List[str]]): 
                読み込み対象のファイル名（のリスト)
            label_file (str): _description_
                全ラベルを記載したjsonファイル label: id 形式。未知ラベルは常に unk: 0
            pos_file (str): 
                短単位品詞IDを記載したJSONファイル label: id 形式。未知ラベルは常に unk: 0 pos_as_tokensの時は利用されない。 train_cli.py create-vocab で作成する。
            lm_tokenizer (str): 
                適切なTransoformesのTokenizerをラップしたTokenizer名
            lm_tokenizer_config (Dict): 
                言語モデルのトークナイザに渡す設定データ
            max_length (int, optional): 
                言語モデルのトークナイザに渡す設定データ. Defaults to 1024.
            pos_as_tokens (bool, optional): 
                形態論情報を短単位の後に付与するかどうか. Defaults to False.
            label_for_all_subwords (bool, optional): 
                サブワード単位でラベル付けをする。. Defaults to False.
            logger (logging.Logger, optional): 
                ログ出力用のロガー. Defaults to :py:const:`~monaka.mylogging.logger`
            store_all (bool, optional): 
                全ての情報を保持するかのフラグ. Defaults to False (保持しない).
            fold_sentence (bool, optional): 
                推論時に最大長(max_length)を超える文を「折り曲げて」格納して、複数回の推論で一文を処理するかどうか (Falseにすると先頭からmax_lengthまでを処理). Defaults to False.
        """
        self.sentences = list()
        self.tokenizer = Tokenizer.by_name(lm_tokenizer)(**lm_tokenizer_config)
        self.pad_token_id = self.tokenizer.pad_token_id
        self.pos_as_tokens = pos_as_tokens
        self.label_for_all_subwords = label_for_all_subwords
        self.max_length = max_length
        self.jsonlfiles = jsonlfiles
        self.logger = logger
        self.store_all = store_all
        self.fold_sentence = fold_sentence
        self.logger.warning(f"fold sentence {fold_sentence}")

        with open(label_file) as f:
            self.label_dic = json.load(f)

        if pos_file is not None and not pos_as_tokens:
            with open(pos_file) as f:
                self.pos_dic = json.load(f)
        else:
            self.pos_dic = None
        
        if isinstance(jsonlfiles, (str, Path)):
            self.logger.info(f"loading {jsonlfiles}")
            self.load(jsonlfiles)
        elif isinstance(jsonlfiles, list):
            for fname in jsonlfiles:
                if isinstance(fname, str):
                    self.logger.info(f"loading {fname}")
                    self.load(fname)
                elif isinstance(fname, dict):
                    #self.logger.warning(fname)
                    self.load_dict(fname)
        
        self.logger.warning(f"total {len(self.sentences)} sentences loaded.")
        super().__init__()

    @staticmethod
    def collate_function(data: List[Dict]):
        """Batchのデータを作成する関数

        Args:
            data (List[Dict]): Batch内の各データ点のデータ

        Returns:
            Dict: DictのキーごとにBatch内のデータをリストにしたもの
        """
        #targets = ["input_ids", "label_ids", "pos_ids"]
        res = dict()
        #for target in targets:
        #    if target not in data[0]:
        #        continue
        #    res[target] = [d[target] for d in data]
        for k in data[0].keys():
            res[k] = [d.get(k) for d in data]
        return res

    def load(self, jsonlfile: str):
        """JSON-Lファイルの読み込み

        Args:
            jsonlfile (str): ファイルへのパス
        """
        with open(jsonlfile) as f:
            for line in f:
                js = json.loads(line)
                self.load_dict(js)

    def load_dict(self, js: dict):
        """JSON-Lの1行に対する読み込み

        Args:
            js (dict): JSON-L1行に相当するデータ
        """
        js['skip'] = False
        prv_fold = False
        if "fold" not in js:
            js["fold"] = -1
        else:
            prv_fold = True

        if len(js["pos"]) != len(js["tokens"]):
            self.logger.warning(f'skip loading {js["sentence"]} because of pos {len(js["pos"])} and token {len(js["tokens"])} length unmatch')
            if self.store_all:
                js['skip'] = True
                self.sentences.append(js)
            return
        if len(js["pos"]) == 0:
            self.logger.warning(f'skip loading {js["sentence"]} because there is no token.')
            if self.store_all:
                js['skip'] = True
                self.sentences.append(js)
            return

        js["subwords"] = self.to_token_ids(js["tokens"], js["pos"] if self.pos_as_tokens else None)
        wids = js["subwords"].word_ids()
        w_len = np.max(wids) if len(wids) > 0 else 0
        
        if len(js["pos"]) > w_len + 1 and  len(js["subwords"]["input_ids"]) >= self.max_length and self.fold_sentence: # folding too long sentence
            self.logger.warning(f"folding {len(js['pos'])} , {js['sentence']}")

            if prv_fold:
                logger.error("dual fold")
                
            indices = np.where(np.char.find(js["pos"], "補助記号-読点") > -1)[0]
            if len(indices) == 0:
                indices = np.where(np.char.find(js["pos"], "補助記号-一般") > -1)[0]
            if len(indices) > 0:
                i = indices[int(np.floor(len(indices) / 2))]
                js1 = copy.deepcopy(js)
                js1["tokens"] = js["tokens"][:i+1]
                js1["pos"] = js["pos"][:i+1]
                if "lemma" in js:
                    js1["lemma"] = js["lemma"][:i+1]

                js1["fold"] = 0
                self.load_dict(js1)


                js2 = copy.deepcopy(js)
                js2["tokens"] = js["tokens"][i+1:]
                js2["pos"] = js["pos"][i+1:]
                if "lemma" in js:
                    js2["lemma"] = js["lemma"][i+1:]

                js2["fold"] = 1
                self.load_dict(js2)
                return

        js["input_ids"] = torch.LongTensor(js["subwords"]["input_ids"])
        js["label_ids"] = self.to_label_ids(js["labels"], js["subwords"].word_ids() if self.label_for_all_subwords else None) if "labels" in js else None

        if 'input_ids' not in js:
            return

            #for l_subs, lid in zip(js["lemma_subwords"].word_ids(), js["lemma_id"]):
            #    js["lemma_target"].extend([lid for _ in range(len(l_subs))])
        
            #print(len(js["lemma_target"] ), len(js["input_ids"] ))

        if self.pos_dic:
            js["pos_ids"] = self.to_pos_ids(js["pos"], js["subwords"].word_ids() if self.label_for_all_subwords else None) 


        if len(js["subwords"].word_ids()) == 0:
            self.logger.warning(f"no words: {js['tokens']}")
            if self.store_all:
                js['skip'] = True
                self.sentences.append(js)
            return

        if len(js["pos"]) != np.max(js["subwords"].word_ids()) + 1:
            self.logger.warning(f'unmatch length {len(js["pos"])} {np.max(js["subwords"].word_ids()) + 1}, {js["tokens"]} {js["subwords"]} {js["subwords"].word_ids()}')
        
        if "lemma" in js:
            js["lemma_subwords"] = self.to_token_ids(js["lemma"])
            js["lemma_ids"] = []
            #torch.LongTensor(js["lemma_subwords"]["input_ids"]) # lemma size list of subword list
            sub_idx = js["lemma_subwords"]["input_ids"]
            w_idx = js["subwords"].word_ids()
            l_idx = js["lemma_subwords"].word_ids()
            if len(l_idx) == 0:
                self.logger.warn(f"no lemma: {js['lemma']}")
                self.sentences.append(js)
                return

            for i in range(max(l_idx) + 1):
                js["lemma_ids"].append(torch.LongTensor([sid for sid, wid in zip(sub_idx, l_idx) if wid == i]))
            
            js["lemma_target"] = [js["lemma_id"][i] for i in w_idx]
        
            #print(len(js["lemma_target"] ), len(js["input_ids"] ))

        if len(js["pos"]) != np.max(js["subwords"].word_ids()) + 1:
            self.logger.warning(f'unmatch length {len(js["pos"])} {np.max(js["subwords"].word_ids()) + 1}, {js["tokens"]} {js["subwords"]} {js["subwords"].word_ids()}')
        self.sentences.append(js)

    def to_label_ids(self, labels: List[str], word_ids: Optional[List[int]]=None) -> torch.LongTensor:
        """トークンや短単位語の単位でラベルに対するPytorch Tensorを出力する

        Args:
            labels (List[str]): ラベル系列
            word_ids (Optional[List[int]], optional): Noneの場合は短単位ごと、word_idsが与えられるとサブワード単位でラベル付与. Defaults to None.

        Returns:
            torch.LongTensor: ラベルに対応する pytorch Tensor
        """
        labels_ = [self.label_dic.get(k, 0) for k in labels]
        if word_ids is not None:
            prv = -1
            labels = list()
            for idx in word_ids:
                if idx != prv:
                    labels.append(labels_[idx])
                    prv = idx
                else:
                    labels.append(1)
        else:
            labels = labels_
        if len(labels) > self.max_length:
            labels = labels[:self.max_length]
        return torch.LongTensor(labels)
    

    def to_pos_ids(self, labels: List[str], word_ids: Optional[List[int]]=None) -> torch.LongTensor:
        """品詞を表すpytorch tensorに変換する

        Args:
            labels (List[str]): ラベル系列
            word_ids (Optional[List[int]], optional): Noneの場合は短単位ごと、word_idsが与えられるとサブワード単位でラベル付与. Defaults to None.

        Returns:
            torch.LongTensor: 品詞に対応する pytorch Tensor
        """
        labels_ = [self.pos_dic.get(k, 0) for k in labels]
        if word_ids is not None:
            prv = -1
            labels = list()
            for idx in word_ids:
                if idx != prv:
                    labels.append(labels_[idx])
                    prv = idx
                else:
                    labels.append(1) # padding index = 1
        else:
            labels = labels_
        if len(labels) > self.max_length:
            labels = labels[:self.max_length]
        return torch.LongTensor(labels)
    
    def to_token_ids(self, tokens: List[str], pos: Optional[List[str]]=None) -> Any:
        """言語モデルのトークナイザによって入力トークンを処理する

        Args:
            tokens (List[str]): トークン列

        Returns:
            Any: トークナイザの出力そのまま
        """
        if pos is not None:
            assert(len(tokens) == len(pos))
            targets = [f"{self.replace_token(t)} {p}" for t, p in zip(tokens, pos)]
        else:
            targets = [self.replace_token(t) for t in tokens]

        return self.tokenizer.tokenize(targets, max_length=self.max_length)

    @staticmethod
    def replace_token(token: str) -> str:
        """うまくtokenizeできない句点記号を置換する

        Args:
            token (str): トークン

        Returns:
            str: 問題のある記号は処理してほかはそのまま
        """
        if token in ["．", "："]:
            return "。"
        if token in ["，", "；"]:
            return "、"
        if token in ["？"]:
            return "?"
        if token in ["！"]:
            return "!"
        if token in ["（"]:
            return "("
        if token in ["）"]:
            return ")"
        return token

    def __repr__(self):
        s = f"{self.__class__.__name__}("
        s += f"n_sentences={len(self.sentences)}"

        return s

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        return self.sentences[index]


class LemmaJsonDataset(torch.utils.data.Dataset):
    r"""長単位の語彙素推定用のデータセット

    Json形式のデータセット: 
        {
            "key (全情報)": {

                "surface": "やまとうた", # 表層形

                "lemma": "やまと歌",  #  長単位原形

                "pos": "名詞-普通名詞-一般", # 品詞

                "suw": [

                {
                    "pos": "名詞-固有名詞-地名-一般", # 短単位品詞

                    "lemma": "ヤマト", # 短単位原形

                    "surface": "やまと"# 短単位原形

                },

                {
                    "pos": "名詞-普通名詞-一般", # 短単位品詞

                    "lemma": "歌", # 短単位原形

                    "surface": "うた" # 短単位原形
                }

                ],

                "input": "pos: 名詞-普通名詞-一般 surface: やまとうた suw_pos: 名詞-固有名詞-地名-一般 名詞-普通名詞-一般 suw_surface: やまと うた suw_lemma: ヤマト 歌" #入力形式

            }

        }

    Args:
        jsonfiles (Union[str, List[Union[str, Dict]]]): 
            読み込むJSONファイル。
        transformer_name (str): 
            言語モデル名(HuggingFace Transformersにおける名前)
        max_length (int, optional): 
            最大トークン長. Defaults to 512.
        fields (Optional[List[str]], optional): 
            語彙素推定に用いる情報を表すfield。読み込むJSONにおけるkeyになっている. Defaults to None.
        logger (logging.Logger): 
            ログ出力用のロガー. Defaults to :py:const:`~monaka.mylogging.logger`.

    Attributes:
    """

    def __init__(self, jsonfiles: Union[str, List[Union[str, Dict]]], transformer_name: str, max_length: int=512, fields: Optional[List[str]]=None, logger=logger,
                 **kwargs):
        """長単位の語彙素推定用のデータセットのinit

        Args:
            jsonfiles (Union[str, List[Union[str, Dict]]]): 
                読み込むJSONファイル。
            transformer_name (str): 
                言語モデル名(HuggingFace Transformersにおける名前)
            max_length (int, optional): 
                最大トークン長. Defaults to 512.
            fields (Optional[List[str]], optional): 
                語彙素推定に用いる情報を表すfield。読み込むJSONにおけるkeyになっている. Defaults to None.
            logger (logging.Logger): 
                ログ出力用のロガー. Defaults to :py:const:`~monaka.mylogging.logger`.
        """
        self.lemma = list()
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(transformer_name)
        if isinstance(jsonfiles, str):
            self.load_data(jsonfiles, fields)
        else:
            for jsfile in jsonfiles:
                if isinstance(jsfile, str):
                    self.load_data(jsfile, fields)
                else:
                    self.load_dict(jsfile, fields)

    def __getitem__(self, idx):
        if np.random.random() < 0.1:
            t = f"input: {self.lemma[idx]['input']} target: {self.lemma[idx]['target']}"
            #print(t)
            logger.info(t)
        return self.lemma[idx]
    
    def __len__(self):
        return len(self.lemma)

    @staticmethod
    def to_label(data: Dict, fields: Optional[List[str]]) -> str:
        """使用するfiledから入力情報を生成する

        Args:
            data (Dict): 入力データ
            fields (Optional[List[str]]): 利用するfieldのリスト

        Returns:
            str: 入力情報文字列
        """
        if fields is None:
            return data["input"]
        
        inputs: List[str] = list()
        for field in fields:
            if not field.startswith("suw_"):
                t = f"{field}: {data.get(field, '')}"
                inputs.append(t)
                continue

            starget = field.split("_")[1]
            vals = [suw[starget] for suw in data["suw"]]
            inputs.append(f'{field}: {" ".join(vals)}')
        
        return " ".join(inputs)
    
    def load_data(self, fname: str, fields: Optional[List[str]]):
        """データ読み込み

        Args:
            fname (str): 対象ファイル
            fields (Optional[List[str]]): 利用するfieldのリスト
        """
        with open(fname) as f:
            data: Dict = json.load(f)

        self.load_dict(data, fields)

    def load_dict(self, data: Dict, fields: Optional[List[str]]):
        """辞書データから読み込み

        Args:
            data (Dict): 対象データ
            fields (Optional[List[str]]): 利用するfieldのリスト
        """
        for val in data.values():
            d = {
                "input": self.to_label(val, fields),
                "target": val["lemma"] if 'lemma' in val else None
            }
            d["subwords"] = self.tokenizer(d["input"], max_length=self.max_length, padding='max_length', truncation=True, return_tensors="pt")
            d["input_ids"] =torch.LongTensor(d["subwords"]["input_ids"][0])
            d["attention_mask"] = torch.tensor(d["subwords"]["attention_mask"])
            d["target_subwords"] = self.tokenizer(d["target"], max_length=self.max_length, padding='max_length') if d['target'] is not None else None
            if d['target'] is not None:
                d["labels"] = d["target_subwords"]["input_ids"]
            #d["decoder_input_ids"] = d["target_subwords"]["input_ids"]
            self.lemma.append(d)
        
