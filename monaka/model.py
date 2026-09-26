# -*- coding: utf-8 -*-
"""Monakaの深層学習モデル群

"""

import os
import sys
import json
from random import Random

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.utils.rnn import pad_sequence

from registrable import Registrable
from typing import Dict, Tuple
from monaka.module import MLP, LMEmbedding, Biaffine, SelfAttentionPooling, PositionalEncoding
from monaka.mylogging import logger


class LUWParserModel(nn.Module, Registrable):
    """文節・長単位の境界・品詞推定を行うモデルの基底クラス

    """

    def __init__(self, *args, **kwargs) -> None:
        nn.Module.__init__(self)
        Registrable.__init__(self)

    @classmethod
    def from_config(cls, config: Dict, label_file: str, pos_file: str, rel_file: str=None, wlsp_file:str=None, **kwargs):
        """コンフィグファイルからの生成メソッド

        Args:
            config (Dict):
                コンフィグ
            label_file (str): 
                推定する対象（ラベル）の名前とIDが記載されたJSONファイルへのパス
            pos_file (str):
                短単位品詞の名前とIDが記載されたJSONファイルへのパス
            rel_file (str, optional): 
                係り受け関係の関係の名前とIDが記載されたJSONファイルへのパス. Defaults to None.
            wlsp_file (str, optional): 
                分類語彙表の情報を記載したファイルへのパス. Defaults to None.

        Returns:
            LUWParserModel: 生成されたモデル
        """
        with open(label_file) as f:
            js = json.load(f)
            config["n_class"] = len(js)

        
        with open(pos_file) as f:
            js = json.load(f)
            config["n_pos"] = len(js)

        if rel_file is not None:
            with open(rel_file) as f:
                js = json.load(f)
                config["rel_class"] = len(js)
        
        if wlsp_file is not None:
            with open(wlsp_file) as f:
                js = json.load(f)
                config["n_wlsp"] = len(js)

        return cls(**config)

    def forward(self, words: torch.Tensor, word_ids: torch.Tensor, pos: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError
    
    def loss(self, out, labels, mask, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError


class LUWLemmaModel(nn.Module, Registrable):
    """
    語彙素原形を推定するモデルの基底クラス
    """

    def __init__(self, *args, **kwargs) -> None:
        nn.Module.__init__(self)
        Registrable.__init__(self)

    @classmethod
    def from_config(cls, config: Dict,  **kwargs):
        """コンフィグファイルからの生成メソッド

        Args:
            config (Dict): コンフィグ

        Returns:
            LUWLemmaModel: 生成されたモデル
        """

        return cls(**config)

    def forward(self, words: torch.Tensor, luw_target: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError
    
    def loss(self, out: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError
    

@LUWParserModel.register("SeqTagging")
class SeqTaggingParserModel(LUWParserModel):
    """
    SubwordレベルでSequence Taggingするモデル。:py:class:`~Registrable`で呼び出すときは`SeqTagging`

    Args:
        n_pos (int):
            形態素の種類数。
        n_pos_emb (int):
            形態素埋め込み表現の次元数。
        n_class (int):
            クラスラベル数
        pos_dropout (float):
            pos埋め込みのdropout
        mlp_dropout (float):
            識別用のMLPのdropout
        lm_class_name (str):
            用いるlm class名 TrasformersのAutoConfig, AutoModelなどが上手く使えない場合は専用クラスが用意されている。
        lm_class_config (dict):
            lm_class用のconfig
        pos_padding_idx (int):
            pos埋め込みのpadding idx
    """
    
    def __init__(self,
            n_pos: int,
            n_pos_emb: int,
            n_class: int,
            pos_dropout: float,
            mlp_dropout: float,
            lm_class_name: str,
            lm_class_config: Dict,
            pos_padding_idx: int = 1,
            **kwargs) -> None:
        super().__init__(**kwargs)
        
        logger.info("Model: SeqTagging")

        self.n_pos = n_pos
        self.n_pos_emb = n_pos_emb
        self.n_class = n_class
        self.pos_dropout = pos_dropout
        self.lm_class_name = lm_class_name
        self.lm_class_config = lm_class_config
        self.pos_padding_idx = pos_padding_idx
        
        self.m_lm = LMEmbedding.by_name(lm_class_name)(**lm_class_config)
        self.m_pos_emb = nn.Embedding(n_pos, n_pos_emb, pos_padding_idx) if n_pos > 0 and n_pos_emb > 0 else None
        self.m_pos_dropout = nn.Dropout(pos_dropout) if self.m_pos_emb else None

        self.n_in = self.m_lm.n_out if self.m_pos_emb is None else self.m_lm.n_out + n_pos_emb
        self.m_out = MLP(self.n_in, n_class, mlp_dropout)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, words: torch.Tensor, word_ids: torch.Tensor, pos: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Args:
            words (~torch.Tensor):
                入力文をサブワード分割して埋め込み表現に変換したもの。shape [batch, words_len, emb], words_lenは最大サブワード長と一致
            word_ids (~torch.Tensor):
                他のモデルとの互換性のために指定しているが不使用。
            pos (~torch.Tensor):
                サブワードに対応する短単位品詞のID列 shape [batch, wrds_len]

        Returns:
            ~torch.Tensor: shape [batch, words_len, n_class] n_classはラベルの次元数。尤もらしいクラスに対応する次元の値が一番大きくなるように学習
        """
        #print(words.size())
        words_emb = self.m_lm(words)
        if self.m_pos_emb:
            #print(pos)
            pos_embs = self.m_pos_emb(pos)
            pos_embs = self.m_pos_dropout(pos_embs)
            #print(words_emb.size())
            #print(pos_embs.size())
            words_emb = torch.cat((words_emb, pos_embs), dim=-1) # batch, words_len, hidden

        out = self.m_out(words_emb)

        return out # batch, words_len, n_class
    
    def loss(self, out: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor, *args, **kwargs):
        """損失関数

        Args:
            out (~torch.Tensor):
                モデルの出力 shape [batch, words_len, n_class]
            labels (~torch.Tensor): [batch, words_len, 1]
                正解ラベル
            mask (~torch.Tensor): 
                バッチ中の最大サブワード長を基準としたテンソルとなっているので、推論に関係する次元のみで損失を計算する。maskは文中であれば1、そうでなければ0となっている、[batch, words_len] テンソル。
        """

        out_size = out.size()
        mask = mask[:out_size[0], :out_size[1]]
        labels = labels[:out_size[0], :out_size[1]]
        return self.criterion(out[mask], labels[mask])



@LUWParserModel.register("WordTagging")
class WordTaggingParserModel(LUWParserModel):
    """
    WordレベルでSequence Taggingするモデル :py:class:`~Registrable`で呼び出すときは`WordTagging`

    Args:
        n_pos (int):
            形態素の種類数。
        n_pos_emb (int):
            形態素埋め込み表現の次元数。
        n_class (int):
            クラスラベル数
        pooling (str):
            subword -> wordのpooling方法 (max, sum, attention)
        pos_dropout (float):
            pos埋め込みのdropout
        mlp_dropout (float):
            識別用のMLPのdropout
        lm_class_name (str):
            用いるlm class名 TrasformersのAutoConfig, AutoModelなどが上手く使えない場合は専用クラスが用意されている。
        lm_class_config (dict):
            lm_class用のconfig
        pos_padding_idx (int):
            pos埋め込みのpadding idx
    """
    
    def __init__(self,
            n_pos: int,
            n_pos_emb: int,
            n_class: int,
            pooling: str,
            pos_dropout: float,
            mlp_dropout: float,
            lm_class_name: str,
            lm_class_config: Dict,
            pos_padding_idx: int = 1,
            **kwargs) -> None:
        super().__init__(**kwargs)
        
        logger.info("Model: WordTagging")

        self.n_pos = n_pos
        self.n_pos_emb = n_pos_emb
        self.n_class = n_class
        self.pooling_name = pooling
        self.pos_dropout = pos_dropout
        self.lm_class_name = lm_class_name
        self.lm_class_config = lm_class_config
        self.pos_padding_idx = pos_padding_idx
        
        self.m_lm = LMEmbedding.by_name(lm_class_name)(**lm_class_config)
        self.m_pos_emb = nn.Embedding(n_pos, n_pos_emb, pos_padding_idx) if n_pos > 0 and n_pos_emb > 0 else None
        self.m_pos_dropout = nn.Dropout(pos_dropout) if self.m_pos_emb else None

        if "max" in pooling:
            self.pooling = self.max
        elif "sum" in pooling:
            self.pooling = torch.sum
        elif "attention":
            self.pooling = SelfAttentionPooling(self.m_lm.n_out)
        else:
            self.pooling = None

        self.n_in = self.m_lm.n_out if self.m_pos_emb is None else self.m_lm.n_out + n_pos_emb
        self.m_out = MLP(self.n_in, n_class, mlp_dropout)

        self.criterion = nn.CrossEntropyLoss()

    @staticmethod
    def max(value, **kwargs):
        """~torch.max のラッパー torch.maxはその値を取り出すときに、.valueを参照する必要がある

        Args:
            value (~torch.Tensor): maxをとる対象のテンソル

        Returns:
            ~torch.Tensor: max値
        """
        return torch.max(value, **kwargs).values

    def forward(self, words: torch.Tensor, word_ids: torch.Tensor, pos: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Args:
            words (~torch.Tensor):
                入力文をサブワード分割して埋め込み表現に変換したもの。shape [batch, subwords_len, emb], subwords_lenは最大サブワード長と一致
            word_ids (~torch.Tensor):
                各サブワードに対応する短単位語の文頭からの位置（0始まり）。 shape [batch, subword_len]
            pos (~torch.Tensor):
                サブワードに対応する短単位品詞のID列 shape [batch, words_len] words_lenは最大短単位語数に一致
        
        Returns:
            ~torch.Tensor: shape [batch, words_len, n_class] n_classはラベルの次元数。尤もらしいクラスに対応する次元の値が一番大きくなるように学習
        """
        words_emb = self.m_lm(words)

        we = list()
        if self.m_pos_emb and torch.max(word_ids)+1 != pos.size()[-1]:
            print(torch.max(word_ids, dim=1), file=sys.stderr)
        L = torch.max(word_ids) + 1 if not self.m_pos_emb else pos.size()[-1] # なぜかPOSが多い時がある。調査要

        for i in range(L):
            wi = word_ids.eq(i).unsqueeze(-1)
            mask = torch.cat([wi for _ in range(words_emb.size()[-1])], dim=-1)
            _o = words_emb * mask # batch, num of subwords in a word, hidden (acctually masked zero)
            we.append(self.pooling(_o, dim=1, keepdim=True)) # batch, 1, hidden

        words_emb = torch.cat(we, 1)

        if self.m_pos_emb:
            #print(pos)
            pos_embs = self.m_pos_emb(pos)
            pos_embs = self.m_pos_dropout(pos_embs)
            words_emb = torch.cat((words_emb, pos_embs), dim=-1) # batch, words_len, hidden

        out = self.m_out(words_emb) # batch, len, hidden
        return out # batch, words_len, n_class
    
    def loss(self, out: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor, *args, **kwargs):
        """損失関数

        Args:
            out (~torch.Tensor):
                モデルの出力 shape [batch, words_len, n_class]
            labels (~torch.Tensor): [batch, words_len, 1]
                正解ラベル
            mask (~torch.Tensor): 
                バッチ中の最大サブワード長を基準としたテンソルとなっているので、推論に関係する次元のみで損失を計算する。maskは文中であれば1、そうでなければ0となっている、[batch, words_len] テンソル。
        """
        out_size = out.size()
        mask = mask[:out_size[0], :out_size[1]]
        labels = labels[:out_size[0], :out_size[1]]

        return self.criterion(out[mask], labels[mask])



@LUWParserModel.register("ChunkDep")
class ChunkDependencyParserModel(LUWParserModel):
    """
    文節係り受けモデル  :py:class:`~Registrable`で呼び出すときは`ChunkDep`

    Args:
        n_pos (int):
            形態素の種類数。
        n_pos_emb (int):
            形態素埋め込み表現の次元数。
        n_wlsp (int):
            分類語彙表の種類数。
        n_wlsp_emb (int):
            分類語彙表の分類番号の埋め込み表現の次元数。
        chunk_class (int):
            chunkクラスラベル数
        word_class (int):
            wordクラスラベル数
        encoder_type (str):
            エンコーダの種類 lstm or transformer
        lstm_layers (int);
            LSTM層数 0以下でLSTMを使用しない
        word_pooling (str):
            subword -> wordのpooling方法 (max, sum, attention)
        chunk_pooling (str):
            subword -> chunのpooling方法 (max, sum, attention)
        eps0 (float):
            rel lossの係数
        eps1 (float);
            word lossの係数
        pos_dropout (float):
            pos埋め込みのdropout
        lstm_dropout (float):
            LSTMのdropout
        mlp_dropout (float):
            識別用のMLPのdropout
        lm_class_name (str):
            用いるlm class名 TrasformersのAutoConfig, AutoModelなどが上手く使えない場合は専用クラスが用意されている。
        lm_class_config (dict):
            lm_class用のconfig
        pos_padding_idx (int):
            pos埋め込みのpadding idx
    """
    
    def __init__(self,
            n_pos: int,
            n_pos_emb: int,
            n_wlsp: int,
            n_wlsp_emb: int,
            rel_class: int,
            n_class: int,
            encoder_type: str,
            lstm_layers: int,
            word_pooling: str,
            chunk_pooling: str,
            eps0: float,
            eps1: float,
            pos_dropout: float,
            lstm_dropout: float,
            mlp_dropout: float,
            lm_class_name: str,
            lm_class_config: Dict,
            pos_padding_idx: int = 1,
            wlsp_padding_idx: int = 1,
            **kwargs) -> None:
        super().__init__(**kwargs)
        
        logger.info("Model: ChunkDep")

        self.n_pos = n_pos
        self.n_pos_emb = n_pos_emb
        self.n_wlsp = n_wlsp
        self.n_wlsp_emb = n_wlsp_emb
        self.rel_class = rel_class
        self.n_class = n_class
        self.encoder_type = encoder_type
        self.lstm_layers = lstm_layers
        self.word_pooling_name = word_pooling
        self.chunk_pooling_name = chunk_pooling
        self.eps0 = eps0
        self.eps1 = eps1
        self.pos_dropout = pos_dropout
        self.lstm_dropout = lstm_dropout
        self.lm_class_name = lm_class_name
        self.lm_class_config = lm_class_config
        self.pos_padding_idx = pos_padding_idx
        self.wlsp_padding_idx = wlsp_padding_idx
        
        if lm_class_name is not None:
            self.m_lm = LMEmbedding.by_name(lm_class_name)(**lm_class_config)
        else:
            self.m_lm = None
        self.m_pos_emb = nn.Embedding(n_pos, n_pos_emb, pos_padding_idx) if n_pos > 0 and n_pos_emb > 0 else None
        self.m_wlsp_emb = nn.Embedding(n_wlsp, n_wlsp_emb, wlsp_padding_idx) if n_wlsp > 0 and n_wlsp_emb > 0 else None
        self.m_pos_dropout = nn.Dropout(pos_dropout) if self.m_pos_emb else None
        self.m_wlsp_dropout = nn.Dropout(pos_dropout) if self.m_wlsp_emb else None

        if self.m_lm:
            self.n_in = self.m_lm.n_out if self.m_pos_emb is None else self.m_lm.n_out + n_pos_emb
            self.n_out = self.m_lm.n_out
        else:
            self.n_in = n_pos_emb + n_wlsp_emb
            self.n_out = self.n_in

        if lstm_layers > 0:
            if encoder_type == 'transformer':
                self.m_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(self.n_in, nhead=8, dropout=lstm_dropout, batch_first=True), lstm_layers)
                self.m_position_enc = PositionalEncoding(self.n_in, lstm_dropout)
            else:
                self.m_encoder = nn.LSTM(self.n_in, self.n_in, num_layers=lstm_layers, batch_first=True, dropout=lstm_dropout, bidirectional=True)
                self.n_in = self.n_in * 2
        else:
            self.m_encoder = None

        if "max" in word_pooling:
            self.word_pooling = self.max
        elif "sum" in word_pooling:
            self.word_pooling = torch.sum
        elif "mean" in word_pooling:
            self.word_pooling = torch.mean
        elif "attention":
            self.word_pooling = SelfAttentionPooling(self.n_out)
        else:
            self.word_pooling = None

        if "max" in chunk_pooling:
            self.chunk_pooling = self.max
        elif "sum" in chunk_pooling:
            self.chunk_pooling = torch.sum
        elif "mean" in chunk_pooling:
            self.chunk_pooling = torch.mean
        elif "attention":
            self.chunk_pooling = SelfAttentionPooling(self.n_in)
        else:
            self.chunk_pooling = None

        self.m_word = MLP(self.n_in, n_class, mlp_dropout)

        self.m_head = MLP(self.n_in, self.n_in)
        self.m_dep = MLP(self.n_in, self.n_in)
        self.m_deprel = Biaffine(self.n_in, self.rel_class)
        self.m_depnd = Biaffine(self.n_in, 1)

        self.dep_criterion = nn.CrossEntropyLoss()
        self.rel_criterion = nn.CrossEntropyLoss()
        self.wrd_criterion = nn.CrossEntropyLoss()

    @staticmethod
    def max(value, **kwargs):
        """~torch.max のラッパー torch.maxはその値を取り出すときに、.valueを参照する必要がある

        Args:
            value (~torch.Tensor): maxをとる対象のテンソル

        Returns:
            ~torch.Tensor: max値
        """
        return torch.max(value, **kwargs).values

    def forward(self, words: torch.Tensor, word_ids: torch.Tensor, chunk_ids: torch.Tensor, pos: torch.Tensor, wlsp: torch.Tensor, *args, **kwargs) -> Tuple[torch.Tensor]:
        """
        Args:
            words (~torch.Tensor):
                入力文をサブワード分割して埋め込み表現に変換したもの。shape [batch, subwords_len, emb], subwords_lenは最大サブワード長と一致
            word_ids (~torch.Tensor):
                各サブワードに対応する短単位語の文頭からの位置（0始まり）。 shape [batch, subword_len]
            chunk_ids (~torch.Tensor):
                各短単位語が所属する文節の文頭からの位置（0始まり）。 shape [batch, word_len]
            pos (~torch.Tensor):
                サブワードに対応する短単位品詞のID列 shape [batch, words_len] words_lenは最大短単位語数に一致
        
        Returns:
            Tuple[~torch.Tensor]: 
                dep_out [batch, chunk_len, chunk_len] 係り受け関係が存在するか

                deprel_out [batch, chunk_class, chunk_len, chunk_len] 係り受け関係ラベルの推定

                word_out [batch, words_len, word_class] 短単位語に付与されるラベル（文節内係り受けなど）
        """

        if self.m_lm:
            words_emb = self.m_lm(words)
        else:
            words_emb = None

        we = list()
        ce = list()

        if self.m_pos_emb and self.m_lm and torch.max(word_ids)+1 != pos.size()[-1]:
            print(torch.max(word_ids, dim=1), file=sys.stderr)
        L = torch.max(word_ids) + 1 if not self.m_pos_emb else pos.size()[-1] # なぜかPOSが多い時がある。調査要
        C = torch.max(chunk_ids) + 1

        if self.m_lm:
            for i in range(L):
                wi = word_ids.eq(i).unsqueeze(-1)
                mask = torch.cat([wi for _ in range(words_emb.size()[-1])], dim=-1)
                _o = words_emb * mask # batch, num of subwords in a word, hidden (acctually masked zero)
                we.append(self.word_pooling(_o, dim=1, keepdim=True)) # batch, 1, hidden


            words_emb = torch.cat(we, 1)

            if self.m_pos_emb:
                pos_embs = self.m_pos_emb(pos)
                pos_embs = self.m_pos_dropout(pos_embs)
                words_emb = torch.cat((words_emb, pos_embs), dim=-1) # batch, words_len, hidden
        else:
            if self.m_pos_emb:
                pos_embs = self.m_pos_emb(pos)
                pos_embs = self.m_pos_dropout(pos_embs)
                words_emb = pos_embs
            if self.m_wlsp_emb:
                wlsp_emb = self.m_wlsp_emb(wlsp)
                wlsp_emb = self.m_wlsp_dropout(wlsp_emb)
                if words_emb is not None:
                    words_emb = torch.cat((words_emb, wlsp_emb), dim=-1)
                else:
                    words_emb = wlsp_emb
            
            if self.m_encoder:
                if self.encoder_type == 'transformer':
                    #words_emb = self.m_position_enc(words_emb)
                    words_emb = self.m_encoder(words_emb)  # batch, words_len, hidden 
                else:
                    words_emb, _ = self.m_encoder(words_emb)  # batch, words_len, hidden *2


        for i in range(C):
            ci = chunk_ids.eq(i).unsqueeze(-1)
            mask = torch.cat([ci for _ in range(words_emb.size()[-1])], dim=-1)
            _o = words_emb * mask # batch, num of subwords in a chunk, hidden (acctually masked zero)
            ce.append(self.chunk_pooling(_o, dim=1, keepdim=True)) # batch, 1, hidden

        feats = torch.cat(ce, 1) #


        head = self.m_head(feats)
        dep = self.m_dep(feats)

        dep_out = self.m_depnd(head, dep)
        deprel_out = self.m_deprel(head, dep)

        word_out = self.m_word(words_emb) # batch, len, hidden

        return dep_out, deprel_out, word_out # (batch, chunk_len, chunk_len), (batch, chunk_class, chunk_len, chunk_len), (batch, words_len, word_class)
    
    def loss(self, dep_out: torch.Tensor, deprel_out: torch.Tensor, word_out: torch.Tensor, 
             dep_labels: torch.Tensor, deprel_labels: torch.Tensor, word_labels: torch.Tensor, 
             word_mask: torch.Tensor, dep_mask: torch.Tensor, rel_mask: torch.Tensor, *args, **kwargs) ->Tuple[torch.Tensor]:
        """損失関数

        Args:
            dep_out (~torch.Tensor): 
                [batch, chunk_len, chunk_len] 文節係り受け関係の推定
            deprel_out (~torch.Tensor): 
                [batch, chunk_class, chunk_len, chunk_len] 文節係り受け関係ラベルの推定
            word_out (~torch.Tensor):
                [batch, word_len, word_class] 短単位語に付与されるラベル（文節内係り受けなど）の推定
            dep_labels (~torch.Tensor):
                [batch, chunk_len] 係り受けの正解
            deprel_labels (~torch.Tensor):
                [batch, chunk_len, chunk_class] 係り受け関係ラベルの正解
            word_labels (~torch.Tensor):
                [batch, word_len] 短単位語に付与されるラベル（文節内係り受けなど）の正解
            word_mask (~torch.Tensor):
                [batch, word_len] 短単位語に付与されるラベルの存在範囲を示すmask
            dep_mask (~torch.Tensor):
                [batch, chunk_len] 係り受け関係の存在範囲を示すmask
            rel_mask (~torch.Tensor):
                [batch, chunk_class, chunk_len] 係り受け関係ラベルの存在範囲を示すmask

        Returns:
            Tuple[~torch.Tensor]:
                (全体の損失, 係り受けの損失, 係り受け関係ラベルの損失, 短単位語ラベルの損失)
        """
        out_size = dep_out.size()
        cmask = dep_mask[:out_size[0], :out_size[1]]
        dep_labels = dep_labels[:out_size[0], :out_size[1]]
        dep_loss = self.dep_criterion(dep_out[cmask], dep_labels[cmask])

        out_size = deprel_out.size()
        rmask = rel_mask[:out_size[0], :out_size[2], :out_size[3]]
        rel_labels = deprel_labels[:out_size[0], :out_size[2], :out_size[3]]
        rel_loss = self.rel_criterion(deprel_out.permute((0,2,3,1))[rmask], rel_labels[rmask])

        out_size = word_out.size()
        wmask = word_mask[:out_size[0], :out_size[1]]
        wrd_labels = word_labels[:out_size[0], :out_size[1]]
        wrd_loss = self.wrd_criterion(word_out[wmask], wrd_labels[wmask])

        return dep_loss + self.eps0 * rel_loss + self.eps1 * wrd_loss, dep_loss, rel_loss, wrd_loss


class DistributedDataParallel(nn.parallel.DistributedDataParallel):

    def __init__(self, module, **kwargs):
        super().__init__(module, **kwargs)

    def __getattr__(self, name):
        wrapped = super().__getattr__('module')
        if hasattr(wrapped, name):
            return getattr(wrapped, name)
        return super().__getattr__(name)


def init_device(device, backend='nccl', host=None, port=None):
    os.environ['CUDA_VISIBLE_DEVICES'] = device
    if torch.cuda.device_count() > 1:
        host = host or os.environ.get('MASTER_ADDR', 'localhost')
        port = port or os.environ.get('MASTER_PORT', str(Random(0).randint(10000, 20000)))
        os.environ['MASTER_ADDR'] = host
        os.environ['MASTER_PORT'] = port
        dist.init_process_group(backend)
        torch.cuda.set_device(dist.get_rank())


def is_master():
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0
