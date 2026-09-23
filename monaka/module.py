# -*- coding: utf-8 -*-
"""Monakaで用いる深層学習モジュール群

"""

import math
import torch
import torch.nn as nn
from typing import Dict
from registrable import Registrable
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizerFast
from transformers import AutoModel, AutoConfig, T5EncoderModel


class LMEmbedding(nn.Module, Registrable):
    """言語モデルで用いる埋め込みのための基底クラス

    基底クラス:
        torch.nn.Module: Pytorchのモジュール基底クラス
        Registrable: Registrable基底クラス。テキストで付与された名前でモジュールを呼び出すことができる。

    Arrtributes:
        n_out (int): 出力の次元数
        n_vocab (int): 語彙の語数
    """
   
    def __init__(self, *args, **kwargs) -> None:
        """言語モデルで用いる埋め込みのための基底クラスのinit

        Args:
            n_out (int): 出力の次元数
            n_vocab (int): 語彙の語数
        """
        self.n_out = kwargs.get("n_out", 0)
        self.n_vocab = kwargs.get("n_vocab", 0)
        nn.Module.__init__(self)
        Registrable.__init__(self)

    @classmethod
    def from_config(cls, config: Dict):
        """JSON形式のコンフィグからの生成

        Args:
            config (Dict): JSON形式のコンフィグ

        Returns:
            monaka.module.LMEmbedding: 生成されたクラス
        """
        return cls(**config)
   

@LMEmbedding.register("FixedEmb")
class FixedEmbedding(LMEmbedding):
    """固定長の埋め込み

    基底クラス:
        LMEmbedding: 言語モデルで用いる埋め込みのための基底クラス

    Arrtributes:
        n_out (int): 出力の次元数
        n_vocab (int): 語彙の語数
        padding_index (int): [PAD]のインデックス
    """

    def __init__(self, n_out, n_vocab, padding_index, *args, **kwargs):
        super().__init__(n_out, n_vocab, padding_index, *args, **kwargs)
        self.emb = nn.Embedding(n_vocab, n_out, padding_index)

    def forward(self, subwords):
        return self.emb(subwords)


@LMEmbedding.register("AutoLM")
class AutoLMEmebedding(LMEmbedding):
    """
    TransformersのAutoConfigとAutoModelを利用するEmbedding

    基底クラス:
        LMEmbedding: 言語モデルで用いる埋め込みのための基底クラス

    Arrtributes:
        n_out (int): 
            出力の次元数
        n_vocab (int): 
            語彙の語数
        model (str):
            モデル名
        requires_grad (bool):
            LMを学習するかどうか
        use_scalar_mix (bool):
            ScalarMixを使うかどうか。使わない場合は最終レイヤ
        sclar_mix_dropout (float):
            ScalarMixのdropout
        use_attentions (bool):
            attentionを出力するかどうか
   """
   
    def __init__(self, model: str, requires_grad: bool, use_scalar_mix: bool, sclar_mix_dropout:float = 0.1, use_attentions: bool=False,
                max_length: int=512, **kwargs) -> None:
        self.model = model
        self.requires_grad = requires_grad
        self.use_scalar_mix = use_scalar_mix
        self.sclar_mix_dropout = sclar_mix_dropout
        self.use_attentions = use_attentions
        self.max_length = max_length

        self.config = AutoConfig.from_pretrained(model, output_hidden_states=True,
                                                    output_attentions=use_attentions)
            
        super().__init__(n_out=self.config.hidden_size, n_vocab=self.config.vocab_size)
        self.lm = AutoModel.from_pretrained(model, config=self.config)
        self.lm.requires_grad_(requires_grad)
        self.n_layers = self.config.num_hidden_layers
        self.pad_index = self.config.pad_token_id
        if self.use_scalar_mix:
            self.scalar_mix = ScalarMix(self.n_layers, sclar_mix_dropout)
        
    def __repr__(self):
        s = f"{self.model}, n_out={self.n_out}"
        s += f", pad_index={self.pad_index}"
        s += f", use_scalar_mix={self.use_scalar_mix}"
        if self.sclar_mix_dropout > 0:
            s += f", mix_dropout={self.sclar_mix_dropout}"
        if self.use_attentions:
            s += f", use_attentions={self.use_attentions}"
        if self.requires_grad:
            s += f", requires_grad={self.requires_grad}"

        return f"{self.__class__.__name__}({s})"

    def forward(self, subwords):
        r"""
        Args:
            subwords (~torch.Tensor): ``[batch_size, subwordlen]``.
        Returns:
            ~torch.Tensor:
                BERT embeddings of shape ``[batch_size, seq_len, n_out]``.
        """

        mask = subwords.ne(self.pad_index)
        if not self.requires_grad:
            self.lm.eval()    # CHECK_ME (supar does not do)
        # [batch_size, n_subwords]
        # Outputs from the transformer:
        # - last_hidden_state: [batch, seq_len, hidden_size]
        # - pooler_output: [batch, hidden_size],
        # - hidden_states (optional): [[batch_size, seq_length, hidden_size]] * (1 + layers)
        # - attentions (optional): [[batch_size, num_heads, seq_length, seq_length]] * layers
        # print('<BERT, GPU MiB:', memory_allocated() // (1024*1024)) # DEBUG
        outputs = self.lm(subwords, attention_mask=mask.float())
        # print('BERT>, GPU MiB:', memory_allocated() // (1024*1024)) # DEBUG
        if self.use_scalar_mix:
            bert_idx = -2 if self.use_attentions else -1
            bert = outputs[bert_idx]
            # [n_layers, batch_size, n_subwords, hidden_size]
            bert = bert[-self.n_layers:]
            # [batch_size, n_subwords, hidden_size]
            bert = self.scalar_mix(bert)
        else:
            bert = outputs[0]
        

        return bert

@LMEmbedding.register("T5Encoder")
class T5EncoderEmbedding(AutoLMEmebedding):
    """
    T5Encoderを利用するEmbedding

    基底クラス:
        AutoLMEmebedding: TransformersのAutoConfigとAutoModelを利用するEmbedding

    Arrtributes:
        n_out (int): 
            出力の次元数
        n_vocab (int): 
            語彙の語数
        model (str):
            モデル名
        requires_grad (bool):
            LMを学習するかどうか
        use_scalar_mix (bool):
            ScalarMixを使うかどうか。使わない場合は最終レイヤ
        sclar_mix_dropout (float):
            ScalarMixのdropout
        use_attentions (bool):
            attentionを出力するかどうか
        max_len (int):
            最大subword長 default 5120
    """
   
    def __init__(self, model: str, requires_grad: bool, use_scalar_mix: bool, sclar_mix_dropout:float = 0.1, use_attentions: bool=False, max_len: int=5120, **kwargs) -> None:
        self.model = model
        self.requires_grad = requires_grad
        self.use_scalar_mix = use_scalar_mix
        self.sclar_mix_dropout = sclar_mix_dropout
        self.use_attentions = use_attentions
        self.max_len = max_len

        self.config = AutoConfig.from_pretrained(model, output_hidden_states=True,
                                                output_attentions=use_attentions)
        self.lm = T5EncoderModel.from_pretrained(model, config=self.config)
        self.lm.requires_grad_(requires_grad)
        self.n_layers = self.config.num_hidden_layers
        self.pad_index = self.config.pad_token_id
        self.sclar_mix = ScalarMix(self.n_layers, sclar_mix_dropout)

        LMEmbedding.__init__(self, n_out=self.config.hidden_size, n_vocab=self.config.vocab_size)
    


class MLP(nn.Module):
    r"""
    多重線形パーセプトロン層 (Multi-Layered Perceptron; MLP) :class:`~torch.nn.LeakyReLU` activation:
    :math:`y = \mathrm{LeakyReLU}(x A^T + b)`

    Arrtributes:
        n_in (int):
            入力次元数
        n_out (int):
            出力次元数
        dropout (float):
            もしゼロでないなら :class:`SharedDropout` を出力に指定のDropout率で付与。 Default: 0.
    """

    def __init__(self, n_in, n_out, dropout=0):
        super().__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.linear = nn.Linear(n_in, n_out)
        self.activation = nn.LeakyReLU(negative_slope=0.1)
        self.dropout = nn.Dropout(p=dropout)

        self.reset_parameters()

    def __repr__(self):
        s = f"n_in={self.n_in}, n_out={self.n_out}"
        if self.dropout.p > 0:
            s += f", dropout={self.dropout.p}"

        return f"{self.__class__.__name__}({s})"

    def reset_parameters(self):
        nn.init.orthogonal_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        r"""
        Args:
            x (~torch.Tensor):
                特徴量の次元数が `n_in`　であるテンソル.

        Returns:
            `n_out`次元数の特徴量を持つテンソル.
        """

        x = self.linear(x)
        x = self.activation(x)
        x = self.dropout(x)

        return x


class Biaffine(nn.Module):
    """Biaffine注意層

    Args:
        n_in (int):
            入力次元数
        n_out (int):
            出力次元数
        bias_x (bool):
            切片(bias)を左側のaffine変換に追加するか
        bias_y (bool):
            切片(bias)を右側のaffine変換に追加するか
    """

    def __init__(self, n_in, n_out=1, bias_x=True, bias_y=True):
        super(Biaffine, self).__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.bias_x = bias_x
        self.bias_y = bias_y
        self.weight = nn.Parameter(torch.Tensor(n_out,
                                                n_in + bias_x,
                                                n_in + bias_y))
        self.reset_parameters()

    def extra_repr(self):
        s = f"n_in={self.n_in}, n_out={self.n_out}"
        if self.bias_x:
            s += f", bias_x={self.bias_x}"
        if self.bias_y:
            s += f", bias_y={self.bias_y}"

        return s

    def reset_parameters(self):
        nn.init.zeros_(self.weight)

    def forward(self, x, y):
        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), -1)
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), -1)
        # [batch_size, n_out, seq_len, seq_len]
        s = torch.einsum('bxi,oij,byj->boxy', x, self.weight, y)
        # remove dim 1 if n_out == 1
        s = s.squeeze(1)

        return s


class ScalarMix(nn.Module):
    r"""言語モデルの各層に重みをつけて混ぜ合わせる層(ScalarMix)
    ScalarMixの計算 :math:`N` tensors, :math:`mixture = \gamma * \sum_{k}(s_k * tensor_k)`
    :math:`s = \mathrm{softmax}(w)`, with :math:`w` and :math:`\gamma` scalar parameters.

    Args:
        n_layers (int):
            混ぜ合わせる層数, i.e., :math:`N`.
        dropout (float):
            ドロップアウト率
    """

    def __init__(self, n_layers: int, dropout: float = 0.0):
        super().__init__()

        self.n_layers = n_layers

        self.weights = nn.Parameter(torch.zeros(n_layers))
        self.gamma = nn.Parameter(torch.tensor([1.0]))
        self.dropout = nn.Dropout(dropout)

    def __repr__(self):
        s = f"n_layers={self.n_layers}"
        if self.dropout.p > 0:
            s += f", dropout={self.dropout.p}"

        return f"{self.__class__.__name__}({s})"

    def forward(self, tensors):
        r"""
        Args:
            tensors (list[~torch.Tensor]):
                :math:`N` 混ぜ合わせるテンソル.

        Returns:
            混ぜ合わせたテンソル :math:`N`.
        """

        normed_weights = self.dropout(self.weights.softmax(-1))
        weighted_sum = sum(w * h for w, h in zip(normed_weights, tensors))

        return self.gamma * weighted_sum


class SelfAttentionPooling(nn.Module):
    """自己注意Pooling機構
    重要度を自分で判定して（自己注意）Poolingする機構
    Original Paper: Self-Attention Encoding and Pooling for Speaker Recognition
    https://arxiv.org/pdf/2008.01077v1.pdf
    """
    def __init__(self, input_dim):
        super(SelfAttentionPooling, self).__init__()
        self.W = nn.Linear(input_dim, 1)
        self.input_dim = input_dim
        
    def forward(self, batch_rep, dim=1, keepdim=True):
        """
        input:
            batch_rep : size (N, T, H), N: batch size, T: 系列長, H: 隠れ層
        
        attention_weight:
            att_w : size (N, T, 1)
        
        Returns:
            ~torch.Tensor:
                系列中のテンソルをまとめ上げる: size (N, H)
        """
        softmax = nn.functional.softmax
        att_w = softmax(self.W(batch_rep).squeeze(-1), dim=dim).unsqueeze(-1)
        utter_rep = torch.sum(batch_rep * att_w, dim=1)


        if keepdim:
            utter_rep = utter_rep.unsqueeze(dim)

        return utter_rep
    

class PositionalEncoding(nn.Module):
    """PositionEncoding: 位置エンコーディング

    Transformerで失われる位置情報を埋め込みとして表現する
    Args:
        d_model (int):
            位置エンコーディングの次元数
        dropout (float):
            ドロップアウト率
        max_len (int):
            最大長
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (~torch.Tensor):
                shape ``[seq_len, batch_size, embedding_dim]``
        
        Returns:
            ~torch.Tensor:
                shape ``[seq_len, batch_size, embedding_dim]``

        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)