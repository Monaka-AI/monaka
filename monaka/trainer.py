# -*- coding: utf-8 -*-
"""モデル学習を行うクラス群"""

import os
import json
import logging
import datetime

import torch
import tqdm

import torch.nn as nn
import torch.distributed as dist
from torch.optim import Adam
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from registrable import Registrable

from typing import List, Optional, Union, Dict
from monaka.dataset import LUWJsonLDataset, LemmaJsonDataset, ChunkDepJsonLDataset
from monaka.mylogging import init_logger, get_logger
from monaka.model import LUWParserModel, LUWLemmaModel, init_device, is_master
from monaka.model import DistributedDataParallel as DDP

import random
import numpy as np

logger = None

def torch_fix_seed(seed=419):
    """実験再現性のためにPyTorchの乱数シード値を指定する関数

    Args:
        seed (int, optional): 
            乱数シード値. Defaults to 419.
    """
    # Python random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True

class Trainer(Registrable):
    """モデル学習の基底クラス

    Args:
        device (int):
            学習を行う際のGPU番号。負の値はCPU利用
    """
    
    def __init__(self, *args, **kwargs):
        Registrable.__init__(self)

    def train(self, device: int=-1, local_rank: int=-1):
        raise NotImplementedError
    
    def evaluate(self, dataloader, device):
        raise NotImplementedError


@Trainer.register("dependency")
class DependencyTrainer(Trainer):
    """文節係り受け学習クラス

    Registrableで呼び出す際の名前は "dependency"

    Args:
        train_files (Union[str, List[str]]): 
            学習セットのファイルパス（複数可）学習には :py:class:`~monaka.dataset.ChunkDepJsonLDataset` 準拠のJSON-Lデータである必要がある
        dev_files (Union[str, List[str]]):
            検証セットのファイルパス（複数可） :py:class:`~monaka.dataset.ChunkDepJsonLDataset` 準拠のJSON-Lデータである必要がある
        test_files (Optional[Union[str, List[str]]]): 
            評価セットのファイルパス（複数可） :py:class:`~monaka.dataset.ChunkDepJsonLDataset` 準拠のJSON-Lデータである必要がある
        dataeset_options (Dict): 
            :py:class:`~monaka.dataset.ChunkDepJsonLDataset` に受け渡す設定データ
        model_name (str):
            用いるモデル名。:py:class:`~monaka.model.LUWParserModel` の派生クラスであり、:py:class:`~Registrable`の機能を使って、クラス定義時に付与されたモデル名を指定することができる。例： "ChunkDep"
        model_config (Dict): 
            モデルに与える設定データ
        batch_size (int, optional): 
            バッチサイズ. Defaults to 8.
        epochs (int, optional): 
            エポック数. Defaults to 1.
        lr (float, optional): 
            学習率。最適化はAdamで固定。. Defaults to 2e-5.
        mu (float, optional): 
            Adamのmu. Defaults to .9.
        nu (float, optional): 
            Adamのnu. Defaults to .9.
        epsilon (float, optional): 
            Adamのepsilon. Defaults to 1e-12.
        clip (float, optional): 
            勾配クリッピング. Defaults to 5.0.
        decay (float, optional): 
            学習率減衰. Defaults to .75.
        decay_steps (float, optional): 
            学習率減衰の基準ステップ数. Defaults to 5000.
        patience (float, optional): 
            学習率減衰が始まるまでの最初のステップ数. Defaults to 100.
        evaluate_step (int, optional): 
            評価ステップ数。この数で割り切れる学習ステップ数時に評価を実行する. Defaults to 20.
        verbose (bool, optional): 
            デバッグ用の情報等を多く出力するようにするオプション. Defaults to True.
        seed (int, optional): 
            乱数シード値. Defaults to 419.
        output_dir (str, optional): 
            モデルの出力フォルダ. Defaults to "".
    """

    def __init__(self,
            train_files: Union[str, List[str]],
            dev_files: Union[str, List[str]],
            test_files: Optional[Union[str, List[str]]],
            dataeset_options: Dict,
            model_name: str,
            model_config: Dict,
            batch_size: int=8,
            epochs: int=1,
            lr: float=2e-5,
            mu: float=.9,
            nu: float=.9,
            epsilon: float=1e-12,
            clip: float=5.0,
            decay: float=.75,
            decay_steps: float=5000,
            patience: float=100,
            evaluate_step:int =20,
            verbose: bool=True,
            seed: int = 419,
            output_dir: str="",
            **kwargs):
        
        global logger
        os.makedirs(output_dir, exist_ok=True)

        logger = get_logger(f"monaka.trainer.{output_dir.replace('/', '.')}")
        init_logger(logger, handlers=[logging.StreamHandler(), logging.FileHandler(f"{output_dir}/train.{datetime.datetime.now().timestamp()}.log", 'w')], verbose=verbose)
        self.output_dir = output_dir

        logger.info("dataset options:")
        logger.info(json.dumps(dataeset_options, indent=True, ensure_ascii=False))
        options = {"logger": logger}
        options.update(dataeset_options)
        logger.info("loading train files")
        self.train_data = ChunkDepJsonLDataset(train_files, **options)

        label_dic = self.train_data.label_dic
        with open(os.path.join(output_dir, "labels.json"), "w") as f:
            json.dump(label_dic, f, indent=True, ensure_ascii=False)

        rel_dic = self.train_data.rel_dic
        with open(os.path.join(output_dir, "rels.json"), "w") as f:
            json.dump(rel_dic, f, indent=True, ensure_ascii=False)

        pos_dic = getattr(self.train_data, "pos_dic", None)
        if pos_dic is not None:
            with open(os.path.join(output_dir, "pos.json"), "w") as f:
                json.dump(pos_dic, f, indent=True, ensure_ascii=False)

        wlsp_dic = getattr(self.train_data, "wlsp_dic", None)
        if wlsp_dic is not None:
            with open(os.path.join(output_dir, "wlsp_dic.json"), "w") as f:
                json.dump(wlsp_dic, f, indent=True, ensure_ascii=False)

        logger.info("loading dev files")
        self.dev_data = ChunkDepJsonLDataset(dev_files, **options)

        logger.info("loading test files")
        self.test_data = ChunkDepJsonLDataset(test_files, **options) if test_files else None

        self.batch_size=batch_size
        self.epochs = epochs
        self.lr = lr
        self.mu = mu
        self.nu = nu
        self.epsilon = epsilon
        self.clip = clip
        self.decay = decay
        self.decay_steps = decay_steps
        self.patience = patience
        self.verbose = verbose
        self.evaluate_step = evaluate_step
        self.model_name = model_name
        if seed < 0:
            seed = np.random.randint(1, 1024*1024)
        conf = {
            "batch_size": batch_size,
            "epochs": epochs,
            "mu": mu,
            "nu": nu,
            "epsilon": epsilon,
            "clip": clip,
            "decay": decay,
            "decay_steps": decay_steps,
            "patience": patience,
            "verbose": verbose,
            "evaluate_step": evaluate_step,
            "seed": seed
        }
        torch_fix_seed(seed)
        conf.update(kwargs)

        logger.info("loading model")
        self.model = LUWParserModel.by_name(model_name).from_config(model_config, **dataeset_options)
        logger.info(str(self.model))
        logger.info(json.dumps(model_config, indent=True, ensure_ascii=False))

        logger.info("training setup:")
        logger.info(json.dumps(conf, indent=True, ensure_ascii=False))

        if dist.is_initialized():
            logger.info("distributed mode ON")
            self.model = DDP(self.model,
                             device_ids=[dist.get_rank()],
                             find_unused_parameters=True)

    def train(self, device: int=-1, local_rank: int=-1):
        """学習実行

        Args:
            device (int, optional):
                学習に使うGPU。負の値はCPU利用. Defaults to -1.
            local_rank (int, optional):
                複数GPU用のオプションだが現状は動作しない. Defaults to -1.
        """
        init_device(str(device), local_rank)
        if dist.is_initialized():
            self.batch_size = self.batch_size // dist.get_world_size()
        try:
            device = int(device)
        except:
            pass
        self.model.to(device)

        optimizer = Adam(self.model.parameters(),
                              self.lr,
                              (self.mu, self.nu),
                              self.epsilon)
        scheduler = ExponentialLR(optimizer, self.decay**(1/self.decay_steps))
        writer = SummaryWriter(log_dir=os.path.join(self.output_dir, "tb"))

        train_loader = DataLoader(self.train_data, self.batch_size, shuffle=True, collate_fn=ChunkDepJsonLDataset.collate_function)
        dev_loader = DataLoader(self.dev_data, batch_size=self.batch_size, shuffle=False, collate_fn=ChunkDepJsonLDataset.collate_function)
        test_loader = DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False, collate_fn=ChunkDepJsonLDataset.collate_function) if self.test_data else None
        metric = -1
        total_itr = 0

        for epoch in range(1, self.epochs + 1):
            start = datetime.datetime.now()

            logger.info(f"Epoch {epoch} / {self.epochs}:")

            for i, data in tqdm.tqdm(enumerate(train_loader)):
                subwords = pad_sequence(data["input_ids"], batch_first=True, padding_value=self.train_data.pad_token_id).to(device) if 'input_ids' in data else None
                if 'input_ids' in data:
                    word_ids = pad_sequence([torch.LongTensor(js.word_ids()) for js in data["subwords"]], batch_first=True, padding_value=-1).to(device)
                else:
                    word_ids = pad_sequence([torch.LongTensor([i for i in range(len(tokens))]) for tokens in data['tokens']]  , batch_first=True, padding_value=-1).to(device)
                chunk_ids = pad_sequence(data["chunk_ids"], batch_first=True, padding_value=-1).to(device)
                dep_ids = pad_sequence(data["dep_ids"], batch_first=True, padding_value=-1).to(device)
                word_rel_ids = pad_sequence(data["word_rel_ids"], batch_first=True, padding_value=self.train_data.pad_token_id).to(device)
                dep_rel_ids = pad_sequence(data["dep_rel_ids"], batch_first=True, padding_value=self.train_data.pad_token_id).to(device)
                pos_ids = pad_sequence(data["pos_ids"], batch_first=True, padding_value=1).to(device) if "pos_ids" in data else None
                wlsp_ids = pad_sequence(data["wlsp_ids"], batch_first=True, padding_value=1).to(device) if "wlsp_ids" in data else None
                wmask = word_rel_ids.ne(1)
                dmask = dep_ids.ne(-1)
                rmask = dep_rel_ids.ne(1)

                dep_out, deprel_out, word_out  = self.model(subwords, word_ids, chunk_ids, pos_ids, wlsp_ids)
                loss, dep_loss, rel_loss, wrd_loss = self.model.loss(dep_out, deprel_out, word_out , dep_ids, dep_rel_ids, word_rel_ids, wmask, dmask, rmask)
                writer.add_scalar("Loss/train", loss, total_itr + i)
                writer.add_scalar("DependencyLoss/train", dep_loss, total_itr + i)
                writer.add_scalar("RelationLoss/train", rel_loss, total_itr + i)
                writer.add_scalar("WordRelLoss/train", wrd_loss, total_itr + i)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
                optimizer.step()
                scheduler.step()
                if (i+1) % self.evaluate_step == 0:
                    dev_loss, dev_dep_loss, dev_rel_loss, dev_wrd_loss, dev_dep_acc, dev_rel_acc, dev_wrd_acc = self.evaluate(dev_loader, device)
                    writer.add_scalar("Loss/dev", dev_loss, total_itr + i)
                    writer.add_scalar("DependencyLoss/dev", dev_dep_loss, total_itr + i)
                    writer.add_scalar("RelationLoss/dev", dev_rel_loss, total_itr + i)
                    writer.add_scalar("WordRelLoss/dev", dev_wrd_loss, total_itr + i)
                    writer.add_scalar("DependencyAcc/dev", dev_dep_acc, total_itr + i)
                    writer.add_scalar("RelationAcc/dev", dev_rel_acc, total_itr + i)
                    writer.add_scalar("WordRelAcc/dev", dev_wrd_acc, total_itr + i)

            total_itr += i
            t = datetime.datetime.now() - start
            logger.info("dev evaluation")
            dev_loss, dev_dep_loss, dev_rel_loss, dev_wrd_loss, dev_dep_acc, dev_rel_acc, dev_wrd_acc = self.evaluate(dev_loader, device)
            writer.add_scalar("Loss/dev", dev_loss, total_itr)
            writer.add_scalar("DependencyLoss/dev", dev_dep_loss, total_itr)
            writer.add_scalar("RelationLoss/dev", dev_rel_loss, total_itr)
            writer.add_scalar("WordRelLoss/dev", dev_wrd_loss, total_itr)
            writer.add_scalar("DependencyAcc/dev", dev_dep_acc, total_itr)
            writer.add_scalar("RelationAcc/dev", dev_rel_acc, total_itr)
            writer.add_scalar("WordRelAcc/dev", dev_wrd_acc, total_itr)

            if dev_dep_acc > metric:
                logger.info("save best model")
                self.save(os.path.join(self.output_dir, f"best.pt"))
                metric = dev_dep_acc

            if test_loader:
                logger.info("test evaluation")
                test_loss, test_dep_loss, test_rel_loss, test_wrd_loss, test_dep_acc, test_rel_acc, test_wrd_acc = self.evaluate(test_loader, device)
                writer.add_scalar("Loss/test", test_loss, total_itr)
                writer.add_scalar("DependencyLoss/test", test_dep_loss, total_itr)
                writer.add_scalar("RelationLoss/test", test_rel_loss, total_itr)
                writer.add_scalar("WordRelLoss/test", test_wrd_loss, total_itr)
                writer.add_scalar("DependencyAcc/test", test_dep_acc, total_itr)
                writer.add_scalar("RelationAcc/test", test_rel_acc, total_itr)
                writer.add_scalar("WordRelAcc/test", test_wrd_acc, total_itr)

            logger.info(f"{t}s elapsed\n")

        self.save(os.path.join(self.output_dir, f"last_at_{epoch}.pt"))
        
        
    @torch.no_grad()
    def evaluate(self, dataloader, device):
        """評価実行

        Args:
            dataloader (~torch.utils.data.DataLoader): 評価データを読み込んだデータローダ
            device (int): 用いるGPU。trainと同様。

        Raises:
            e (~Exception): 評価時に何らかのエラーが発生した場合 

        Returns:
            tupple: 
                (全体損失, 係り受け損失, 係り受け関係ラベル損失, 短単位語ラベル損失, 係り受け正解率, 係り受け関係ラベル正解率, 短単位語ラベル正解率)
        """
        dep_correct = 0
        dep_length = 0
        rel_correct = 0
        rel_length = 0
        wrd_correct = 0
        wrd_length = 0

        loss = 0
        dep_loss = 0
        rel_loss = 0
        wrd_loss = 0
        self.model.eval()
        for data in dataloader:
                subwords = pad_sequence(data["input_ids"], batch_first=True, padding_value=self.train_data.pad_token_id).to(device) if 'input_ids' in data else None
                if 'input_ids' in data:
                    word_ids = pad_sequence([torch.LongTensor(js.word_ids()) for js in data["subwords"]], batch_first=True, padding_value=-1).to(device)
                else:
                    word_ids = pad_sequence([torch.LongTensor([i for i in range(len(tokens))]) for tokens in data['tokens']]  , batch_first=True, padding_value=-1).to(device)
                chunk_ids = pad_sequence(data["chunk_ids"], batch_first=True, padding_value=-1).to(device)
                dep_ids = pad_sequence(data["dep_ids"], batch_first=True, padding_value=-1).to(device)
                word_rel_ids = pad_sequence(data["word_rel_ids"], batch_first=True, padding_value=1).to(device)
                dep_rel_ids = pad_sequence(data["dep_rel_ids"], batch_first=True, padding_value=1).to(device)
                pos_ids = pad_sequence(data["pos_ids"], batch_first=True, padding_value=1).to(device) if "pos_ids" in data else None
                wlsp_ids = pad_sequence(data["wlsp_ids"], batch_first=True, padding_value=1).to(device) if "wlsp_ids" in data else None
                wmask = word_rel_ids.ne(1)
                dmask = dep_ids.ne(-1)
                rmask = dep_rel_ids.ne(1)

                dep_out, deprel_out, word_out  = self.model(subwords, word_ids, chunk_ids, pos_ids, wlsp_ids)
                l, dep_l, rel_l, wrd_l = self.model.loss(dep_out, deprel_out, word_out , dep_ids, dep_rel_ids, word_rel_ids, wmask, dmask, rmask)
                loss += l.detach().cpu().item()
                dep_loss += dep_l.detach().cpu().item()
                rel_loss += rel_l.detach().cpu().item()
                wrd_loss += wrd_l.detach().cpu().item()
                dep_pred = torch.argmax(dep_out, dim=-1)
                rel_pred = torch.argmax(deprel_out, dim=1)
                wrd_pred = torch.argmax(word_out, dim=-1)
                try:
                    dep_correct += ((dep_pred == dep_ids) & dmask).sum().detach().cpu().item()
                    dep_length += (dmask).sum().detach().cpu().item()
                    rel_size = rel_pred.size()
                    rel_correct += ((rel_pred == dep_rel_ids[:, :rel_size[1], :rel_size[2]]) & rmask[:, :rel_size[1], :rel_size[2]]).sum().detach().cpu().item()
                    rel_length += (rmask[:, :rel_size[1], :rel_size[2]]).sum().detach().cpu().item()
                    wrd_correct += ((wrd_pred == word_rel_ids) & wmask).sum().detach().cpu().item()
                    wrd_length += (wmask).sum().detach().cpu().item()
                except Exception as e:
                    raise e
                    logger.info(f"evaluation skipped: {data['text']}")
        logger.info(f"dep accuracy: {dep_correct/dep_length*100}, rel accuracy: {rel_correct/rel_length*100}, word rel accuracy: {wrd_correct/wrd_length*100}, loss: {loss}")
        self.model.train(True)
        return loss, dep_loss, rel_loss, wrd_loss, dep_correct/dep_length, rel_correct/rel_length, wrd_correct/wrd_length
    
    def save(self, path):
        """モデルの保存

        Args:
            path (str): 保存先パス
        """
        model = self.model
        if hasattr(model, 'module'):
            model = self.model.module
        state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        torch.save(state_dict, path)


@Trainer.register("segmentation")
class SegmentationTrainer(Trainer):
    """文節・長単位境界と品詞を推定するモデル

    Registrableで呼び出す際の名前は "segmentation"
    基本的には短単位語を基準とする系列ラベリング器であるので、長単位語に限らず利用できる

    Args:
        train_files (Union[str, List[str]]): 
            学習セットのファイルパス（複数可）学習には :py:class:`~monaka.dataset.LUWJsonLDataset` 準拠のJSON-Lデータである必要がある
        dev_files (Union[str, List[str]]):
            検証セットのファイルパス（複数可） :py:class:`~monaka.dataset.LUWJsonLDataset` 準拠のJSON-Lデータである必要がある
        test_files (Optional[Union[str, List[str]]]): 
            評価セットのファイルパス（複数可） :py:class:`~monaka.dataset.LUWJsonLDataset` 準拠のJSON-Lデータである必要がある
        dataeset_options (Dict): 
            :py:class:`~monaka.dataset.LUWJsonLDataset` に受け渡す設定データ
        model_name (str):
            用いるモデル名。:py:class:`~monaka.model.LUWParserModel` の派生クラスであり、:py:class:`~Registrable`の機能を使って、クラス定義時に付与されたモデル名を指定することができる。例： "WordTagging"
        model_config (Dict): 
            モデルに与える設定データ
        batch_size (int, optional): 
            バッチサイズ. Defaults to 8.
        epochs (int, optional): 
            エポック数. Defaults to 1.
        lr (float, optional): 
            学習率。最適化はAdamで固定。. Defaults to 2e-5.
        mu (float, optional): 
            Adamのmu. Defaults to .9.
        nu (float, optional): 
            Adamのnu. Defaults to .9.
        epsilon (float, optional): 
            Adamのepsilon. Defaults to 1e-12.
        clip (float, optional): 
            勾配クリッピング. Defaults to 5.0.
        decay (float, optional): 
            学習率減衰. Defaults to .75.
        decay_steps (float, optional): 
            学習率減衰の基準ステップ数. Defaults to 5000.
        patience (float, optional): 
            学習率減衰が始まるまでの最初のステップ数. Defaults to 100.
        evaluate_step (int, optional): 
            評価ステップ数。この数で割り切れる学習ステップ数時に評価を実行する. Defaults to 20.
        verbose (bool, optional): 
            デバッグ用の情報等を多く出力するようにするオプション. Defaults to True.
        seed (int, optional): 
            乱数シード値. Defaults to 419.
        output_dir (str, optional): 
            モデルの出力フォルダ. Defaults to "".
    """

    def __init__(self,
            train_files: Union[str, List[str]],
            dev_files: Union[str, List[str]],
            test_files: Optional[Union[str, List[str]]],
            dataeset_options: Dict,
            model_name: str,
            model_config: Dict,
            batch_size: int=8,
            epochs: int=1,
            lr: float=2e-5,
            mu: float=.9,
            nu: float=.9,
            epsilon: float=1e-12,
            clip: float=5.0,
            decay: float=.75,
            decay_steps: float=5000,
            patience: float=100,
            evaluate_step:int =20,
            verbose: bool=True,
            seed: int = 419,
            output_dir: str="",
            **kwargs):
        
        global logger
        os.makedirs(output_dir, exist_ok=True)

        logger = get_logger(f"monaka.trainer.{output_dir.replace('/', '.')}")
        init_logger(logger, handlers=[logging.StreamHandler(), logging.FileHandler(f"{output_dir}/train.log", 'w')], verbose=verbose)
        self.output_dir = output_dir

        logger.info("dataset options:")
        logger.info(json.dumps(dataeset_options, indent=True, ensure_ascii=False))
        options = {"logger": logger}
        options.update(dataeset_options)
        logger.info("loading train files")
        self.train_data = LUWJsonLDataset(train_files, **options)

        label_dic = self.train_data.label_dic
        with open(os.path.join(output_dir, "labels.json"), "w") as f:
            json.dump(label_dic, f, indent=True, ensure_ascii=False)

        pos_dic = getattr(self.train_data, "pos_dic", None)
        if pos_dic is not None:
            with open(os.path.join(output_dir, "pos.json"), "w") as f:
                json.dump(pos_dic, f, indent=True, ensure_ascii=False)

        logger.info("loading dev files")
        self.dev_data = dict()
        for dev_f in dev_files:
            logger.info(f"loading dev file: {dev_f}")
            self.dev_data[dev_f] = LUWJsonLDataset([dev_f], **options)

        logger.info("loading test files")
        if test_files:
            self.test_data = dict()
            for test_f in test_files:
                logger.info(f"loading test file: {test_f}")
                self.test_data[test_f] = LUWJsonLDataset([test_f], **options)
        else:
            self.test_data = None

        self.batch_size=batch_size
        self.epochs = epochs
        self.lr = lr
        self.mu = mu
        self.nu = nu
        self.epsilon = epsilon
        self.clip = clip
        self.decay = decay
        self.decay_steps = decay_steps
        self.patience = patience
        self.verbose = verbose
        self.evaluate_step = evaluate_step
        self.model_name = model_name
        conf = {
            "batch_size": batch_size,
            "epochs": epochs,
            "mu": mu,
            "nu": nu,
            "epsilon": epsilon,
            "clip": clip,
            "decay": decay,
            "decay_steps": decay_steps,
            "patience": patience,
            "verbose": verbose,
            "evaluate_step": evaluate_step,
            "seed": seed
        }
        torch_fix_seed(seed)
        conf.update(kwargs)

        logger.info("loading model")
        self.model = LUWParserModel.by_name(model_name).from_config(model_config, **dataeset_options)
        logger.info(str(self.model))
        logger.info(json.dumps(model_config, indent=True, ensure_ascii=False))

        logger.info("training setup:")
        logger.info(json.dumps(conf, indent=True, ensure_ascii=False))

        if dist.is_initialized():
            logger.info("distributed mode ON")
            self.model = DDP(self.model,
                             device_ids=[dist.get_rank()],
                             find_unused_parameters=True)

    def train(self, device: int=-1, local_rank: int=-1):
        """学習実行

        Args:
            device (int, optional):
                学習に使うGPU。負の値はCPU利用. Defaults to -1.
            local_rank (int, optional):
                複数GPU用のオプションだが現状は動作しない. Defaults to -1.
        """
        init_device(str(device), local_rank)
        if dist.is_initialized():
            self.batch_size = self.batch_size // dist.get_world_size()
        try:
            device = int(device)
        except:
            pass
        self.model.to(device)

        optimizer = Adam(self.model.parameters(),
                              self.lr,
                              (self.mu, self.nu),
                              self.epsilon)
        scheduler = ExponentialLR(optimizer, self.decay**(1/self.decay_steps))
        writer = SummaryWriter(log_dir=os.path.join(self.output_dir, "tb"))

        train_loader = DataLoader(self.train_data, self.batch_size, shuffle=True, collate_fn=LUWJsonLDataset.collate_function)
        
        dev_loader = dict()
        for k, d in self.dev_data.items():
            dev_loader[k] = DataLoader(d, batch_size=self.batch_size, shuffle=False, collate_fn=LUWJsonLDataset.collate_function)
        
        if self.test_data:
            test_loader = dict()
            for k, d in self.test_data.items():
                test_loader[k] = DataLoader(d, batch_size=self.batch_size, shuffle=False, collate_fn=LUWJsonLDataset.collate_function) if self.test_data else None
        metric = -1
        total_itr = 0

        for epoch in range(1, self.epochs + 1):
            start = datetime.datetime.now()

            logger.info(f"Epoch {epoch} / {self.epochs}:")

            for i, data in tqdm.tqdm(enumerate(train_loader)):
                if 'input_ids' not in data:
                    logger.error(data)
                subwords = pad_sequence(data["input_ids"], batch_first=True, padding_value=self.train_data.pad_token_id).to(device)
                word_ids = pad_sequence([torch.LongTensor(js.word_ids()) for js in data["subwords"]], batch_first=True, padding_value=-1).to(device)
                label_ids = pad_sequence(data["label_ids"], batch_first=True, padding_value=1).to(device)
                pos_ids = pad_sequence(data["pos_ids"], batch_first=True, padding_value=1).to(device) if "pos_ids" in data else None
                mask = label_ids.ne(1)

                out = self.model(subwords, word_ids, pos_ids)
                loss = self.model.loss(out, label_ids, mask)
                writer.add_scalar("Loss/train", loss, total_itr + i)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
                optimizer.step()
                scheduler.step()
                if (i+1) % self.evaluate_step == 0:
                    dev_loss = 0
                    dev_acc = 0 
                    c = 0
                    for k, l in dev_loader.items():
                        c += 1
                        dl, da = self.evaluate(l, device)
                        dev_acc += da
                        dev_loss += dl
                        logger.info(f"dev evaluation: {k}")
                        writer.add_scalar(f"Loss/{k}/dev", dev_loss, total_itr + i)
                        writer.add_scalar(f"Acc/{k}/dev", dev_acc, total_itr + i)
                    
                    writer.add_scalar("Loss/dev", dev_loss/c, total_itr + i)
                    writer.add_scalar("Acc/dev", dev_acc/c, total_itr + i)

            total_itr += i
            t = datetime.datetime.now() - start
            logger.info("dev evaluation")
            dev_loss = 0
            dev_acc = 0 
            c = 0
            for k, l in dev_loader.items():
                c += 1
                dl, da = self.evaluate(l, device)
                dev_acc += da
                dev_loss += dl
                logger.info(f"dev evaluation: {k}")
                writer.add_scalar(f"Loss/{k}/dev", dl, total_itr + i)
                writer.add_scalar(f"Acc/{k}/dev", da, total_itr + i)
            
            writer.add_scalar("Loss/dev", dev_loss/c, total_itr + i)
            writer.add_scalar("Acc/dev", dev_acc/c, total_itr + i)

            if dev_acc > metric:
                logger.info("save best model")
                self.save(os.path.join(self.output_dir, f"best_at_{epoch}.pt"))
                metric = dev_acc

            if test_loader:
                logger.info("test evaluation")
                test_loss = 0
                test_acc = 0 
                c = 0
                for k, l in test_loader.items():
                    c += 1
                    dl, da = self.evaluate(l, device)
                    test_acc += da
                    test_loss += dl
                    logger.info(f"test evaluation: {k}")
                    writer.add_scalar(f"Loss/{k}/test", dl, total_itr + i)
                    writer.add_scalar(f"Acc/{k}/test", da, total_itr + i)
                
                writer.add_scalar("Loss/test", test_loss/c, total_itr + i)
                writer.add_scalar("Acc/test", test_acc/c, total_itr + i)
            logger.info(f"{t}s elapsed\n")

        self.save(os.path.join(self.output_dir, f"last_at_{epoch}.pt"))
        
        
    @torch.no_grad()
    def evaluate(self, dataloader, device):
        """評価実行

        Args:
            dataloader (~torch.utils.data.DataLoader): 評価データを読み込んだデータローダ
            device (int): 用いるGPU。trainと同様。

        Returns:
            tupple: 
                (損失, 正解率)
        """
        correct = 0
        length = 0
        loss = 0
        self.model.eval()
        for data in dataloader:
                subwords = pad_sequence(data["input_ids"], batch_first=True, padding_value=self.train_data.pad_token_id).to(device)
                word_ids = pad_sequence([torch.LongTensor(js.word_ids()) for js in data["subwords"]], batch_first=True, padding_value=-1).to(device)
                label_ids = pad_sequence(data["label_ids"], batch_first=True, padding_value=1).to(device)
                pos_ids = pad_sequence(data["pos_ids"], batch_first=True, padding_value=1).to(device) if "pos_ids" in data else None
                mask = label_ids.ne(1)

                out = self.model(subwords, word_ids, pos_ids)
                loss += self.model.loss(out, label_ids, mask).detach().cpu().item()
                pred = torch.argmax(out, dim=-1)
                try:
                    correct += ((pred == label_ids) & mask).sum().detach().cpu().item()
                    length += (mask).sum().detach().cpu().item()
                except:
                    logger.info(f"evaluation skipped: {data['sentence']}")
        logger.info(f"accuracy: {correct/length*100}, loss: {loss}")
        self.model.train(True)
        return loss, correct/length
    
    def save(self, path):
        """モデルの保存

        Args:
            path (str): 保存先パス
        """
        model = self.model
        if hasattr(model, 'module'):
            model = self.model.module
        state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        torch.save(state_dict, path)

from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer as TTrainer, AutoModelForSeq2SeqLM


@Trainer.register("lemma-decoder")
class LemmaDeocderTrainer(Trainer):
    """長単位語彙素の推定モデル
    Registrableで呼び出す際の名前は "lemma-decoder"

    Args:
        train_files (Union[str, List[str]]): 
            学習セットのファイルパス（複数可）学習には :py:class:`~monaka.dataset.LemmaJsonDataset` 準拠のJSON-Lデータである必要がある
        dev_files (Union[str, List[str]]):
            検証セットのファイルパス（複数可） :py:class:`~monaka.dataset.LemmaJsonDataset` 準拠のJSON-Lデータである必要がある
        test_files (Optional[Union[str, List[str]]]): 
            評価セットのファイルパス（複数可） :py:class:`~monaka.dataset.LemmaJsonDataset` 準拠のJSON-Lデータである必要がある
        dataeset_options (Dict): 
            :py:class:`~monaka.dataset.LemmaJsonDataset` に受け渡す設定データ
        model_name (str):
            用いるモデル名。:py:class:`~monaka.model.LUWParserModel` の派生クラスであり、:py:class:`~Registrable`の機能を使って、クラス定義時に付与されたモデル名を指定することができる。例： "WordTagging"
        model_config (Dict): 
            モデルに与える設定データ
        batch_size (int, optional): 
            バッチサイズ. Defaults to 8.
        epochs (int, optional): 
            エポック数. Defaults to 1.
        steps (int, optional): _description_. 
            学習ステップ数。 epochか、step数を指定する。 Defaults to 1.
        lr (float, optional): 
            学習率. Defaults to 2e-5.
        decay (float, optional): 
            学習減衰率. Defaults to .75.
        evaluate_step (int, optional): 
            評価ステップ数。この数で割り切れる学習ステップ数時に評価を実行する. Defaults to 20.
        verbose (bool, optional): 
            デバッグ用の情報等を多く出力するようにするオプション. Defaults to True.
        seed (int, optional): 
            乱数シード値. Defaults to 419.
        output_dir (str, optional): 
            モデルの出力フォルダ. Defaults to "".
    """

    def __init__(self,
            train_files: Union[str, List[str]],
            dev_files: Union[str, List[str]],
            test_files: Optional[Union[str, List[str]]],
            dataset_options: Dict,
            model_name: str,
            model_config: Dict,
            batch_size: int=8,
            epochs: int=1,
            steps: int=1,
            lr: float=2e-5,
            decay: float=.75,
            evaluate_step:int =20,
            verbose: bool=True,
            seed: int = 419,
            output_dir: str="",
            **kwargs):
        
        global logger
        os.makedirs(output_dir, exist_ok=True)

        logger = get_logger(f"monaka.trainer.lemma.{output_dir.replace('/', '.')}")
        init_logger(logger, handlers=[logging.StreamHandler(), logging.FileHandler(f"{output_dir}/train.lemma.log", 'w')], verbose=verbose)
        self.output_dir = output_dir
        self.training_config = Seq2SeqTrainingArguments(
            output_dir = output_dir,
            num_train_epochs = epochs, 
            max_steps = steps,
            evaluation_strategy="steps",
            per_device_train_batch_size = batch_size,
            per_device_eval_batch_size = batch_size,
            eval_accumulation_steps = 100,
            learning_rate = lr,
            weight_decay = decay,
            save_steps = evaluate_step,
            eval_steps=evaluate_step,
            logging_dir=output_dir,
            seed=seed,
            predict_with_generate=True,
            do_eval = True
        )
        logger.info(train_files)
        logger.info(dev_files)
        self.dataset_options = dataset_options
        self.train_dataset = LemmaJsonDataset(train_files, **dataset_options)
        self.dev_dataset = LemmaJsonDataset(dev_files, **dataset_options)
        self.test_dataset = None
        if test_files:
            self.test_dataset = LemmaJsonDataset(test_files, **dataset_options)

        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.trainer = TTrainer(self.model, args=self.training_config, train_dataset=self.train_dataset, eval_dataset=self.dev_dataset, compute_metrics=self.compute_metrics)
        self.tokenizer = self.train_dataset.tokenizer

    def compute_metrics(self, eval_preds):
        """評価実行

        Args:
            eval_preds (Tupple[~torch.Tensor, ~torch.Tensor]): 
            (生成モデルからの出力列, 正解ラベル)

        Returns:
            Dict: 正解率(accuracy), 予測値(preds), 正解(labels)を持つ辞書型データ
        """
        preds, labels = eval_preds
        #preds_ = np.argmax(preds)
        preds_ = [np.argmax(prd, axis=-1) for prd in preds]
        logger.info(len(preds_))
        logger.info(preds_[0].shape)

        # decode preds and labels
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_preds = self.tokenizer.batch_decode(preds_, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        correct = [1 for p,l in zip(decoded_preds, decoded_labels) if p.strip() == l.strip()]
        return {"accuracy": len(correct) / len(decoded_preds), "preds": decoded_preds, "labels": decoded_labels}
    
    def train(self, device, local_rank):
        """学習実行

        Args:
            device (int, optional):
                学習に使うGPUなのだが、HuggingFaceのSeq2SeqTrainerを使うため、ここから指定できない。他との互換性のため. Defaults to -1.
            local_rank (int, optional):
                複数GPU用のオプションだが現状は動作しない. Defaults to -1.
        """
        self.trainer.train()
        self.trainer.save_model(os.path.join(self.output_dir, "last-checkpoint"))
        if self.test_dataset:
            metrics = self.trainer.evaluate(self.test_dataset)
            logger.info(metrics)
            with open(os.path.join(self.output_dir, "predict.json"), 'w') as f:
                json.dump(metrics, f, indent=True, ensure_ascii=False)

    @torch.no_grad()
    def evaluate(self, dataloader, device):
        """評価実行
        他との互換性のため記述しているが利用しない
        """
        pass