# -*- coding: utf-8 -*-
"""専用ロガーのセットアップ"""

import logging
import os


def get_logger(name: str) -> logging.Logger:
    """ロガーの作成

    Args:
        name (str): ロガー名

    Returns:
        logging.Logger: ロガー
    """
    return logging.getLogger(name)


def init_logger(logger: logging.Logger,
                path=None,
                mode='w',
                level=None,
                handlers=None,
                verbose=True):
    """ロガーの初期化

    Args:
        logger (logging.Logger): 
            初期化対象のロガー
        path (str, optional): 
            ファイル書き込み時のファイルパス. Defaults to None.
        mode (str, optional): 
            書き込みモード. Defaults to 'w'.
        level (int, optional): 
            ログレベル. Defaults to None.
        handlers (logging.Handler, optional): 
            ログハンドラー. Defaults to None.
        verbose (bool, optional): 
            書き込みのVerbose. Defaults to True.
    """
    level = level or logging.WARNING
    if not handlers:
        handlers = [logging.StreamHandler()]
        logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S',
                            level=level,
                            handlers=handlers)
        if path:
            if os.path.dirname(path):
                os.makedirs(os.path.dirname(path), exist_ok=True)
            handlers.append(logging.FileHandler(path, mode))
    for handler in handlers:
        handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)


logger = get_logger("monaka")
"""デフォルトロガー
"""