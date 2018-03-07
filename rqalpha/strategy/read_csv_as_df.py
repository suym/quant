from rqalpha.api import *


def read_csv_as_df(csv_path):
    # 通过 pandas 读取 csv 文件，并生成 DataFrame
    import pandas as pd
    data = pd.read_csv(csv_path)
    return data


def init(context):
    import os
    # 获取当前运行策略的文件路径
    strategy_file_path = context.config.base.strategy_file
    # 根据当前策略的文件路径寻找到相对路径为 "../../notebook/fd.csv" 的 csv 文件
    csv_path = os.path.join(os.path.dirname(strategy_file_path), "../../notebook/fd.csv")
    # 读取 csv 文件并生成 df
    IF1706_df = read_csv_as_df(csv_path)
    # 传入 context 中
    context.IF1706_df = IF1706_df


def handle_bar(context,bar):
    price = context.IF1706_df.loc[:0,['Close']]
    logger.info("每一个Bar执行")
    logger.info(price)
