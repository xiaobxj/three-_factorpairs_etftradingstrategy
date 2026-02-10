from jqdata import *
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
from six import BytesIO


def initialize(context):
    # 1. 基础设置
    set_benchmark('000300.XSHG')
    set_option('use_real_price', True)
    set_order_cost(OrderCost(close_tax=0.001, open_commission=0.0003, close_commission=0.0003, min_commission=5),
                   type='stock')

    # 2. 定义多因子策略池
    # 结构: '策略名': ['基准资产(0)', '互补资产(1)', '模型文件']
    g.strategies = {
        'Size': {  # 规模因子
            'assets': ['510300.XSHG', '510500.XSHG'],  # [大盘, 小盘]
            'model_file': 'xgb_model_Size.pkl',
            'model': None
        },
        'Style': {  # 风格因子
            'assets': ['510050.XSHG', '159915.XSHE'],  # [价值, 成长]
            'model_file': 'xgb_model_Style.pkl',
            'model': None
        },
        'MomRev': {  # 动量/反转因子
            'assets': ['510880.XSHG', '159901.XSHE'],  # [红利(反转), 深证100(动量)]
            'model_file': 'xgb_model_MomRev.pkl',
            'model': None
        }
    }

    # 3. 加载所有模型
    for name, strat in g.strategies.items():
        try:
            content = read_file(strat['model_file'])
            strat['model'] = pickle.load(BytesIO(content))
            log.info(f"{name} 模型加载成功")
        except:
            log.error(f"{name} 模型加载失败，请检查研究环境是否运行并保存了模型")

    # 4. 每日运行
    run_daily(trade_logic, time='09:30')


def trade_logic(context):
    # 计算资金分配: 将总资金 3 等分 (Equal Weighting across pairs)
    num_strategies = len(g.strategies)
    total_value = context.portfolio.total_value
    capital_per_strat = total_value / num_strategies

    # 设定调仓阈值 (防止微小金额触发废单)
    threshold = capital_per_strat * 0.05
    lookback = 30

    # ================== 循环执行每个因子对 ==================
    for name, strat in g.strategies.items():
        if strat['model'] is None:
            continue

        code_base = strat['assets'][0]  # 资产0 (如大盘/红利)
        code_comp = strat['assets'][1]  # 资产1 (如小盘/深证100)

        # 1. 获取数据 & 计算特征 (必须与训练逻辑完全一致)
        # 获取收盘价
        p_base = get_price(code_base, count=lookback, end_date=context.previous_date, fields=['close'], panel=False)[
            'close']
        p_comp = get_price(code_comp, count=lookback, end_date=context.previous_date, fields=['close'], panel=False)[
            'close']

        # 计算对数收益
        ret_base = np.log(p_base / p_base.shift(1))
        ret_comp = np.log(p_comp / p_comp.shift(1))

        # 特征工程 (Spread = 资产1 - 资产0)
        spread = ret_comp - ret_base
        feat_mom = spread.rolling(20).mean().iloc[-1]
        feat_vol = (ret_comp.rolling(20).std() - ret_base.rolling(20).std()).iloc[-1]

        if np.isnan(feat_mom) or np.isnan(feat_vol):
            continue

        # 2. 预测
        X = pd.DataFrame([[feat_mom, feat_vol]], columns=['feat_mom_20', 'feat_vol_diff'])
        pred = strat['model'].predict(X)[0]

        # 3. 交易执行
        # pred==1 代表看好资产1(互补), pred==0 代表看好资产0(基准)
        target_code = code_comp if pred == 1 else code_base
        other_code = code_base if pred == 1 else code_comp

        # 获取当前持仓市值
        val_target = context.portfolio.positions[target_code].value
        val_other = context.portfolio.positions[other_code].value

        # === 核心交易逻辑 ===

        # A. 发生切换 (持有非目标资产) -> 坚决调仓
        if val_other > 0:
            order_target_value(other_code, 0)
            order_target_value(target_code, capital_per_strat)
            log.info(f"[{name}] 信号切换: 卖出 {other_code}, 买入 {target_code}")

        # B. 信号未变 (继续持有目标资产) -> 检查再平衡阈值
        else:
            # 目标是持有 capital_per_strat 这么多金额
            diff = capital_per_strat - val_target

            # 只有偏差超过阈值，且金额足够买1手时，才进行微调
            if abs(diff) > threshold and abs(diff) > 500:
                order_target_value(target_code, capital_per_strat)
                # 记录日志方便调试 (可选)
                # log.info(f"[{name}] 仓位再平衡: {target_code}, 调整 {diff:.2f}")