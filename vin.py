from freqtrade.strategy.interface import IStrategy
from freqtrade.exchange import timeframe_to_prev_date
from freqtrade.persistence import Trade
import logging
import numpy as np
from pandas import DataFrame, Series
from functools import reduce
from datetime import datetime, timedelta
import locale
locale.setlocale(category=locale.LC_ALL, locale='')
log = logging.getLogger(__name__)

class ViN(IStrategy):
    INTERFACE_VERSION = 2

    min_candle_vol: int = 0
    custom_buy_info = {}
    stoploss_count: int = 0
    sideways_candles: int = 80

    minimal_roi = {"0": 100}
    stoploss = -0.99
    stoploss_on_exchange = False
    trailing_stop = False
    use_custom_stoploss = False
    timeframe = '5m'
    process_only_new_candles = True
    use_sell_signal = True
    sell_profit_only = False
    startup_candle_count: int = 72

    def populate_indicators_buy(self, df: DataFrame, metadata: dict) -> DataFrame:
        return df

    def populate_indicators_sell(self, df: DataFrame, metadata: dict) -> DataFrame:
        return df

    def populate_indicators(self, df: DataFrame, metadata: dict) -> DataFrame:
        df['green'] = (df['close'] - df['open']).ge(0)
        df['bodysize'] = (df['close'] / df['open']).where(df['green'], df['open'] / df['close'])
        df['uppertail'] = (df['high'] / df['close']).where(df['green'], df['high'] / df['open'])
        df['lowertail'] = (df['open'] / df['low']).where(df['green'], df['close'] / df['low'])
        hi_adj = df['close'].where(df['green'], df['open']) + (df['high'] - df['close']).where(df['green'], (df['high'] - df['open'])) / df['bodysize']
        lo_adj = df['open'].where(df['green'], df['close']) - (df['open'] - df['low']).where(df['green'], (df['close'] - df['low'])) / df['bodysize']
        df['hlc3_adj'] = (hi_adj + lo_adj + df['close']) / 3
        df['lc2_adj'] = (lo_adj + df['close']) / 2
        df['hc2_adj'] = (hi_adj + df['close']) / 2
        df[f"candle_count_{self.startup_candle_count}"] = df['volume'].rolling(window=self.startup_candle_count, min_periods=self.startup_candle_count).count()
        df_closechange = df['close'] - df['close'].shift(1)
        s = (1, 2, 3)
        for i in s:
            df['updown'] = np.where(df_closechange.rolling(window=i, min_periods=1).sum().gt(0), 1, np.where(df_closechange.rolling(window=i, min_periods=1).sum().lt(0), -1, 0))
            df[f"streak_{i}"] = df['updown'].groupby((df['updown'].ne(df['updown'].shift(1))).cumsum()).cumsum()
        df['streak_s_min'] = df[[f"streak_{i}" for i in s]].min(axis=1)
        df['streak_s_min_change'] = df['close'] / df['close'].to_numpy()[df.index.to_numpy() + np.where(df['streak_s_min'].lt(0), df['streak_s_min'].to_numpy(), 0)]
        df['streak_s_max'] = df[[f"streak_{i}" for i in s]].max(axis=1)
        df['streak_s_max_change'] = df['close'] / df['close'].to_numpy()[df.index.to_numpy() - np.where(df['streak_s_max'].gt(0), df['streak_s_max'].to_numpy(), 0)]
        df.drop(columns=[f"streak_{i}" for i in s], inplace=True)
        df = self.populate_indicators_buy(df, metadata)
        df = self.populate_indicators_sell(df, metadata)
        return df

    def populate_buy_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df.loc[:, 'buy'] = False
        self.fill_custom_info(df, metadata)
        return df

    def populate_sell_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df.loc[:, 'sell'] = False
        return df

    def custom_sell(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float, current_profit: float, **kwargs):
            return None

    def bot_loop_start(self, **kwargs) -> None:
        self.min_candle_vol = self.config['stake_amount']
        return None

def vwrs(df: DataFrame, length: int) -> Series:
    mf: Series = df['hlc3_adj'] * df['volume']
    mfp = mf.where(df['hlc3_adj'].pct_change().gt(0), 0).rolling(window=length, min_periods=length).sum()
    mfn = mf.where(df['hlc3_adj'].pct_change().lt(0), 0).rolling(window=length, min_periods=length).sum()
    return 100 * (mfp / (mfp + mfn))


class ViNBuyPct(ViN):
    lookback_range = range(12, 33)
    def populate_indicators_buy(self, df: DataFrame, metadata: dict) -> DataFrame:
        df_closechange = df['close'] - df['close'].shift(1)
        i = 16
        df['updown'] = np.where(df_closechange.rolling(window=i, min_periods=1).sum().gt(0), 1, np.where(df_closechange.rolling(window=i, min_periods=1).sum().lt(0), -1, 0))
        df[f"streak_b"] = df['updown'].groupby((df['updown'].ne(df['updown'].shift(1))).cumsum()).cumsum()
        df.drop(columns=['updown'], inplace=True)
        ef = df['close'].reset_index()
        for i in self.lookback_range:
            j = i * 2
            df[f"pctchange_{i}"] = df['close'].pct_change(periods=i)
            # df[f"pctchange_{i}"] = (df['close'] - df['close'].shift(1)).rolling(window=i, min_periods=i).sum()
            pctchange_mean = df[f"pctchange_{i}"].rolling(window=i, min_periods=i).mean()
            pctchange_std = df[f"pctchange_{i}"].rolling(window=i, min_periods=i).std()
            df[f"bb_pctchange_{i}_up"] = pctchange_mean + 2 * pctchange_std
            df[f"bb_pctchange_{i}_lo"] = pctchange_mean - 2 * pctchange_std
            df[f"vwrs_{i}"] = vwrs(df, length=i)
            df[f"close_corr_{i}"] = ef['index'].rolling(window=i, min_periods=i).corr(ef['close'], method='spearman')
            df[f"close_corr_{j}"] = ef['index'].rolling(window=j, min_periods=j).corr(ef['close'], method='spearman')
            df = df.copy()
        return df

    def populate_buy_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df.loc[:, 'buy_tag'] = ''
        for i in self.lookback_range:
            j = i * 2
            buy_conditions = [
                df[f"candle_count_{self.startup_candle_count}"].ge(self.startup_candle_count),
                df['volume'].ge(self.min_candle_vol * 2),
                df['streak_s_min'].le(-1),
                df['streak_s_min_change'].le(0.98),
                df['streak_s_min'].ge(df['streak_b']),
                (df[f"pctchange_{i}"] / df[f"bb_pctchange_{i}_lo"]).between(1.01, 1.20),
                (df[f"bb_pctchange_{i}_up"] - df[f"bb_pctchange_{i}_lo"]).ge(0.03),
                (df['lc2_adj'] / df['close']).le(0.994),
                df[f"vwrs_{i}"].le(i + 4),
                df[f"close_corr_{j}"].gt(-0.80),
                df[f"close_corr_{j}"].gt(df[f"close_corr_{i}"])
            ]
            buy = reduce(lambda x, y: x & y, buy_conditions)
            df.loc[buy, 'buy_tag'] += f"{i} "
        df.loc[:, 'buy'] = df['buy_tag'].ne('') & df['buy_tag'].str.len().lt(42)
        df.loc[:, 'buy_tag'] = df['buy_tag'].str.strip() #(df['buy_tag'].str.len() // 3).where(df['buy_tag'].ne(''), '')
        return df

class ViNSellCorrV1(ViN):
    def custom_sell(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        df: DataFrame = df
        trade_open_date = timeframe_to_prev_date(self.timeframe, trade.open_date_utc)
        # df_trade: DataFrame = df.loc[df['date'].ge(trade_open_date)]
        # trade_len = len(df_trade)
        df_trade: DataFrame = df.loc[df['date'].ge(trade_open_date - timedelta(minutes=5))]
        trade_len = len(df_trade) - 1
        candle_1: Series = df_trade.iloc[-1]
        current_profit = (candle_1['close'] - trade.open_rate) / trade.open_rate
        buy_tag = trade.buy_tag
        if trade_len <= 2:
            if current_profit < -0.08:
                log.info(f"custom_sell: immediate stoploss for pair {pair} with loss {current_profit:.2n} and trade len {trade_len} on candle {candle_1['date']}.")
                return f"immediate stoploss {buy_tag} ({len(buy_tag) // 3 + 1})"
            else:
                return None
        if trade_len < self.sideways_candles and -0.015 < current_profit < 0.015:
            return None
        # do not sell if price is more above lowest price compared with below highest price
        i = min(trade_len, self.sideways_candles) # maybe change to i = trade_len
        j = i // 2
        ef = df_trade[['close', 'hlc3_adj', 'volume']].reset_index()
        # ef = df.reset_index() # maybe change to df_trade, and use only two columns?
        ef['vwrs_sell'] = vwrs(ef, length=i)
        ef['vwrs_corr'] = ef['index'].rolling(window=i, min_periods=i).corr(ef['vwrs_sell'], method='spearman')
        ef['close_corr_i'] = ef['index'].rolling(window=i, min_periods=i).corr(ef['close'], method='spearman')
        ef['close_corr_j'] = ef['index'].rolling(window=j, min_periods=j).corr(ef['close'], method='spearman')
        vwrs_corr_diff = ef['vwrs_corr'].iat[-1] - ef['vwrs_corr'].iat[-2]
        close_corr_i_diff = ef['close_corr_i'].iat[-1] - ef['close_corr_i'].iat[-2]
        close_corr_ij_diff = ef['close_corr_i'].iat[-1] - ef['close_corr_j'].iat[-1]
        if current_profit <= -0.04:
            offset = 0.04 + (current_profit / 8 - i / (self.sideways_candles * 8))
        else:
            offset = 0.04 + (current_profit / 18 - i / (self.sideways_candles * 18))
        if vwrs_corr_diff < -offset and close_corr_i_diff > offset and close_corr_ij_diff > 0:
            log.info(f"custom_sell: corr sell for pair {pair} with profit/loss {current_profit:.2n} and trade len {trade_len} on candle {candle_1['date']}.")
            return f"corr sell {buy_tag} ({len(buy_tag) // 3 + 1})"
        else:
            return None

class ViNSellCorr(ViN):
    def custom_sell(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        df: DataFrame = df
        trade_open_date = timeframe_to_prev_date(self.timeframe, trade.open_date_utc)
        df_trade: DataFrame = df.loc[df['date'].ge(trade_open_date - timedelta(minutes=5))]
        trade_len = len(df_trade) - 1
        candle_1: Series = df_trade.iloc[-1]
        current_profit = (candle_1['close'] - trade.open_rate) / trade.open_rate
        buy_tag = trade.buy_tag
        if trade_len <= 2:
            if current_profit < -0.08:
                log.info(f"custom_sell: immediate stoploss for pair {pair} with loss {current_profit:.2n} and trade len {trade_len} on candle {candle_1['date']}.")
                return f"immediate stoploss {buy_tag} ({len(buy_tag) // 3 + 1})"
            else:
                return None
        if trade_len < self.sideways_candles and -0.015 < current_profit < 0.015:
            return None
        i = trade_len
        j = i // 2
        ef = df_trade[['close', 'hlc3_adj', 'volume']].reset_index()
        vwrs_sell = vwrs(ef, length=i)
        close_corr_i = ef['index'].rolling(window=i, min_periods=i).corr(ef['close'], method='spearman')
        close_corr_j = ef['index'].rolling(window=j, min_periods=j).corr(ef['close'], method='spearman')
        if current_profit <= min(-0.015, -0.08 + i / 400) and vwrs_sell.iat[-1] > max(18, 54 - i / 4) and candle_1['streak_s_max'] <= -1:
            log.info(f"custom_sell: corr stop for pair {pair} with loss {current_profit:.2n} and trade len {trade_len} on candle {candle_1['date']}.")
            return f"corr stop {buy_tag} ({len(buy_tag) // 3 + 1})"
        if current_profit >= -0.015 and vwrs_sell.iat[-1] > 100 - i / 8 and close_corr_i.iat[-1] < 0.9 and close_corr_i.iat[-1] > close_corr_i.iat[-2] and close_corr_j.iat[-1] < close_corr_i.iat[-1]:
            log.info(f"custom_sell: corr sell for pair {pair} with profit/loss {current_profit:.2n} and trade len {trade_len} on candle {candle_1['date']}.")
            return f"corr sell {buy_tag} ({len(buy_tag) // 3 + 1})"
        else:
            return None

class ViNSellPct(ViN):
    def custom_sell(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        df: DataFrame = df
        trade_open_date = timeframe_to_prev_date(self.timeframe, trade.open_date_utc)
        df_trade: DataFrame = df.loc[df['date'].ge(trade_open_date - timedelta(minutes=5))]
        trade_len = len(df_trade) - 1
        candle_1: Series = df_trade.iloc[-1]
        current_profit = (candle_1['close'] - trade.open_rate) / trade.open_rate
        buy_tag = trade.buy_tag
        if trade_len <= 2:
            if current_profit < -0.08:
                log.info(f"custom_sell: immediate stoploss for pair {pair} with loss {current_profit:.2n} and trade len {trade_len} on candle {candle_1['date']}.")
                return f"immediate stoploss {buy_tag} ({len(buy_tag) // 3 + 1})"
            else:
                return None
        if trade_len < self.sideways_candles and -0.015 < current_profit < 0.015:
            return None
        i = trade_len
        j = i // 2
        ef = df_trade[['close', 'hlc3_adj', 'volume']].reset_index()
        vwrs_sell = vwrs(ef, length=i)
        close_corr_i = ef['index'].rolling(window=i, min_periods=i).corr(ef['close'], method='spearman')
        close_corr_j = ef['index'].rolling(window=j, min_periods=j).corr(ef['close'], method='spearman')
        if current_profit <= min(0.015, -0.08 + i / 400) and vwrs_sell.iat[-1] > max(18, 54 - i / 4) and candle_1['streak_s_max'] <= -1:
            log.info(f"custom_sell: corr stop for pair {pair} with loss {current_profit:.2n} and trade len {trade_len} on candle {candle_1['date']}.")
            return f"corr stop {buy_tag} ({len(buy_tag) // 3 + 1})"
        if current_profit >= 0.015 and vwrs_sell.iat[-1] > 100 - i / 10 and close_corr_i.iat[-1] < 0.8 and close_corr_i.iat[-1] > close_corr_i.iat[-2] and close_corr_j.iat[-1] < close_corr_i.iat[-1]:
            log.info(f"custom_sell: corr sell for pair {pair} with profit {current_profit:.2n} and trade len {trade_len} on candle {candle_1['date']}.")
            return f"corr sell {buy_tag} ({len(buy_tag) // 3 + 1})"
        df_pctchange = ef['close'].pct_change()
        df_pctchange_mean = df_pctchange.rolling(window=i, min_periods=i).mean()
        df_pctchange_std = df_pctchange.rolling(window=i, min_periods=i).std()
        pctchange = df_pctchange.iat[-1]
        bb_pctchange_up = df_pctchange_mean.iat[-1] + 2 * df_pctchange_std.iat[-1]
        bb_pctchange_lo = df_pctchange_mean.iat[-1] - 2 * df_pctchange_std.iat[-1]
        i = candle_1['streak_s_max']
        if i >= 2:
            j = i // 2
            vwrs_sell = vwrs(ef, length=i)
            close_corr_i = ef['index'].rolling(window=i, min_periods=i).corr(ef['close'], method='spearman')
            close_corr_j = ef['index'].rolling(window=j, min_periods=j).corr(ef['close'], method='spearman')
            if (
                candle_1['volume'] >= self.min_candle_vol
                and candle_1['streak_s_max_change'] >= 1.01
                and pctchange / bb_pctchange_up > 1.005
                and bb_pctchange_up - bb_pctchange_lo >= 0.02
                and candle_1['hc2_adj'] / candle_1['close'] >= 1.002
                and vwrs_sell.iat[-1] > 72 + i
                and close_corr_i.iat[-1] < 0.95
                and close_corr_j.iat[-1] < close_corr_i.iat[-1]
            ):
                log.info(f"custom_sell: pct up sell for pair {pair} with profit/loss {current_profit:.2n} and trade len {trade_len} on candle {candle_1['date']}.")
                return f"pct up sell {buy_tag} ({len(buy_tag) // 3 + 1})"
        return None

class ViNBuyPctSellCorrV1(ViNBuyPct, ViNSellCorrV1):
    pass

class ViNBuyPctSellCorr(ViNBuyPct, ViNSellCorr):
    pass

class ViNBuyPctSellPct(ViNBuyPct, ViNSellPct):
    pass

class ViresInNumeris(ViNBuyPct, ViNSellCorrV1):
    pass
