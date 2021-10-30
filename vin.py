from freqtrade.strategy.interface import IStrategy
from freqtrade.exchange import timeframe_to_prev_date
from freqtrade.persistence import Trade
import logging
import numpy as np
from pandas import DataFrame, Series
from functools import reduce
from datetime import datetime
import locale
locale.setlocale(category=locale.LC_ALL, locale='')
log = logging.getLogger(__name__)

class ViN(IStrategy):
    INTERFACE_VERSION = 2

    min_candle_vol: int = 0
    custom_buy_info = {}
    stoploss_count: int = 0
    sideways_candles: int = 60

    minimal_roi = {"0": 100}
    stoploss = -0.99
    stoploss_on_exchange = False
    trailing_stop = False
    use_custom_stoploss = False
    timeframe = '5m'
    process_only_new_candles = True
    use_sell_signal = True
    sell_profit_only = False
    startup_candle_count: int = sideways_candles

    def populate_indicators_buy(self, df: DataFrame, metadata: dict) -> DataFrame:
        return df

    def populate_indicators_sell(self, df: DataFrame, metadata: dict) -> DataFrame:
        return df

    def populate_indicators(self, df: DataFrame, metadata: dict) -> DataFrame:
        if len(df) < self.startup_candle_count:
            return df

        df['green'] = (df['close'] - df['open']).ge(0)
        df['bodysize'] = (df['close'] / df['open']).where(df['green'], df['open'] / df['close'])
        df['uppertail'] = (df['high'] / df['close']).where(df['green'], df['high'] / df['open'])
        df['lowertail'] = (df['open'] / df['low']).where(df['green'], df['close'] / df['low'])
        hi_adj = df['close'].where(df['green'], df['open']) + (df['high'] - df['close']).where(df['green'], (df['high'] - df['open'])) / df['bodysize']
        lo_adj = df['open'].where(df['green'], df['close']) - (df['open'] - df['low']).where(df['green'], (df['close'] - df['low'])) / df['bodysize']
        df['hlc3_adj'] = (hi_adj + lo_adj + df['close']) / 3
        df['lc2_adj'] = (lo_adj + df['close']) / 2

        df_closechange = df['close'] - df['close'].shift(1)
        for i in (1, 2, 3):
            df['updown'] = np.where(df_closechange.rolling(window=i, min_periods=1).sum().gt(0), 1, np.where(df_closechange.rolling(window=i, min_periods=1).sum().lt(0), -1, 0))
            df[f"streak_{i}"] = df['updown'].groupby((df['updown'].ne(df['updown'].shift(1))).cumsum()).cumsum()
        df['streak_min'] = df[['streak_1', 'streak_2', 'streak_3']].min(axis=1)
        df['streak_min_change'] = df['close'] / df['close'].to_numpy()[df.index.to_numpy() + np.where(df['streak_min'].lt(0), df['streak_min'].to_numpy(), 0)]
        df['streak_max'] = df[['streak_1', 'streak_2', 'streak_3']].max(axis=1)
        df['streak_max_change'] = df['close'] / df['close'].to_numpy()[df.index.to_numpy() - np.where(df['streak_max'].gt(0), df['streak_max'].to_numpy(), 0)]
        df.drop(columns=['updown', 'streak_1', 'streak_2', 'streak_3'], inplace=True)

        df[f"candle_count_{self.startup_candle_count}"] = df['volume'].rolling(window=self.startup_candle_count, min_periods=self.startup_candle_count).count()

        df = self.populate_indicators_sell(df, metadata)
        df = self.populate_indicators_buy(df, metadata)
        return df

    def fill_custom_info(self, df: DataFrame, metadata: dict):
        df_buy: DataFrame = df.loc[df.loc[:, 'buy'], ['date', 'buy_tag', 'buy']]
        for index, row in df_buy.iterrows():
            buy_date = row['date']
            try:
                self.custom_buy_info[buy_date]['buy_tags'] += row['buy_tag']
                self.custom_buy_info[buy_date]['buy_signals'] += 1
            except:
                self.custom_buy_info[buy_date] = {}
                self.custom_buy_info[buy_date]['buy_tags'] = row['buy_tag']
                self.custom_buy_info[buy_date]['buy_signals'] = 1
            self.custom_buy_info[buy_date][metadata['pair']] = row['buy_tag']

    def populate_buy_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df.loc[:, 'buy'] = False
        # self.fill_custom_info(df, metadata)
        return df

    def populate_sell_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df.loc[:, 'sell'] = False
        return df

    def custom_sell(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float, current_profit: float, **kwargs):
            return None

    def bot_loop_start(self, **kwargs) -> None:
        self.min_candle_vol = self.config['stake_amount']
        return None

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                            time_in_force: str, current_time: datetime, **kwargs) -> bool:
        # df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        # candle_1: Series = df.iloc[-1]
        # buy_candle_date = candle_1['date']
        # try:
        #     buy_tags = self.custom_buy_info[buy_candle_date][pair]
        #     pairs = len(self.dp.current_whitelist())
        #     max_concurrent_buy_signals = int(pairs * 0.08)
        #     if  max_concurrent_buy_signals > 0:
        #         buy_info = self.custom_buy_info[buy_candle_date]
        #         buy_signal_count = buy_info['buy_signals']
        #         if buy_signal_count > max_concurrent_buy_signals:
        #             log.info(f"confirm_trade_entry: Buy for pair {pair} with buy tag {buy_tags} on candle {buy_candle_date} is cancelled. There are {buy_signal_count} concurrent buy signals (max = {max_concurrent_buy_signals}).")
        #             return False
        # except:
        #     log.warning(f"confirm_trade_entry: No buy info for pair {pair} on candle {buy_candle_date}.")
        return True

def vwrs(df: DataFrame, length: int) -> Series:
    mf: Series = df['hlc3_adj'] * df['volume']
    mfp = mf.where(df['hlc3_adj'].pct_change().gt(0), 0).rolling(window=length, min_periods=length).sum()
    mfn = mf.where(df['hlc3_adj'].pct_change().lt(0), 0).rolling(window=length, min_periods=length).sum()
    return 100 * (mfp / (mfp + mfn))


class ViNBuyPct(ViN):
    lookback_range = range(11, 34)

    def populate_indicators_buy(self, df: DataFrame, metadata: dict) -> DataFrame:
        ef = df['close'].reset_index()
        for i in self.lookback_range:
            j = i * 2
            pctchange = df['close'].pct_change(periods=i)
            pctchange_mean = pctchange.rolling(window=i, min_periods=i).mean()
            pctchange_std = pctchange.rolling(window=i, min_periods=i).std()
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
                df['volume'].ge(self.min_candle_vol),
                # df['streak_min'].le(-1),
                ~df['green'],
                df['streak_min_change'].le(0.99),
                (df['close'].pct_change(periods=i) / df[f"bb_pctchange_{i}_lo"]).between(1.01, 1.49),
                (df[f"bb_pctchange_{i}_up"] - df[f"bb_pctchange_{i}_lo"]).ge(0.03),
                (df['lc2_adj'] / df['close']).le(0.992),
                df[f"vwrs_{i}"].le(i + 4),
                df[f"close_corr_{j}"].gt(df[f"close_corr_{i}"])
            ]
            buy = reduce(lambda x, y: x & y, buy_conditions)
            df.loc[buy, 'buy_tag'] += f"{i} "

        df.loc[:, 'buy'] = df['buy_tag'].ne('') & df['buy_tag'].str.len().lt(42)
        df.loc[:, 'buy_tag'] = (df['buy_tag'].str.len() // 3).where(df['buy_tag'].ne(''), '')

        # self.fill_custom_info(df, metadata)
        return df

class ViNSellCorr(ViN):
    def custom_sell(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        df: DataFrame = df
        trade_open_date = timeframe_to_prev_date(self.timeframe, trade.open_date_utc)
        df_trade: DataFrame = df.loc[df['date'].ge(trade_open_date)]
        trade_len = len(df_trade)
        candle_1: Series = df_trade.iloc[-1]
        current_profit = (candle_1['close'] - trade.open_rate) / trade.open_rate
        if trade_len <= 1 or (trade_len < self.sideways_candles and -0.015 < current_profit < 0.015):
            return None
        buy_tag = trade.buy_tag
        i = min(trade_len, self.sideways_candles)
        ef = df.reset_index()
        ef['vwrs_sell'] = vwrs(ef, length=i)
        ef['vwrs_corr'] = ef['index'].rolling(window=i, min_periods=i).corr(ef['vwrs_sell'], method='spearman')
        ef['close_corr'] = ef['index'].rolling(window=i, min_periods=i).corr(ef['close'], method='spearman')
        vwrs_corr_diff = ef['vwrs_corr'].iat[-1] - ef['vwrs_corr'].iat[-2]
        close_corr_diff = ef['close_corr'].iat[-1] - ef['close_corr'].iat[-2]
        offset = 0.04 + (current_profit / 18 - i / (self.sideways_candles * 18))
        if vwrs_corr_diff < -offset and close_corr_diff > offset and abs(vwrs_corr_diff) > abs(close_corr_diff):
            log.info(f"custom_sell: corr sell for pair {pair} with profit/loss {current_profit:.2n}, offset {offset:.3n} and trade len {trade_len} on candle {candle_1['date']}.")
            return f"corr sell ({buy_tag})"
        else:
            return None

class ViNSellStreaks(ViN):
    lookback_candles: int = 15
    sell_indicator_range = range(2, lookback_candles + 1)

    def populate_indicators_sell(self, df: DataFrame, metadata: dict) -> DataFrame:
        ef = df['close'].reset_index()
        for i in self.sell_indicator_range:
            df[f"volume_{i}"] = df['volume'].rolling(window=i, min_periods=i).sum()
            df[f"close_corr_{i}"] = ef['index'].rolling(window=i, min_periods=i).corr(ef['close'], method='spearman')

        return df.copy()

    def populate_sell_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df.loc[:, 'sell_tag'] = ''
        sell_conditions = [
            df['volume'].ge(self.min_candle_vol * 1.4),
            df['green'],
            df['streak_max'].ge(3)
        ]
        sell = reduce(lambda x, y: x & y, sell_conditions)
        df.loc[sell, 'sell_tag'] = 'pct+' + df['streak_max'].astype(str)

        time_periods = range(3, self.lookback_candles + 1)
        for i in time_periods:
            sell_conditions = [
                df['volume'].ge(self.min_candle_vol * 1.4),
                df[f"volume_{i}"].ge(self.min_candle_vol * i * 0.8),
                df['streak_max'].ne(df['streak_min']),
                df['streak_min'].ne(-1),
                df[f"close_corr_{i}"].between(0.75, 0.95),
                df[f"close_corr_{i-1}"].lt(df[f"close_corr_{i}"])
            ]
            if i == max(time_periods):
                sell_conditions.append(df['streak_max'].ge(i))
            else:
                sell_conditions.append(df['streak_max'].eq(i))
            sell = reduce(lambda x, y: x & y, sell_conditions)
            df.loc[sell, 'sell_tag'] = 'corr+' + df['streak_max'].astype(str)

        for i in time_periods:
            sell_conditions = [
                df['volume'].ge(self.min_candle_vol * 1.4),
                df[f"volume_{i}"].ge(self.min_candle_vol * i * 0.8),
                df['streak_min'].ne(-1),
                df[f"close_corr_{i}"].gt(0.75),
                df['uppertail'].ge(1.02)
            ]
            if i == max(time_periods):
                sell_conditions.append(df['streak_max'].ge(i))
            else:
                sell_conditions.append(df['streak_max'].eq(i))
            sell = reduce(lambda x, y: x & y, sell_conditions)
            df.loc[sell, 'sell_tag'] = 'tail+' + df['streak_max'].astype(str)

        for i in time_periods:
            sell_conditions = [
                df['volume'].ge(self.min_candle_vol * 1.2),
                df[f"close_corr_{i}"].lt(0),
                df[f"close_corr_{i-1}"].lt(df[f"close_corr_{i}"]),
                df['close'].pct_change().lt(0)
            ]
            if i == max(time_periods):
                sell_conditions.append(df['streak_min'].le(-i))
            else:
                sell_conditions.append(df['streak_min'].eq(-i))
            sell = reduce(lambda x, y: x & y, sell_conditions)
            df.loc[sell, 'sell_tag'] = 'corr' + df['streak_min'].astype(str)

        df.loc[:, 'stop_tag'] = ''
        for i in time_periods:
            sell_conditions = [
                df['volume'].ge(self.min_candle_vol * 1.2),
                df[f"close_corr_{i}"].le(0),
                df[f"close_corr_{i-1}"].le(df[f"close_corr_{i}"]),
                df['close'].pct_change().le(0)
            ]
            if i == min(time_periods):
                sell_conditions.append(df['streak_min'].between(-i, -2))
            if i == max(time_periods):
                sell_conditions.append(df['streak_min'].le(-i))
            else:
                sell_conditions.append(df['streak_min'].eq(-i))
            sell = reduce(lambda x, y: x & y, sell_conditions)
            df.loc[sell, 'stop_tag'] = 'stop' + df['streak_min'].astype(str)

        return df

    def custom_sell(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        trade_open_date = timeframe_to_prev_date(self.timeframe, trade.open_date_utc)
        df_trade: DataFrame = df.loc[df['date'] >= trade_open_date]
        if hasattr(trade, 'buy_tag') and trade.buy_tag is not None:
            buy_tag = trade.buy_tag
        trade_len = len(df_trade)
        candle_1: Series = df_trade.iloc[-1].copy()
        sell_tag: str = candle_1['sell_tag']
        stop_tag: str = candle_1['stop_tag']
        if len(df_trade) < 2 or (stop_tag == '' and sell_tag == ''):
            return None

        current_profit = (candle_1['close'] - trade.open_rate) / trade.open_rate
        i = min(trade_len, self.sideways_candles)
        sell_reason = None
        if 'stop' in stop_tag and current_profit < -0.04:
            streaks = int("".join(filter(str.isdigit, stop_tag)))
            if streaks <= i + 1:
                self.stoploss_count += 1
                log.info(f"custom_sell: stop # {self.stoploss_count} for pair {pair} with loss {round(current_profit, 2)} and stop_tag {stop_tag} on candle {candle_1['date']}.")
                sell_reason = f"{stop_tag} ({buy_tag})"
        if sell_tag != '':
            if 'corr' in sell_tag and trade_len >= self.sideways_candles:
                df_sw = df_trade['close'].tail(i)
                if df_sw.max() / df_sw.min() <= 1.03:
                    log.info(f"custom_sell: sideways sell for pair {pair} with profit/loss {round(current_profit, 2)} and sell_tag {sell_tag} on candle {candle_1['date']}.")
                    sell_reason = f"{sell_tag.replace('corr', 'sideways')} ({buy_tag})"
            streaks = int("".join(filter(str.isdigit, sell_tag)))
            if streaks <= i:
                mfi = vwrs(df, length=i)
                if current_profit > 0.015 and mfi.iat[-1] > 66:
                    if 'corr' in sell_tag:
                        sell_reason = f"{sell_tag} ({buy_tag})"
                        log.info(f"custom_sell: sell for pair {pair} with profit {round(current_profit, 2)} and sell_tag {sell_tag} on candle {candle_1['date']}.")
                    elif 'pct' in sell_tag and i >= 3:
                        pctchange_mean = df_trade['close'].pct_change(periods=i).rolling(window=i, min_periods=i).mean()
                        pctchange_std = df_trade['close'].pct_change(periods=i).rolling(window=i, min_periods=i).std()
                        pctchange_up = pctchange_mean + 2 * pctchange_std
                        if df_trade['close'].pct_change(i).iat[-1] / pctchange_up.iat[-1] >= 1.1:
                            sell_reason = f"{sell_tag} ({buy_tag})"
                            log.info(f"custom_sell: sell for pair {pair} with profit {round(current_profit, 2)} and sell_tag {sell_tag} on candle {candle_1['date']}.")
        candle_1['sell'] = sell_reason is not None

        return sell_reason

class ViNBuyPctSellCorr(ViNBuyPct, ViNSellCorr):
    pass

class ViNBuyPctSellStreaks(ViNBuyPct, ViNSellStreaks):
    pass

class ViresInNumeris(ViNBuyPct, ViNSellCorr):
    pass
