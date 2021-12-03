from math import isqrt
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

    write_to_csv = False
    df_csv = './user_data/df.csv'
    min_candle_vol: int = 0
    custom_buy_info = {}
    max_concurrent_buy_signals_check = True

    minimal_roi = {"0": 100}
    stoploss = -1
    stoploss_on_exchange = False
    trailing_stop = False
    use_custom_stoploss = False
    timeframe = '5m'
    process_only_new_candles = True
    use_sell_signal = True
    sell_profit_only = False
    startup_candle_count: int = 90

    @property
    def protections(self):
        return [
            {
                "method": "CooldownPeriod",
                "stop_duration_candles": 36
            }
        ]

    def populate_indicators_buy(self, df: DataFrame, metadata: dict) -> DataFrame:
        return df

    def populate_indicators_sell(self, df: DataFrame, metadata: dict) -> DataFrame:
        return df

    def populate_indicators(self, df: DataFrame, metadata: dict) -> DataFrame:
        df['green'] = (df['close'] - df['open']).ge(0)
        df['bodysize'] = (df['close'] / df['open']).where(df['green'], df['open'] / df['close'])
        hi_adj = df['close'].where(df['green'], df['open']) + (df['high'] - df['close']).where(df['green'], (df['high'] - df['open'])) / df['bodysize'].pow(0.25)
        lo_adj = df['open'].where(df['green'], df['close']) - (df['open'] - df['low']).where(df['green'], (df['close'] - df['low'])) / df['bodysize'].pow(0.25)
        df['hlc3_adj'] = (hi_adj + lo_adj + df['close']) / 3
        df['lc2_adj'] = (lo_adj + df['close']) / 2
        df['hc2_adj'] = (hi_adj + df['close']) / 2
        df[f"candle_count_{self.startup_candle_count}"] = df['volume'].rolling(window=self.startup_candle_count, min_periods=self.startup_candle_count).count()
        df_closechange = df['close'] - df['close'].shift(1)
        s = (1, 2, 3)
        for i in s:
            df['updown'] = np.where(df_closechange.rolling(window=i, min_periods=i).sum().gt(0), 1, np.where(df_closechange.rolling(window=i, min_periods=i).sum().lt(0), -1, 0))
            df[f"streak_{i}"] = df['updown'].groupby((df['updown'].ne(df['updown'].shift(1))).cumsum()).cumsum()
        df['streak_s_min'] = df[[f"streak_{i}" for i in s]].min(axis=1)
        df['streak_s_min_change'] = df['close'] / df['close'].to_numpy()[df.index.to_numpy() - df['streak_s_min'].abs().to_numpy()]
        df['streak_s_max'] = df[[f"streak_{i}" for i in s]].max(axis=1)
        df.drop(columns=[f"streak_{i}" for i in s], inplace=True)
        df_closechange = df['close'] - df['close'].shift(1)
        i = 12
        df['updown'] = np.where(df_closechange.rolling(window=i, min_periods=i).sum().gt(0), 1, np.where(df_closechange.rolling(window=i, min_periods=i).sum().lt(0), -1, 0))
        df[f"streak_h"] = df['updown'].groupby((df['updown'].ne(df['updown'].shift(1))).cumsum()).cumsum()
        df.drop(columns=['updown'], inplace=True)
        df = self.populate_indicators_buy(df, metadata)
        df = self.populate_indicators_sell(df, metadata)
        if self.config['runmode'].value not in ('live', 'dry_run') and self.write_to_csv:
            ef = df[['date', 'open', 'high', 'low', 'close', 'volume', 'bodysize', 'hlc3_adj', 'lc2_adj', 'hc2_adj', 'streak_s_min', 'streak_s_max']]
            ef['pair'] = metadata['pair']
            with open(self.df_csv, 'a') as f:
                ef.to_csv(f, sep=';', header=f.tell()==0, index=False)
        return df

    def fill_custom_buy_info(self, df:DataFrame, metadata: dict):
        df_buy: DataFrame = df.loc[df['buy'], ['date', 'buy_tag']]
        for index, row in df_buy.iterrows():
            buy_date = row['date']
            if buy_date not in self.custom_buy_info:
                self.custom_buy_info[buy_date] = {}
                self.custom_buy_info[buy_date]['buy_signals'] = 1
            else:
                self.custom_buy_info[buy_date]['buy_signals'] += 1
            self.custom_buy_info[buy_date][metadata['pair']] = row['buy_tag']
        return None

    def populate_buy_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df.loc[:, 'buy'] = False
        return df

    def populate_sell_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df.loc[:, 'sell'] = False
        return df

    def custom_sell(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float, current_profit: float, **kwargs):
            return None

    def bot_loop_start(self, **kwargs) -> None:
        self.min_candle_vol = self.config['stake_amount']
        if self.write_to_csv:
            with open(self.df_csv, 'w') as f:
                pass
        return None

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                            time_in_force: str, current_time: datetime, **kwargs) -> bool:
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        buy_candle_date = df['date'].iloc[-1]
        d = buy_candle_date.strftime('%Y-%m-%d %H:%M')
        try:
            buy_info = self.custom_buy_info[buy_candle_date]
            buy_tag = buy_info[pair]
            buy_signal_count = buy_info['buy_signals']
            if self.max_concurrent_buy_signals_check:
                pairs = len(self.dp.current_whitelist())
                max_concurrent_buy_signals = max(int(pairs * 0.08), 8)
                if buy_signal_count > max_concurrent_buy_signals:
                    log.info(f"{d} confirm_trade_entry: Cancel buy for pair {pair} with buy tag {buy_tag}. There are {buy_signal_count} concurrent buy signals (max = {max_concurrent_buy_signals}).")
                    return False
            log.info(f"{d} confirm_trade_entry: Buy for pair {pair} with buy tag {buy_tag} and {buy_signal_count} concurrent buy signals.")
        except:
            log.warning(f"{d} confirm_trade_entry: No buy info for pair {pair}.")
            return False
        return True

    def confirm_trade_exit(self, pair: str, trade: "Trade", order_type: str, amount: float,
                           rate: float, time_in_force: str, sell_reason: str, **kwargs) -> bool:
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        buy_candle_date = df['date'].iloc[-1]
        d = buy_candle_date.strftime('%Y-%m-%d %H:%M')
        try:
            buy_info = self.custom_buy_info[buy_candle_date]
            buy_signal_count = buy_info['buy_signals']
            if self.max_concurrent_buy_signals_check:
                pairs = len(self.dp.current_whitelist())
                max_concurrent_buy_signals = max(int(pairs * 0.04), 4)
                if buy_signal_count > max_concurrent_buy_signals:
                    log.info(f"{d} confirm_trade_exit: Cancel sell for pair {pair}. There are {buy_signal_count} concurrent buy signals (max = {max_concurrent_buy_signals}).")
                    return False
        except:
            return True
        return True

def vwrs(df: DataFrame, length: int) -> Series:
    mf: Series = df['hlc3_adj'] * df['volume']
    mfp = mf.where(df['hlc3_adj'].pct_change().gt(0), 0).rolling(window=length, min_periods=1).sum()
    mfn = mf.where(df['hlc3_adj'].pct_change().lt(0), 0).rolling(window=length, min_periods=1).sum()
    return 100 * (mfp / (mfp + mfn))


class ViNBuyPct(ViN):
    buy_lookback_range = range(9, 33)
    def populate_indicators_buy(self, df: DataFrame, metadata: dict) -> DataFrame:
        for i in self.buy_lookback_range:
            df[f"pctchange_{i}"] = df['close'].pct_change(periods=i)
            pctchange_mean = df[f"pctchange_{i}"].rolling(window=i, min_periods=i).mean()
            pctchange_std = df[f"pctchange_{i}"].rolling(window=i, min_periods=i).std()
            df[f"bb_pctchange_{i}_up"] = pctchange_mean + 2 * pctchange_std
            df[f"bb_pctchange_{i}_lo"] = pctchange_mean - 2 * pctchange_std
            df = df.copy()
        return df

    def populate_buy_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df.loc[:, 'buy_tag'] = ''
        df.loc[:, 'buy_tag_sum'] = 0
        for i in self.buy_lookback_range:
            buy_conditions = [
                df[f"candle_count_{self.startup_candle_count}"].ge(self.startup_candle_count),
                df['volume'].ge(self.min_candle_vol * 1.8),
                df['streak_s_min'].le(-1),
                df['streak_s_max'].between(-5, 0),
                df['streak_h'].ge(-19),
                df['streak_s_min_change'].le(0.98),
                df['streak_s_min'].ge(df['streak_h']),
                (df[f"pctchange_{i}"] / df[f"bb_pctchange_{i}_lo"]).between(1.01, 1.39),
                (df[f"bb_pctchange_{i}_up"] - df[f"bb_pctchange_{i}_lo"]).ge(0.02),
                (df['lc2_adj'] / df['close']).between(0.975, 0.995)
            ]
            buy = reduce(lambda x, y: x & y, buy_conditions)
            df.loc[buy, 'buy_tag'] += f"{i} "
            df.loc[buy, 'buy_tag_sum'] += i
        df.loc[:, 'buy'] = (
              df['buy_tag_sum'].between(9, 888)
            # & df['buy_tag'].ne('8 9 10 ')
            # & df['buy_tag'].ne('8 9 10 11 ')
            # & df['buy_tag'].ne('8 9 10 11 12 13 ')
            # & df['buy_tag'].ne('8 9 10 11 12 13 14 ')
            # & df['buy_tag'].ne('8 9 10 11 12 13 14 15 ')
            & df['buy_tag'].ne('9 ')
            & df['buy_tag'].ne('10 ')
            & df['buy_tag'].ne('12 ')
            & df['buy_tag'].ne('14 ')
            & df['buy_tag'].ne('21 ')
            & df['buy_tag'].ne('23 ')
            & df['buy_tag'].ne('15 16 17 ')
            & df['buy_tag'].ne('16 ')
            & df['buy_tag'].ne('16 17 ')
            & df['buy_tag'].ne('16 17 18 ')
            & df['buy_tag'].ne('16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 ')
            & df['buy_tag'].ne('20 21 ')
            & df['buy_tag'].ne('22 ')
            & df['buy_tag'].ne('22 23 24 ')
            & df['buy_tag'].ne('22 23 24 25 26 27 28 29 30 31 32 ')
            & df['buy_tag'].ne('24 25 ')
            & df['buy_tag'].ne('26 27 ')
            & df['buy_tag'].ne('29 ')
            & df['buy_tag'].ne('25 26 27 28 29 30 31 32 ')
            & df['buy_tag'].ne('28 29 30 31 32 ')
            & df['buy_tag'].ne('31 ')
        )
        df.loc[df['buy'], 'buy_tag'] = 'pct ' + df['buy_tag'].str.strip()
        self.fill_custom_buy_info(df, metadata)
        # print(df.loc[df['buy'], ['date', 'close', 'lc2_adj', 'volume', 'streak_s_min', 'streak_s_max', 'streak_h']])
        return df

class ViNBuyEwm(ViN):
    buy_lookback_range = (4, 6, 7, 8)
    def populate_indicators_buy(self, df: DataFrame, metadata: dict) -> DataFrame:
        ef = df[['close', 'hlc3_adj', 'volume']].reset_index()
        for i in self.buy_lookback_range:
            j = i * 4
            df[f"close_ewm_{i}"] = df['close'].ewm(span=i, min_periods=1).mean()
            df[f"close_corr_{i}"] = ef['index'].rolling(window=i, min_periods=1).corr(ef['close'], method='spearman')
            df[f"vwrs_{i}"] = vwrs(df, length=i)
            df[f"low_low_{j}"] = df['low'].rolling(window=j, min_periods=1).min()
            df = df.copy()
        return df

    def populate_buy_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df.loc[:, 'buy_tag'] = ''
        df.loc[:, 'buy_tag_sum'] = 0
        for i in self.buy_lookback_range:
            j = i * 4
            buy_conditions = [
                df[f"candle_count_{self.startup_candle_count}"].ge(self.startup_candle_count),
                df['volume'].ge(self.min_candle_vol * 1.8),
                (df['close'] / df[f"low_low_{j}"]).le(1.04),
                (df['close'] / df[f"close_ewm_{i}"]).between(0.95, 0.98),
                df[f"close_corr_{i}"].between(-0.95, -0.75),
                df[f"vwrs_{i}"].between(4, 24),
                (df['lc2_adj'] / df['close']).between(0.975, 0.995)
            ]
            buy = reduce(lambda x, y: x & y, buy_conditions)
            df.loc[buy, 'buy_tag'] += f"{i} "
            df.loc[buy, 'buy_tag_sum'] += i
        df.loc[:, 'buy'] = (
              df['buy_tag'].eq('4 ')
            | df['buy_tag'].eq('6 ')
            | df['buy_tag'].eq('4 6 ')
            | df['buy_tag'].eq('4 7 ')
        )
        df.loc[df['buy'], 'buy_tag'] = 'ewm ' + df['buy_tag'].str.strip()
        self.fill_custom_buy_info(df, metadata)
        # print(df.loc[df['buy'], ['date', 'close', 'lc2_adj', 'volume', 'low_low_20', 'close_ewm_5', 'close_corr_5', 'vwrs_5', 'low_low_24', 'close_ewm_6', 'close_corr_6', 'vwrs_6']])
        return df

class ViNSellCorr(ViN):
    lookback_candles = 25
    def custom_sell(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        df: DataFrame = df
        trade_open_date = timeframe_to_prev_date(self.timeframe, trade.open_date_utc)
        df_trade: DataFrame = df.loc[df['date'].ge(trade_open_date - timedelta(minutes=5))]
        trade_len = len(df_trade) - 1
        candle_1 = df_trade.iloc[-1]
        d = candle_1['date'].strftime('%Y-%m-%d %H:%M')
        current_profit = (candle_1['close'] - trade.open_rate) / trade.open_rate

        # current profit is not important
        # rise and fall from lowest and highest point should matter more
        # use hlc3 top and bottom

        buy_vol = df_trade['volume'].iat[0]
        trade_vol = df_trade['volume'].tail(trade_len).sum()
        trade_recent_buys = df_trade['buy'].tail(min(trade_len, 6)).sum()
        if trade_len <= 2 or trade_recent_buys >= 1 or candle_1['streak_h'] >= candle_1['streak_s_min'] or ((trade_len < 2 * self.lookback_candles or trade_vol < buy_vol * 0.1 * trade_len) and -0.06 < current_profit < 0.02):
            return None
        if candle_1['green'] and candle_1['hc2_adj'] / candle_1['close'] <= 1.002:
            return None
        t = 'profit' if current_profit >= 0.005 else 'loss'
        t += f" ({trade.buy_tag[:3]})"
        i = min(trade_len, 3 * self.lookback_candles - int(current_profit * 18))
        j = i // 2
        ef = df_trade[['close', 'hlc3_adj', 'volume', 'hc2_adj']].reset_index()

        # volume corr becomes lower and hc2_adj corr higher might indicate sell
        # ef['df_volume_corr_i'] = ef['index'].rolling(window=i, min_periods=1).corr(ef['volume'], method='spearman')
        # ef['df_hc2_adj_corr_i'] = ef['index'].rolling(window=i, min_periods=1).corr(ef['hc2_adj'], method='spearman')
        # print(ef)

        close_corr_i = ef['index'].rolling(window=i, min_periods=1).corr(ef['close'], method='spearman').iat[-1]
        close_corr_j = ef['index'].rolling(window=j, min_periods=1).corr(ef['close'], method='spearman').iat[-1]
        close_corr_ij_diff = close_corr_i - close_corr_j
        if current_profit < -0.03:
            if close_corr_ij_diff < -0.18 - 2 * current_profit and candle_1['streak_s_max'] < 1 and candle_1['lc2_adj'] / candle_1['close'] >= 0.998:
                log.info(f"{d} custom_sell: corr sell for pair {pair} with loss {current_profit:.2f} and trade len {trade_len}.")
                return f"corr {t}"
        elif current_profit > 0.01:
            if close_corr_ij_diff > 0.18 + 0.5 * current_profit:
                log.info(f"{d} custom_sell: corr sell for pair {pair} with profit {current_profit:.2f} and trade len {trade_len}.")
                return f"corr {t}"
        close_min_j = ef['close'].tail(j).min()
        close_max_j = ef['close'].tail(j).max()
        if trade_len > 4 * self.lookback_candles and close_max_j / close_min_j < min(1.04, trade_len / self.lookback_candles) and candle_1['streak_s_max'] < 1 and candle_1['streak_s_min'] < 0:
            log.info(f"{d} custom_sell: sideways sell for pair {pair} with loss {current_profit:.2f} and trade len {trade_len}.")
            return f"sideways ({trade.buy_tag[:3]})"
        return None

class ViNPctCorr(ViNBuyPct, ViNSellCorr):
    pass

class ViNEwmCorr(ViNBuyEwm, ViNSellCorr):
    pass

class ViresInNumeris(ViNBuyPct, ViNSellCorr):
    pass