from freqtrade.strategy.interface import IStrategy
from freqtrade.exchange import timeframe_to_prev_date
from freqtrade.persistence import Trade
import logging
import numpy as np
from pandas import DataFrame, Series
from functools import reduce
from datetime import datetime, timedelta
from math import sqrt, pow
import locale
locale.setlocale(category=locale.LC_ALL, locale='')
log = logging.getLogger(__name__)

class ViN(IStrategy):
    INTERFACE_VERSION = 2

    write_to_csv = False
    df_csv = './user_data/df.csv'
    write_to_txt = False
    f_buys = './user_data/vinbuys.txt'
    f_trades = './user_data/vintrades.txt'
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
    startup_candle_count: int = 72

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
        df = self.populate_indicators_buy(df, metadata)
        df = self.populate_indicators_sell(df, metadata)
        if self.config['runmode'].value not in ('live', 'dry_run') and self.write_to_csv:
            ef = df[['date', 'open', 'high', 'low', 'close', 'volume', 'bodysize', 'hlc3_adj', 'lc2_adj', 'hc2_adj', 'streak_s_min', 'streak_s_max']]
            ef['pair'] = metadata['pair']
            with open(self.df_csv, 'a') as f:
                ef.to_csv(f, sep=';', header=f.tell()==0, index=False)
        return df

    def fill_custom_buy_info(self, df:DataFrame, metadata: dict):
        df_buy: DataFrame = df.loc[df.loc[:, 'buy'], ['date', 'buy_tag', 'buy']]
        for index, row in df_buy.iterrows():
            buy_date = row['date']
            if buy_date not in self.custom_buy_info:
                self.custom_buy_info[buy_date] = {}
                self.custom_buy_info[buy_date]['buy_tags'] = row['buy_tag']
                self.custom_buy_info[buy_date]['buy_signals'] = 1
            else:
                self.custom_buy_info[buy_date]['buy_tags'] += row['buy_tag']
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
        candle_1 = df.iloc[-1]
        buy_candle_date = candle_1['date']
        try:
            buy_info = self.custom_buy_info[buy_candle_date]
            buy_tags = buy_info[pair]
            buy_signal_count = buy_info['buy_signals']
            if self.max_concurrent_buy_signals_check:
                pairs = len(self.dp.current_whitelist())
                max_concurrent_buy_signals = max(int(pairs * 0.18), 18)
                if buy_signal_count > max_concurrent_buy_signals:
                    log.info(f"confirm_trade_entry: Buy for pair {pair} with buy tag {buy_tags} on candle {buy_candle_date} is cancelled. There are {buy_signal_count} concurrent buy signals (max = {max_concurrent_buy_signals}).")
                    return False
            log.info(f"confirm_trade_entry: Buy for pair {pair} with buy tag {buy_tags} on candle {buy_candle_date} ({buy_signal_count} concurrent buy signals).")
            if self.config['runmode'].value not in ('live', 'dry_run') and self.write_to_txt:
                close_1_price = candle_1['close']
                indicator = []
                i = 12
                indicator.append(candle_1[f"mom_{i}"])
                indicator.append(candle_1[f"mom_{i}_low"])
                indicator.append(candle_1[f"mfi_{i}"])
                indicator.append(candle_1[f"close_corr_{i}"])
                with open(self.f_buys, 'a') as f:
                    print(f"{pair};{buy_candle_date};{rate:.10n};{buy_tags};{close_1_price:.10n}", *indicator, sep=';', file=f)
            return True
        except:
            log.warning(f"confirm_trade_entry: No buy info for pair {pair} on candle {buy_candle_date}.")
            return False

    def confirm_trade_exit(self, pair: str, trade: "Trade", order_type: str, amount: float,
                           rate: float, time_in_force: str, sell_reason: str, **kwargs) -> bool:
        if self.config['runmode'].value not in ('live', 'dry_run') and self.write_to_txt:
            df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            df: DataFrame = df
            trade_open_date = timeframe_to_prev_date(self.timeframe, trade.open_date_utc)
            trade_close_date = timeframe_to_prev_date(self.timeframe, trade.close_date_utc)
            df_trade = df.loc[(df['date'].ge(trade_open_date)) & (df['date'].le(trade_close_date))]
            if df_trade.empty:
                log.warning(f"confirm_trade_exit: Empty trade dataframe for pair {pair} on trade with open date {trade_open_date} and close date {trade_close_date}.")
                return False
            buy_tag = trade.buy_tag if trade is not None else 'empty'
            max_close_candle = df_trade.nlargest(1, columns=['close'])
            min_close_candle = df_trade.nsmallest(1, columns=['close'])
            min_close_date = min_close_candle['date'].to_numpy()[0]
            max_close_date = max_close_candle['date'].to_numpy()[0]
            profit = (rate - trade.open_rate) / trade.open_rate
            max_profit = (trade.max_rate - trade.open_rate) / trade.open_rate
            max_loss = (trade.min_rate - trade.open_rate) / trade.open_rate
            with open(self.f_trades, 'a') as f:
                print(f'{pair};{trade_open_date};{trade.open_rate:.10n};{trade_close_date};{rate:.10n};{buy_tag};{sell_reason.partition(" (")[0]};{profit:.10n};{max_profit:.10n};{max_loss:.10n};{trade.max_rate:.10n};{trade.min_rate:.10n};{max_close_date};{min_close_date};', file=f)
        return True

def vwrs(df: DataFrame, length: int) -> Series:
    mf: Series = df['hlc3_adj'] * df['volume']
    mfp = mf.where(df['hlc3_adj'].pct_change().gt(0), 0).rolling(window=length, min_periods=1).sum()
    mfn = mf.where(df['hlc3_adj'].pct_change().lt(0), 0).rolling(window=length, min_periods=1).sum()
    return 100 * (mfp / (mfp + mfn))


class ViNBuyPct(ViN):
    buy_lookback_range = range(10, 33)
    def populate_indicators_buy(self, df: DataFrame, metadata: dict) -> DataFrame:
        df_closechange = df['close'] - df['close'].shift(1)
        i = 12
        df['updown'] = np.where(df_closechange.rolling(window=i, min_periods=2).sum().gt(0), 1, np.where(df_closechange.rolling(window=i, min_periods=2).sum().lt(0), -1, 0))
        df[f"streak_b"] = df['updown'].groupby((df['updown'].ne(df['updown'].shift(1))).cumsum()).cumsum()
        df.drop(columns=['updown'], inplace=True)
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
        for i in self.buy_lookback_range:
            buy_conditions = [
                df[f"candle_count_{self.startup_candle_count}"].ge(self.startup_candle_count),
                df['volume'].ge(self.min_candle_vol),
                df['streak_s_min'].le(-1),
                df['streak_s_min_change'].le(0.98),
                df['streak_s_min'].ge(df['streak_b']),
                (df[f"pctchange_{i}"] / df[f"bb_pctchange_{i}_lo"]).between(1.01, 1.20),
                (df[f"bb_pctchange_{i}_up"] - df[f"bb_pctchange_{i}_lo"]).ge(0.03),
                (df['lc2_adj'] / df['close']).le(0.994)
            ]
            buy = reduce(lambda x, y: x & y, buy_conditions)
            df.loc[buy, 'buy_tag'] += f"{i} "
        df.loc[:, 'buy_signals'] = df['buy_tag'].str.len() // 3
        n = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
        df.loc[:, 'buy'] = df['buy_tag'].str.contains('25|26|27|28|29|30|31|32') | df['buy_signals'].isin(n)
        df.loc[:, 'buy_tag'] = df['buy_tag'].str.strip()
        self.fill_custom_buy_info(df, metadata)
        return df

class ViNSellCorrV1(ViN):
    def custom_sell(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        df: DataFrame = df
        trade_open_date = timeframe_to_prev_date(self.timeframe, trade.open_date_utc)
        df_trade: DataFrame = df.loc[df['date'].ge(trade_open_date - timedelta(minutes=5))]
        trade_len = len(df_trade) - 1
        candle_1 = df_trade.iloc[-1]
        current_profit = (candle_1['close'] - trade.open_rate) / trade.open_rate
        if trade_len <= 2 or candle_1['buy'] or candle_1['streak_b'] >= candle_1['streak_s_min'] or (trade_len < self.startup_candle_count and -0.08 < current_profit < 0.04):
            return None
        buy_signals = (len(trade.buy_tag) + 1) // 3
        streak_s_max_lt = 1
        streak_s_min_lt = 0
        if current_profit < -0.18 and candle_1['streak_s_max'] < streak_s_max_lt and candle_1['streak_s_min'] < streak_s_min_lt:
            log.info(f"custom_sell: stop for pair {pair} with loss {current_profit:.2f} and trade len {trade_len} on candle {candle_1['date']}.")
            return f"stop loss ({buy_signals})"
        i = min(trade_len, self.startup_candle_count)
        j = i // 2
        ef = df_trade[['close', 'hlc3_adj', 'volume']].reset_index()
        ef['vwrs_sell'] = vwrs(ef, length=i)
        ef['vwrs_corr'] = ef['index'].rolling(window=i, min_periods=1).corr(ef['vwrs_sell'], method='spearman')
        ef['close_corr_i'] = ef['index'].rolling(window=i, min_periods=1).corr(ef['close'], method='spearman')
        ef['close_corr_j'] = ef['index'].rolling(window=j, min_periods=1).corr(ef['close'], method='spearman')
        vwrs_corr_diff = ef['vwrs_corr'].iat[-1] - ef['vwrs_corr'].iat[-2]
        close_corr_i_diff = ef['close_corr_i'].iat[-1] - ef['close_corr_i'].iat[-2]
        close_corr_ij_diff = ef['close_corr_i'].iat[-1] - ef['close_corr_j'].iat[-1]
        if current_profit < -0.08:
            offset = 0.4 - pow(current_profit * 100, 4) / 1000
            t = 'big loss'
        elif current_profit < 0:
            offset = 0.6 - pow(current_profit * 100, 2) / 1000 - trade_len * 0.018
            streak_s_max_lt = 1
            streak_s_min_lt = -1
            t = 'loss'
        else:
            offset = -0.18 + sqrt(current_profit) / 100
            streak_s_max_lt = 36
            streak_s_min_lt = 18
            t = 'profit'
        if ef['vwrs_sell'].iat[-1] > 18 and close_corr_i_diff > offset and close_corr_ij_diff > 0.1 and candle_1['streak_s_max'] < streak_s_max_lt and candle_1['streak_s_min'] < streak_s_min_lt:
            log.info(f"custom_sell: corr sell for pair {pair} with {t} {current_profit:.2f} and trade len {trade_len} on candle {candle_1['date']}.")
            return f"corr sell {t} ({buy_signals})"
        elif ef['vwrs_sell'].iat[-1] > 96 + abs(current_profit) * 18 - max(54, trade_len * 0.18) and vwrs_corr_diff < -0.01 and candle_1['hc2_adj'] / candle_1['close'] >= 1.006:
            log.info(f"custom_sell: vwrs sell for pair {pair} with {t} {current_profit:.2f} and trade len {trade_len} on candle {candle_1['date']}.")
            return f"vwrs sell {t} ({buy_signals})"
        else:
            return None

class ViresInNumeris(ViNBuyPct, ViNSellCorrV1):
    pass