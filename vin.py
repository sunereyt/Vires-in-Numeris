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
    startup_candle_count: int = 90

    @property
    def protections(self):
        return [
            {
                "method": "CooldownPeriod",
                "stop_duration_candles": 90
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
                max_concurrent_buy_signals = max(int(pairs * 0.08), 18)
                if buy_signal_count > max_concurrent_buy_signals:
                    log.info(f"{candle_1['date']:%Y-%m-%d %H:%M} confirm_trade_entry: Buy for pair {pair} with buy tag {buy_tags} is cancelled. There are {buy_signal_count} concurrent buy signals (max = {max_concurrent_buy_signals}).")
                    return False
            log.info(f"{candle_1['date']:%Y-%m-%d %H:%M} confirm_trade_entry: Buy for pair {pair} with buy tag {buy_tags} and ({buy_signal_count} concurrent buy signals).")
            if self.config['runmode'].value not in ('live', 'dry_run') and self.write_to_txt:
                indicator = []
                i = 30
                indicator.append(candle_1[f"pct_change_{i}"])
                indicator.append(candle_1[f"pct_change_{i}_lo"])
                with open(self.f_buys, 'a') as f:
                    print(f"{pair};{buy_candle_date};{rate:.10n};{buy_tags};", *indicator, sep=';', file=f)
            return True
        except:
            log.warning(f"{buy_candle_date:%Y-%m-%d %H:%M} confirm_trade_entry: No buy info for pair {pair}.")
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
    buy_lookback_range = range(8, 33)
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
        df.loc[:, 'buy_tags_sum'] = 0
        for i in self.buy_lookback_range:
            j = i * 2
            buy_conditions = [
                df[f"candle_count_{self.startup_candle_count}"].ge(self.startup_candle_count),
                df['volume'].ge(self.min_candle_vol),
                df['streak_s_min'].le(-1),
                df['streak_s_min_change'].le(0.98),
                df['streak_s_min'].ge(df['streak_b']),
                (df[f"pctchange_{i}"] / df[f"bb_pctchange_{i}_lo"]).between(1.01, 1.39),
                (df[f"bb_pctchange_{i}_up"] - df[f"bb_pctchange_{i}_lo"]).ge(0.02),
                (df['lc2_adj'] / df['close']).le(0.995)
            ]
            buy = reduce(lambda x, y: x & y, buy_conditions)
            df.loc[buy, 'buy_tag'] += f"{i} "
            df.loc[buy, 'buy_tags_sum'] += i
        df.loc[:, 'buy'] = df['buy_tags_sum'].ge(24)
        df.loc[:, 'buy_tag'] = df['buy_tag'].str.strip()
        self.fill_custom_buy_info(df, metadata)
        return df

class ViNSellCorr(ViN):
    lookback_candles = 72
    def custom_sell(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        df: DataFrame = df
        trade_open_date = timeframe_to_prev_date(self.timeframe, trade.open_date_utc)
        df_trade: DataFrame = df.loc[df['date'].ge(trade_open_date - timedelta(minutes=5))]
        trade_len = len(df_trade) - 1
        candle_1 = df_trade.iloc[-1]
        current_profit = (candle_1['close'] - trade.open_rate) / trade.open_rate
        if trade_len <= 2 or candle_1['buy'] or candle_1['streak_b'] >= candle_1['streak_s_min'] or (trade_len < self.lookback_candles and -0.04 < current_profit < 0.04):
            return None
        t = 'profit' if current_profit > 0 else 'loss'
        i = min(trade_len, self.lookback_candles + int(current_profit * 18))
        j = i // 2
        ef = df_trade[['close', 'hlc3_adj', 'volume']].reset_index()
        ef['vwrs_sell'] = vwrs(ef, length=i)
        if ef['vwrs_sell'].iat[-1] < 24:
            return None
        close_min_j = ef['close'].tail(j).min()
        close_max_j = ef['close'].tail(j).max()
        if trade_len > self.lookback_candles and close_max_j / close_min_j < min(1.04, trade_len / self.lookback_candles) and candle_1['streak_s_max'] < 1 and candle_1['streak_s_min'] < 0:
            log.info(f"{candle_1['date']:%Y-%m-%d %H:%M} custom_sell: sideways sell for pair {pair} with loss {current_profit:.2f} and trade len {trade_len}.")
            return f"sideways {t}"
        ef['close_corr_i'] = ef['index'].rolling(window=i, min_periods=1).corr(ef['close'], method='spearman')
        ef['close_corr_j'] = ef['index'].rolling(window=j, min_periods=1).corr(ef['close'], method='spearman')
        close_corr_ij_diff = ef['close_corr_i'].iat[-1] - ef['close_corr_j'].iat[-1]
        if current_profit < 0:
            if close_corr_ij_diff < -0.18 - 2 * current_profit and candle_1['streak_s_max'] < 1 and candle_1['streak_s_min'] < 0 and abs(candle_1['streak_s_min']) > 1:
                log.info(f"{candle_1['date']:%Y-%m-%d %H:%M} custom_sell: corr sell for pair {pair} with {t} {current_profit:.2f} and trade len {trade_len}.")
                return f"corr {t}"
        else:
            if close_corr_ij_diff > 0.18 + 0.5 * current_profit and abs(candle_1['streak_s_min']) > 1:
                log.info(f"{candle_1['date']:%Y-%m-%d %H:%M} custom_sell: corr sell for pair {pair} with {t} {current_profit:.2f} and trade len {trade_len}.")
                return f"corr {t}"
        return None

class ViresInNumeris(ViNBuyPct, ViNSellCorr):
    pass