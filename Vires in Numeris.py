import logging
import numpy as np
import talib.abstract as ta
from pandas import DataFrame, Series
from functools import reduce
from datetime import datetime
import time
import pandas_ta as pta

import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.strategy.interface import IStrategy
from freqtrade.exchange import timeframe_to_prev_date
from freqtrade.persistence import Trade

import locale
locale.setlocale(category=locale.LC_ALL, locale='')

log = logging.getLogger(__name__)

class ViN(IStrategy):
    INTERFACE_VERSION = 2

    f_buys = './user_data/vinbuys.txt'
    f_trades = './user_data/vintrades.txt'
    write_to_csv = False
    df_csv = './user_data/df.csv'
    candle_periods = (3, 14)
    has_bt_agefilter = True
    has_downtime_protection = False
    min_vol_candle: int = 1000
    min_vol_1h: int = 10000
    max_concurrent_buy_signals: int = 10
    custom_buy_info = {}

    minimal_roi = {"0": 10}
    stoploss = -0.04
    trailing_stop = False
    use_custom_stoploss = False
    timeframe = '5m'
    process_only_new_candles = True
    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = True
    startup_candle_count: int = 144

    def populate_indicators(self, df: DataFrame, metadata: dict) -> DataFrame:
        if len(df) < max(self.candle_periods):
            return df

        # positive = (df['close'] - df['open']).ge(0)
        # bodysize = (df['close'] / df['open']).where(positive, 0 - df['open'] / df['close'])
        # df['uppertail'] = (df['high'] / df['close']).where(positive, df['high'] / df['open'])
        # df['lowertail'] = (df['open'] / df['low']).where(positive, df['close'] / df['low'])

        # streaks
        df_closechange = df['close'] / df['close'].shift(1)
        df['updown'] = np.where(df_closechange.gt(1), 1, np.where(df_closechange.lt(1), -1, 0))
        df['streak'] = df['updown'].groupby((df['updown'].ne(df['updown'].shift(1))).cumsum()).cumsum()

        df['updown_smoothed_2'] = np.where(df['close'].rolling(2).sum().gt(1), 1, np.where(df['close'].rolling(2).sum().lt(1), -1, 0))
        df['streak_smoothed_2'] = df['updown_smoothed_2'].groupby((df['updown_smoothed_2'].ne(df['updown_smoothed_2'].shift(1))).cumsum()).cumsum()

        # tops and bottoms (be careful with backtesting, this is a look ahead)
        # df['topbottom'] = np.where(df_closechange.ge(1) & df_closechange.shift(-1).le(1), 'T', np.where(df_closechange.le(1) & df_closechange.shift(-1).ge(1), 'B', ''))
        # change since last top or bottom
        # df['topbottomchange'] = df['close'] / (df['open'] * df['topbottom'].replace({'':np.nan, 'T':1, 'B': 1})).fillna(method='ffill').shift(1)

        for i in self.candle_periods:
            mom = ta.MOM(df, timeperiod=i)
            df[f"mom_{i}"] = mom
            upperband, middleband, lowerband = ta.BBANDS(mom, timeperiod=i, nbdevup=2.0, nbdevdn=2.0, matype=0)
            df[f"mom_{i}_upp"] = upperband
            df[f"mom_{i}_low"] = lowerband
            df[f"rsi_{i}"] = ta.RSI(df, timeperiod=i)
            df[f"mfi_{i}"] = ta.MFI(df, timeperiod=i)
            df[f"cti_{i}"] = pta.cti(df['close'], length=i)
            df[f"cmf_{i}"] = chaikin_money_flow(df, i)

        # crsi_closechange = df['close'] / df['close'].shift(1)
        # crsi_updown = np.where(crsi_closechange.gt(1), 1.0, np.where(crsi_closechange.lt(1), -1.0, 0.0))
        # df['crsi_100'] = (ta.RSI(df['close'], timeperiod=3) + ta.RSI(crsi_updown, timeperiod=2) + ta.ROC(df['close'], 100)) / 3
        # df['crsi_480'] = (ta.RSI(df['close'], timeperiod=6) + ta.RSI(crsi_updown, timeperiod=3) + ta.ROC(df['close'], 480)) / 3

        df['volume_12'] = df['volume'].rolling(12).sum()

        if self.config['runmode'].value in ('live', 'dry_run'):
            if self.has_downtime_protection:
                df['live_data_ok'] = (df['volume'].rolling(window=72, min_periods=72).min() > 0)
        else:
            if self.has_bt_agefilter:
                df['bt_agefilter'] = df['volume'].rolling(window=self.startup_candle_count, min_periods=self.startup_candle_count).count()

        return df

    def populate_buy_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        if len(df) < max(self.candle_periods):
            return df

        conditions = []
        df.loc[:, 'buy_tag'] = ''

        conditions.append(df['volume'].ge(self.min_vol_candle))
        conditions.append(df['volume_12'].ge(self.min_vol_1h))
        if self.config['runmode'].value in ('live', 'dry_run'):
            if self.has_downtime_protection:
                conditions.append(df['live_data_ok'])
        else:
            if self.has_bt_agefilter:
                conditions.append(df['bt_agefilter'].ge(self.startup_candle_count))

        i = 14
        conditions.append((df[f"mom_{i}"] / df[f"mom_{i}_low"]).between(1.1, 1.3))
        conditions.append(df[f"rsi_{i}"].between(10, 30))
        conditions.append(df[f"mfi_{i}"].between(0, 20))
        conditions.append(df[f"cti_{i}"].between(-0.95, -0.75))
        conditions.append(df[f"cti_{i}"].ge(df[f"cti_{i}"].shift(1)))

        buy_signal = reduce(lambda x, y: x & y, conditions)
        df.loc[:, 'buy'] = buy_signal
        df.loc[buy_signal, 'buy_tag'] = f"cp{i}"

        df_buy: DataFrame = df.loc[df.loc[:, 'buy'], ['date', 'buy_tag', 'buy']]
        for index, row in df_buy.iterrows():
            buy_date = row['date']
            if buy_date not in self.custom_buy_info:
                self.custom_buy_info[buy_date] = {}
                self.custom_buy_info[buy_date][metadata['pair']] = row['buy_tag']
                self.custom_buy_info[buy_date]['buy_tags'] = row['buy_tag']
                self.custom_buy_info[buy_date]['buy_signals'] = 1
            else:
                self.custom_buy_info[buy_date][metadata['pair']] = row['buy_tag']
                self.custom_buy_info[buy_date]['buy_tags'] += row['buy_tag']
                self.custom_buy_info[buy_date]['buy_signals'] += 1

        if self.config['runmode'].value not in ('live', 'dry_run'):
            if self.write_to_csv:
                df['pair'] = metadata['pair']
                with open(self.df_csv, 'a') as f:
                    df.to_csv(f, sep='\t', header=f.tell()==0, index=False)

        return df

    def populate_sell_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        if len(df) < max(self.candle_periods):
            return df

        df.loc[:, 'sell'] = False
        df.loc[:, 'sell_tag'] = ''

        df.loc[(df['mom_14'] / df['mom_14_upp']).between(1.0, 1.1), 'sell_tag'] +='mom'
        df.loc[df['rsi_14'].ge(65), 'sell_tag'] +='rsi'
        df.loc[df['mfi_14'].ge(75), 'sell_tag'] +='mfi'
        df.loc[df['cti_14'].ge(0.75) & df['cti_14'].le(df['cti_14'].shift(1)), 'sell_tag'] +='cti'
        df.loc[df['cmf_14'].le(0.05) & df['cmf_14'].le(df['cmf_14'].shift(1)), 'sell_tag'] +='cmf'

        return df

    def custom_sell(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)

        trade_open_date = timeframe_to_prev_date(self.timeframe, trade.open_date_utc)
        df_trade: DataFrame = df[df['date'] >= trade_open_date]
        if df_trade.empty:
            return None

        buy_tag = 'empty'
        if hasattr(trade, 'buy_tag') and trade.buy_tag is not None:
            buy_tag = trade.buy_tag

        # use close instead of trade prices
        candle_1 = df.iloc[-1].copy(deep=False)
        max_close_candle = df_trade.nlargest(1, columns=['close'])
        min_close_candle = df_trade.nsmallest(1, columns=['close'])
        current_rate = candle_1['close']
        current_profit = (current_rate - trade.open_rate) / trade.open_rate
        trade.max_rate = max_close_candle['close'].iloc[0]
        trade.min_rate = min_close_candle['close'].iloc[0]

        if current_profit < -0.01:
            candle_1['buy'] = False
            return f"stoploss ({buy_tag})"

        if len(candle_1['sell_tag']) >= 6 and current_profit > 0.01:
                return f"{candle_1['sell_tag']} ({buy_tag})"

        return None

    def bot_loop_start(self, **kwargs) -> None:
        if self.config['runmode'].value not in ('live', 'dry_run'):
            with open(self.f_buys, 'w') as f:
                print('pair;date open;trade open rate;buy tags;close_1;close_2;mom;mom_low;rsi;mfi;cti;cmf;mom;rsi;mfi;cti;cmf', file=f)
            with open(self.f_trades, 'w') as f:
                print('pair;date open;trade open rate;date close;trade rate;buy tags;sell reason;profit;max profit;max loss;max rate;min rate;max close date;min close date', file=f)
            if self.write_to_csv:
                with open(self.df_csv, 'w') as f:
                    pass

        return None

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                            time_in_force: str, current_time: datetime, **kwargs) -> bool:
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        candle_1: Series = df.iloc[-1]
        candle_2: Series = df.iloc[-2]
        buy_candle_date = candle_1['date']

        if buy_candle_date in self.custom_buy_info.keys():
            buy_tags = self.custom_buy_info[buy_candle_date][pair]
            if  self.max_concurrent_buy_signals > 0:
                buy_info = self.custom_buy_info[buy_candle_date]
                buy_signal_count = buy_info['buy_signals']
                if buy_signal_count > self.max_concurrent_buy_signals:
                    log.info(f"Buy for pair {pair} with buy tag {buy_tags} on candle {buy_candle_date} is cancelled. There are {buy_signal_count} concurrent buy signals (max = {self.max_concurrent_buy_signals}).")
                    return False

            if self.config['runmode'].value not in ('live', 'dry_run'):
                close_1_price = candle_1['close']
                close_2_price = candle_2['close']
                indicator = []
                period = 14
                indicator.append(candle_1[f"mom_{period}"])
                indicator.append(candle_1[f"mom_{period}_low"])
                indicator.append(candle_1[f"rsi_{period}"])
                indicator.append(candle_1[f"mfi_{period}"])
                indicator.append(candle_1[f"cti_{period}"])
                indicator.append(candle_1[f"cmf_{period}"])
                indicator.append(candle_2[f"mom_{period}"])
                indicator.append(candle_2[f"rsi_{period}"])
                indicator.append(candle_2[f"mfi_{period}"])
                indicator.append(candle_2[f"cti_{period}"])
                indicator.append(candle_2[f"cmf_{period}"])
                with open(self.f_buys, 'a') as f:
                    print(f"{pair};{buy_candle_date};{rate:.10n};{buy_tags};{close_1_price:.10n};{close_2_price:.10n}", *indicator, sep=';', file=f)
        else:
            log.warning(f"confirm_trade_entry: No buy info for pair {pair} on candle {buy_candle_date}.")

        return True

    def confirm_trade_exit(self, pair: str, trade: "Trade", order_type: str, amount: float,
                           rate: float, time_in_force: str, sell_reason: str, **kwargs) -> bool:
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)

        if self.config['runmode'].value not in ('live', 'dry_run'):
            trade_open_date = timeframe_to_prev_date(self.timeframe, trade.open_date_utc)
            trade_close_date = timeframe_to_prev_date(self.timeframe, trade.close_date_utc)
            buy_tag = trade.buy_tag if trade is not None else 'empty'

            df_trade: DataFrame = df[(df['date'].ge(trade_open_date)) & (df['date'].le(trade_close_date))]
            if df_trade.empty:
                log.warning(f"confirm_trade_exit: Empty trade df for pair {pair} on trade open date {trade_open_date}.")
                return False

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

# Chaikin Money Flow
def chaikin_money_flow(df: DataFrame, n=20, fillna=False) -> Series:
    mfv: Series = (2 * df['close'] - df['high'] - df['low']) / (df['high'] - df['low'])
    mfv = mfv.fillna(0.0)  # float division by zero
    mfv *= df['volume']
    cmf = (mfv.rolling(n, min_periods=0).sum() / df['volume'].rolling(n, min_periods=0).sum())
    if fillna:
        cmf = cmf.replace([np.inf, -np.inf], np.nan).fillna(0)

    return cmf

class ViNall(ViN):

    f_buys = './user_data/vinallbuys.txt'
    f_trades = './user_data/vinalltrades.txt'

    # Maximum number of concurrent buy signals (0 is disable)
    max_concurrent_buy_signals = 0
    # Maximum number of buys with the same buy tag (0 is disable)
    max_same_buy_tags = 0

class ViN3(ViN):

    f_buys = './user_data/vin3buys.txt'
    f_trades = './user_data/vin3trades.txt'

    def populate_buy_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        if len(df) < 42:
            return df

        conditions = []
        df.loc[:, 'buy_tag'] = ''

        conditions.append(df['volume'].ge(self.min_vol_candle))
        conditions.append(df['volume_12'].ge(self.min_vol_1h))
        if self.config['runmode'].value in ('live', 'dry_run'):
            if self.has_downtime_protection:
                conditions.append(df['live_data_ok'])
        else:
            if self.has_bt_agefilter:
                conditions.append(df['bt_agefilter'].ge(self.startup_candle_count))

        df.loc[(df['mom_3'] / df['mom_3_low']).between(1.0, 1.3), 'buy_tag'] +='mom'
        df.loc[df['rsi_3'].between(5, 20), 'buy_tag'] += 'rsi'
        df.loc[df['mfi_3'].between(5, 20), 'buy_tag'] += 'mfi'
        df.loc[df['cti_3'].between(-0.75, -0.50) & df['cti_3'].ge(df['cti_3'].shift(1)), 'buy_tag'] += 'cti'
        df.loc[df['cmf_3'].between(-0.75, -0.05) & df['cmf_3'].ge(df['cmf_3'].shift(1)), 'buy_tag'] += 'cmf'
        conditions.append(df.loc[:, 'buy_tag'].str.len().ge(12))

        if conditions:
            df.loc[:, 'buy'] = reduce(lambda x, y: x & y, conditions)

            df_buy: DataFrame = df.loc[df.loc[:, 'buy'], ['date', 'buy_tag', 'buy']]
            for index, row in df_buy.iterrows():
                buy_date = row['date']
                if buy_date not in self.custom_buy_info:
                    self.custom_buy_info[buy_date] = {}
                    self.custom_buy_info[buy_date][metadata['pair']] = row['buy_tag']
                    self.custom_buy_info[buy_date]['buy_tags'] = row['buy_tag']
                    self.custom_buy_info[buy_date]['buy_signals'] = 1
                else:
                    self.custom_buy_info[buy_date][metadata['pair']] = row['buy_tag']
                    self.custom_buy_info[buy_date]['buy_tags'] += row['buy_tag']
                    self.custom_buy_info[buy_date]['buy_signals'] += 1

        if self.config['runmode'].value not in ('live', 'dry_run'):
            if self.write_to_csv:
                df['pair'] = metadata['pair']
                with open(self.df_csv, 'a') as f:
                    df.to_csv(f, sep='\t', header=f.tell()==0, index=False)

        return df

