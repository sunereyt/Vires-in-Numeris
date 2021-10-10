import logging
import numpy as np
import talib.abstract as ta
from pandas import DataFrame, Series
from functools import reduce
from datetime import datetime

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
    buy_time_periods = (13, 14, 15, 16, 17)
    indicator_range = range(3, max(buy_time_periods)+1)
    has_bt_agefilter = True
    has_downtime_protection = False
    min_vol_candle: int = 1000
    min_vol_1h: int = 10000
    max_concurrent_buy_signals: int = 15
    custom_buy_info = {}

    minimal_roi = {"0": 10}
    stoploss = -0.99 #-0.08
    stoploss_on_exchange = False # True
    trailing_stop = False
    use_custom_stoploss = False
    timeframe = '5m'
    process_only_new_candles = True
    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = True
    startup_candle_count: int = 144

    def populate_indicators(self, df: DataFrame, metadata: dict) -> DataFrame:
        if len(df) < self.startup_candle_count:
            return df

        # positive = (df['close'] - df['open']).ge(0)
        # bodysize = (df['close'] / df['open']).where(positive, 0 - df['open'] / df['close'])
        # df['uppertail'] = (df['high'] / df['close']).where(positive, df['high'] / df['open'])
        # df['lowertail'] = (df['open'] / df['low']).where(positive, df['close'] / df['low'])

        df_closechange = df['close'] - df['close'].shift(1)
        for i in (1, 2, 3):
            df['updown'] = np.where(df_closechange.rolling(window=i, min_periods=1).sum().gt(0), 1, np.where(df_closechange.rolling(window=i, min_periods=1).sum().lt(0), -1, 0))
            df[f"streak_{i}"] = df['updown'].groupby((df['updown'].ne(df['updown'].shift(1))).cumsum()).cumsum()

        df['streak_min'] = df[['streak_1', 'streak_2', 'streak_3']].min(axis=1)
        df['streak_max'] = df[['streak_1', 'streak_2', 'streak_3']].max(axis=1)
        df.drop(columns=['updown', 'streak_1', 'streak_2', 'streak_3'])

        cf = df['close'].reset_index()
        for i in self.indicator_range:
            df[f"mom_{i}"] = ta.MOM(df, timeperiod=i)
            upp, mid, df[f"mom_{i}_low"] = ta.BBANDS(df[f"mom_{i}"], timeperiod=i, nbdevup=2.0, nbdevdn=2.0, matype=0)
            df[f"rsi_{i}"] = ta.RSI(df, timeperiod=i)
            df[f"mfi_{i}"] = ta.MFI(df, timeperiod=i)
            df[f"cti_{i}"] = cf['index'].rolling(i).corr(cf['close'], method='spearman')

        df['volume_12'] = df['volume'].rolling(12).sum()
        if self.config['runmode'].value in ('live', 'dry_run'):
            if self.has_downtime_protection:
                df['live_data_ok'] = (df['volume'].rolling(window=72, min_periods=72).min() > 0)
        else:
            if self.has_bt_agefilter:
                df['bt_agefilter'] = df['volume'].rolling(window=self.startup_candle_count, min_periods=self.startup_candle_count).count()

        return df.copy()

    def populate_buy_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        if len(df) < self.startup_candle_count:
            return df

        df.loc[:, 'buy_tag'] = ''
        conditions = []
        conditions.append(df['volume'].ge(self.min_vol_candle))
        conditions.append(df['volume_12'].ge(self.min_vol_1h))
        if self.config['runmode'].value in ('live', 'dry_run'):
            if self.has_downtime_protection:
                conditions.append(df['live_data_ok'])
        else:
            if self.has_bt_agefilter:
                conditions.append(df['bt_agefilter'].ge(self.startup_candle_count))

        for i in self.buy_time_periods:
            buy_condition = []
            if i == min(self.buy_time_periods):
                buy_condition.append(df['streak_min'].between(-i, -3))
            elif i == max(self.buy_time_periods):
                buy_condition.append(df['streak_min'].le(-i))
            else:
                buy_condition.append(df['streak_min'].eq(-i))
            buy_condition.append((df[f"mom_{i}"] / df[f"mom_{i}_low"]).between(1.05, 1.25))
            buy_condition.append(df[f"rsi_{i}"].between(10, 10 + i))
            buy_condition.append(df[f"mfi_{i}"].between(0, 7 + i))
            buy_condition.append(df[f"cti_{i}"].between(-0.95, (-0.90 + i / 100)))
            buy_condition.append(df[f"cti_{i-1}"].ge(df[f"cti_{i}"]))

            buy = reduce(lambda x, y: x & y, buy_condition)
            df.loc[buy, 'buy_tag'] = f"buy_{i}"

        conditions.append(df.loc[:, 'buy_tag'] != '')
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

    def populate_sell_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        if len(df) < self.startup_candle_count:
            return df

        df.loc[:, 'sell_tag'] = ''
        sell_up_time_periods = range(6, 16)
        for i in sell_up_time_periods:
            sell_condition = []
            if i == min(sell_up_time_periods):
                sell_condition.append(df['streak_max'].between(3, i))
            elif i == max(sell_up_time_periods):
                sell_condition.append(df['streak_max'].ge(i))
            else:
                sell_condition.append(df['streak_max'].eq(i))
            sell_condition.append(df[f"rsi_{i}"].between(92 - i, 100))
            sell_condition.append(df[f"mfi_{i}"].between(97 - i, 100))
            sell_condition.append(df[f"cti_{i}"].ge((i - 6) / 10))
            sell_condition.append(df[f"cti_{i-1}"].le(df[f"cti_{i}"]))

            sell = reduce(lambda x, y: x & y, sell_condition)
            df.loc[sell, 'sell_tag'] = f"sell_up_{i}"

        try:
            candles_between = df.index[-1] - df.loc[df['buy_tag'] != ''].index[-1]
        except:
            candles_between = self.startup_candle_count
        sell_down_time_periods = range(4, min(candles_between, 16))
        for i in sell_down_time_periods:
            sell_condition = []
            if i == min(sell_down_time_periods):
                sell_condition.append(df['streak_min'].between(-i, -3))
            elif i == max(sell_down_time_periods):
                sell_condition.append(df['streak_min'].le(-i))
            else:
                sell_condition.append(df['streak_min'].eq(-i))
            sell_condition.append(df[f"rsi_{i}"].between(43 - i, 100))
            sell_condition.append(df[f"mfi_{i}"].between(35 - i, 100))
            sell_condition.append(df[f"cti_{i}"].le((-i + 6) / 10))
            sell_condition.append(df[f"cti_{i-1}"].gt(df[f"cti_{i}"]))

            sell = reduce(lambda x, y: x & y, sell_condition)
            df.loc[sell, 'sell_tag'] = f"sell_down_{i}"

        df.loc[:, 'sell'] = False
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

        candle_1 = df_trade.iloc[-1].copy(deep=False)
        # max_close_candle = df_trade.nlargest(1, columns=['close'])
        # min_close_candle = df_trade.nsmallest(1, columns=['close'])
        # trade.min_rate = min_close_candle['close'].iloc[0]
        # max_rate = max_close_candle['close'].iloc[0]
        current_rate = candle_1['close']
        current_profit = (current_rate - trade.open_rate) / trade.open_rate
        # max_profit = (max_rate - trade.open_rate) / trade.open_rate

        if current_profit > 0.015 or current_profit < -0.015:
            sell_tag = candle_1['sell_tag']
            if sell_tag != '':
                if candle_1['buy']:
                    log.info(f"custom sell: sell cancelled with sell_tag {candle_1['sell_tag']} and buy_tag {candle_1['buy_tag']}")
                    return None
                else:
                    candle_1['sell'] = True
                    return f"{sell_tag} ({buy_tag})"

        return None

    def bot_loop_start(self, **kwargs) -> None:
        if self.config['runmode'].value not in ('live', 'dry_run'):
            with open(self.f_buys, 'w') as f:
                print('pair;date open;trade open rate;buy tags;close_1;close_2;mom;mom_low;rsi;mfi;cti;mom;rsi;mfi;cti', file=f)
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
                    log.info(f"confirm_trade_entry: Buy for pair {pair} with buy tag {buy_tags} on candle {buy_candle_date} is cancelled. There are {buy_signal_count} concurrent buy signals (max = {self.max_concurrent_buy_signals}).")
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
                indicator.append(candle_2[f"mom_{period}"])
                indicator.append(candle_2[f"rsi_{period}"])
                indicator.append(candle_2[f"mfi_{period}"])
                indicator.append(candle_2[f"cti_{period}"])
                with open(self.f_buys, 'a') as f:
                    print(f"{pair};{buy_candle_date};{rate:.10n};{buy_tags};{close_1_price:.10n};{close_2_price:.10n}", *indicator, sep=';', file=f)
        else:
            log.warning(f"confirm_trade_entry: No buy info for pair {pair} on candle {buy_candle_date}.")

        return True

    def confirm_trade_exit(self, pair: str, trade: "Trade", order_type: str, amount: float,
                           rate: float, time_in_force: str, sell_reason: str, **kwargs) -> bool:
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        df: DataFrame = df

        if self.config['runmode'].value not in ('live', 'dry_run'):
            trade_open_date = timeframe_to_prev_date(self.timeframe, trade.open_date_utc)
            trade_close_date = timeframe_to_prev_date(self.timeframe, trade.close_date_utc)
            buy_tag = trade.buy_tag if trade is not None else 'empty'

            df_trade = df[(df['date'].ge(trade_open_date)) & (df['date'].le(trade_close_date))]
            if df_trade.empty:
                log.warning(f"confirm_trade_exit: Empty trade dataframe for pair {pair} on trade with open date {trade_open_date} and close date {trade_close_date}.")
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


class ViNall(ViN):

    f_buys = './user_data/vinallbuys.txt'
    f_trades = './user_data/vinalltrades.txt'

    # Maximum number of concurrent buy signals (0 is disable)
    max_concurrent_buy_signals = 0
    # Maximum number of buys with the same buy tag (0 is disable)
    max_same_buy_tags = 0
