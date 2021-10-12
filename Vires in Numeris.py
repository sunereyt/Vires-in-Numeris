from freqtrade.strategy.interface import IStrategy
from freqtrade.exchange import timeframe_to_prev_date
from freqtrade.persistence import Trade
import logging
import numpy as np
import talib.abstract as ta
from pandas import DataFrame, Series
from functools import reduce
from datetime import datetime
import locale
locale.setlocale(category=locale.LC_ALL, locale='')
log = logging.getLogger(__name__)

class ViN(IStrategy):
    INTERFACE_VERSION = 2

    write_to_txt = False
    f_buys = './user_data/vinbuys.txt'
    f_trades = './user_data/vintrades.txt'
    write_to_csv = False
    df_csv = './user_data/df.csv'
    buy_time_periods = (6, 13, 14, 15, 16, 19)
    indicator_range = range(2, 20) #max(buy_time_periods)+1)
    min_candle_vol: int = 1500
    min_candle_vol_6: int = min_candle_vol * 4
    custom_buy_info = {}

    minimal_roi = {"0": 10}
    stoploss = -0.08
    stoploss_on_exchange = True
    trailing_stop = False
    use_custom_stoploss = False
    timeframe = '5m'
    process_only_new_candles = True
    use_sell_signal = True
    sell_profit_only = False
    startup_candle_count: int = 36

    d_buy = { 3: [1.05, 1.25, 20, -0.85, 1.002],
              4: [1.05, 1.25, 20, -0.85, 1.002],
              5: [1.05, 1.25, 20, -0.85, 1.002],
              6: [1.05, 1.25, 30, -0.75, 1.002],
              7: [1.05, 1.25, 20, -0.85, 1.002],
              8: [1.05, 1.25, 20, -0.85, 1.002],
              9: [1.05, 1.25, 20, -0.85, 1.002],
             10: [1.05, 1.25, 20, -0.85, 1.002],
             11: [1.05, 1.25, 20, -0.85, 1.002],
             12: [1.05, 1.25, 20, -0.85, 1.002],
             13: [1.05, 1.25, 30, -0.75, 1.002],
             14: [1.05, 1.25, 20, -0.75, 1.002],
             15: [1.05, 1.25, 20, -0.75, 1.002],
             16: [1.05, 1.25, 20, -0.75, 1.002],
             17: [1.05, 1.25, 20, -0.85, 1.002],
             18: [1.05, 1.25, 20, -0.85, 1.002],
             19: [1.05, 1.25, 20, -0.75, 1.002],
             20: [1.05, 1.25, 20, -0.85, 1.002],
            }

    def populate_indicators(self, df: DataFrame, metadata: dict) -> DataFrame:
        if len(df) < self.startup_candle_count:
            return df

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
            df[f"mfi_{i}"] = ta.MFI(df, timeperiod=i)
            df[f"cti_{i}"] = cf['index'].rolling(i).corr(cf['close'], method='spearman')

        positive = (df['close'] - df['open']).ge(0)
        # bodysize = (df['close'] / df['open']).where(positive, 0 - df['open'] / df['close'])
        # df['uppertail'] = (df['high'] / df['close']).where(positive, df['high'] / df['open'])
        df['lowertail'] = (df['open'] / df['low']).where(positive, df['close'] / df['low'])
        df['closechange'] = df['close'].pct_change()

        df['volume_6'] = df['volume'].rolling(6).sum()
        df['candle_count'] = df['volume'].rolling(window=self.startup_candle_count, min_periods=self.startup_candle_count).count()

        return df.copy()

    def populate_buy_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        if len(df) < self.startup_candle_count:
            return df

        df.loc[:, 'buy_tag'] = ''
        conditions = []
        conditions.append(df['candle_count'].ge(self.startup_candle_count))
        conditions.append(df['volume'].ge(self.min_candle_vol))
        conditions.append(df['volume_6'].ge(self.min_candle_vol_6))

        for i in self.buy_time_periods:
            buy_condition = []
            buy_par = self.d_buy[i]
            # if i == min(self.buy_time_periods):
            #     buy_condition.append(df['streak_min'].between(-i, -3))
            # elif i == max(self.buy_time_periods):
            #     buy_condition.append(df['streak_min'].le(-i))
            # else:
            buy_condition.append(df['streak_min'].eq(-i))
            buy_condition.append((df[f"mom_{i}"] / df[f"mom_{i}_low"]).between(buy_par[0], buy_par[1]))
            buy_condition.append(df[f"mfi_{i}"].between(0, buy_par[2]))
            buy_condition.append(df[f"cti_{i}"].between(-0.95, buy_par[3]))
            buy_condition.append(df[f"cti_{i-1}"].ge(df[f"cti_{i}"]))
            buy_condition.append(df['lowertail'].ge(buy_par[4]))

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

        if self.config['runmode'].value not in ('live', 'dry_run') and self.write_to_csv:
            df['pair'] = metadata['pair']
            with open(self.df_csv, 'a') as f:
                df.to_csv(f, sep='\t', header=f.tell()==0, index=False)

        return df

    def populate_sell_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        if len(df) < self.startup_candle_count:
            return df

        df.loc[:, 'sell_tag'] = ''
        try:
            candles_between = df.index[-1] - df.loc[df['buy_tag'] != ''].index[-1]
        except:
            candles_between = self.startup_candle_count
        time_periods = range(5, min(candles_between, 16))
        for i in time_periods:
            sell_conditions = []
            sell_conditions.append(df['candle_count'].ge(self.startup_candle_count))
            sell_conditions.append(df['volume'].ge(self.min_candle_vol))
            sell_conditions.append(df['volume_6'].ge(self.min_candle_vol_6))
            if i == min(time_periods):
                sell_conditions.append(df['streak_max'].between(3, i))
            elif i == max(time_periods):
                sell_conditions.append(df['streak_max'].ge(i))
            else:
                sell_conditions.append(df['streak_max'].eq(i))
            sell_conditions.append(df[f"mfi_{i}"].between(96 - i, 100))
            sell_conditions.append(df[f"cti_{i}"].ge((i - 6) / 10))
            sell_conditions.append(df[f"cti_{i-1}"].le(df[f"cti_{i}"]))

            sell = reduce(lambda x, y: x & y, sell_conditions)
            df.loc[sell, 'sell_tag'] = f"sell_up_{i}"

        time_periods = range(5, min(candles_between, 16))
        for i in time_periods:
            sell_conditions = []
            sell_conditions.append(df['candle_count'].ge(self.startup_candle_count))
            sell_conditions.append(df['volume'].ge(self.min_candle_vol))
            if i == min(time_periods):
                sell_conditions.append(df['streak_min'].between(-i, -3))
            elif i == max(time_periods):
                sell_conditions.append(df['streak_min'].le(-i))
            else:
                sell_conditions.append(df['streak_min'].eq(-i))
            sell_conditions.append(df[f"mfi_{i}"].between(35 - i, 100))
            sell_conditions.append(df[f"cti_{i}"].le((-i + 6) / 10))
            sell_conditions.append(df[f"cti_{i-1}"].gt(df[f"cti_{i}"]))
            sell_conditions.append(df[f"cti_2"].lt(0.01))

            sell = reduce(lambda x, y: x & y, sell_conditions)
            df.loc[sell, 'sell_tag'] = f"sell_down_{i}"

        # time_periods = range(5, min(candles_between, 16))
        # for i in time_periods:
        #     sell_conditions = []
        #     sell_conditions.append(df['candle_count'].ge(self.startup_candle_count))
        #     sell_conditions.append(df['volume'].ge(self.min_candle_vol))
        #     if i == min(time_periods):
        #         sell_conditions.append(df['streak_min'].between(-i, -3))
        #     elif i == max(time_periods):
        #         sell_conditions.append(df['streak_min'].le(-i))
        #     else:
        #         sell_conditions.append(df['streak_min'].eq(-i))
        #     sell_conditions.append(df[f"mfi_{i}"].between(35 - i, 100))
        #     sell_conditions.append(df[f"cti_{i}"].le((-i + 6) / 10))
        #     sell_conditions.append(df[f"cti_{i-1}"].gt(df[f"cti_{i}"]))
        #     sell_conditions.append(df[f"cti_2"].lt(0.01))

        #     sell = reduce(lambda x, y: x & y, sell_conditions)
        #     df.loc[sell, 'sell_tag'] += f"stop_{i}"

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

        candle_1 = df_trade.iloc[-1]
        current_rate = candle_1['close']
        current_profit = (current_rate - trade.open_rate) / trade.open_rate
        sell_tag = candle_1['sell_tag']
        if current_profit < -0.015 and 'sell_down' in sell_tag:
            return f"{sell_tag} ({buy_tag})"
        if current_profit > 0.015 and 'sell' in sell_tag:
            if candle_1['buy']:
                log.info(f"custom_sell: sell for pair {pair} with sell_tag {sell_tag} and buy_tag {buy_tag} on candle {candle_1['date']} is cancelled.")
                return None
            else:
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
            pairs = len(self.dp.current_whitelist())
            max_concurrent_buy_signals = int(pairs * 0.08)
            buy_tags = self.custom_buy_info[buy_candle_date][pair]
            if  max_concurrent_buy_signals > 0:
                buy_info = self.custom_buy_info[buy_candle_date]
                buy_signal_count = buy_info['buy_signals']
                if buy_signal_count > max_concurrent_buy_signals:
                    log.info(f"confirm_trade_entry: Buy for pair {pair} with buy tag {buy_tags} on candle {buy_candle_date} is cancelled. There are {buy_signal_count} concurrent buy signals (max = {max_concurrent_buy_signals}).")
                    return False

            if self.config['runmode'].value not in ('live', 'dry_run') and self.write_to_txt:
                close_1_price = candle_1['close']
                close_2_price = candle_2['close']
                indicator = []
                period = 14
                indicator.append(candle_1[f"mom_{period}"])
                indicator.append(candle_1[f"mom_{period}_low"])
                indicator.append(candle_1[f"mfi_{period}"])
                indicator.append(candle_1[f"cti_{period}"])
                indicator.append(candle_2[f"mom_{period}"])
                indicator.append(candle_2[f"mfi_{period}"])
                indicator.append(candle_2[f"cti_{period}"])
                with open(self.f_buys, 'a') as f:
                    print(f"{pair};{buy_candle_date};{rate:.10n};{buy_tags};{close_1_price:.10n};{close_2_price:.10n}", *indicator, sep=';', file=f)
        else:
            log.warning(f"confirm_trade_entry: No buy info for pair {pair} on candle {buy_candle_date}.")

        return True

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

class ViN1(ViN):
    b = __name__.lower()
    if b not in ('vin'):
        n = int("".join(filter(str.isdigit, b)))
        buy_time_periods = (n,)

class ViN3(ViN1):
    pass

class ViN4(ViN1):
    pass

class ViN5(ViN1):
    pass

class ViN6(ViN1):
    pass

class ViN7(ViN1):
    pass

class ViN8(ViN1):
    pass

class ViN9(ViN1):
    pass

class ViN10(ViN1):
    pass

class ViN11(ViN1):
    pass

class ViN12(ViN1):
    pass

class ViN13(ViN1):
    pass

class ViN14(ViN1):
    pass

class ViN15(ViN1):
    pass

class ViN16(ViN1):
    pass

class ViN17(ViN1):
    pass

class ViN18(ViN1):
    pass

class ViN19(ViN1):
    pass

class ViN20(ViN1):
    pass

