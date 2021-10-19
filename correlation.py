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

class CR(IStrategy):
    INTERFACE_VERSION = 2

    stoploss_count: int = 0
    write_to_txt = False
    f_buys = './user_data/crbuys.txt'
    f_trades = './user_data/crtrades.txt'
    write_to_csv = False
    df_csv = './user_data/df.csv'
    indicator_range = range(2, 16)
    min_candle_vol: int = 0
    custom_buy_info = {}

    minimal_roi = {"0": 10}
    stoploss = -0.99 # -0.08
    stoploss_on_exchange = False # True
    trailing_stop = False
    use_custom_stoploss = False
    timeframe = '5m'
    process_only_new_candles = True
    use_sell_signal = True
    sell_profit_only = False
    startup_candle_count: int = 18

    def populate_indicators(self, df: DataFrame, metadata: dict) -> DataFrame:
        if len(df) < self.startup_candle_count:
            return df

        df['positive'] = (df['close'] - df['open']).ge(0)
        df['uppertail'] = (df['high'] / df['close']).where(df['positive'], df['high'] / df['open'])
        df['lowertail'] = (df['open'] / df['low']).where(df['positive'], df['close'] / df['low'])
        df['candle_count'] = df['volume'].rolling(window=self.startup_candle_count, min_periods=self.startup_candle_count).count()
        df['close_change'] = df['close'].pct_change()

        df_closechange = df['close'] - df['close'].shift(1)
        for i in (1, 2, 3):
            df['updown'] = np.where(df_closechange.rolling(window=i, min_periods=1).sum().gt(0), 1, np.where(df_closechange.rolling(window=i, min_periods=1).sum().lt(0), -1, 0))
            df[f"streak_{i}"] = df['updown'].groupby((df['updown'].ne(df['updown'].shift(1))).cumsum()).cumsum()
        df['streak_min'] = df[['streak_1', 'streak_2', 'streak_3']].min(axis=1)
        df['streak_max'] = df[['streak_1', 'streak_2', 'streak_3']].max(axis=1)
        df.drop(columns=['updown', 'streak_1', 'streak_2', 'streak_3'])

        ef = df[['close', 'uppertail', 'lowertail']].reset_index()
        for i in self.indicator_range:
            df[f"volume_{i}"] = df['volume'].rolling(window=i, min_periods=i).sum()
            df[f"mom_{i}"] = ta.MOM(df, timeperiod=i)
            df[f"mom_{i}_up"], mid, df[f"mom_{i}_low"] = ta.BBANDS(df[f"mom_{i}"], timeperiod=i, nbdevup=2.0, nbdevdn=2.0, matype=0)
            df[f"mfi_{i}"] = mfi_enh(df, length=i)
            df[f"close_corr_{i}"] = ef['index'].rolling(window=i, min_periods=i).corr(ef['close'], method='spearman')

        return df.copy()

    def populate_buy_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        if len(df) < self.startup_candle_count:
            return df

        df.loc[:, 'buy_tag'] = ''
        buy_time_periods = (13,)
        for i in buy_time_periods:
            buy_conditions = []
            buy_conditions.append(df['candle_count'].ge(self.startup_candle_count))
            buy_conditions.append(df['volume'].ge(self.min_candle_vol * 1.2))
            buy_conditions.append(df[f"volume_{i}"].ge(self.min_candle_vol * i * 0.8))
            buy_conditions.append(df['streak_min'].le(-1))
            buy_conditions.append((df[f"mom_{i}"].shift(1) / df[f"mom_{i}_low"].shift(1)).between(1.1, 1.2))
            buy_conditions.append(df[f"mfi_{i}"].shift(1).le(18))
            buy_conditions.append(df[f"close_corr_{i}"].shift(1).between(-0.95, -0.75))
            buy_conditions.append(df['close_change'].lt(0))
            buy_conditions.append(df['close_change'].gt(df['close_change'].shift(1)))

            buy = reduce(lambda x, y: x & y, buy_conditions)
            df.loc[buy, 'buy_tag'] = 'buy' + df['streak_min'].astype(str)

        df.loc[:, 'buy'] = df['buy_tag'] != ''

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
        df.loc[:, 'stop_tag'] = ''
        time_periods = range(4, 16)
        for i in time_periods:
            sell_conditions = []
            sell_conditions.append(df['candle_count'].ge(self.startup_candle_count))
            sell_conditions.append(df['volume'].ge(self.min_candle_vol * 1.4))
            sell_conditions.append(df[f"volume_{i}"].ge(self.min_candle_vol * i * 0.8))
            if i == max(time_periods):
                sell_conditions.append(df['streak_max'].ge(i))
            else:
                sell_conditions.append(df['streak_max'].eq(i))
            sell_conditions.append(df['streak_max'].ne(df['streak_min']))
            sell_conditions.append(df['streak_min'].ne(-1))
            sell_conditions.append(df[f"close_corr_{i}"].between(0.80 - i / 100, 0.90))
            sell_conditions.append(df[f"close_corr_{i-1}"].lt(df[f"close_corr_{i}"]))

            sell = reduce(lambda x, y: x & y, sell_conditions)
            df.loc[sell, 'sell_tag'] = 'sell+' + df['streak_max'].astype(str)

        time_periods = range(3, 16)
        for i in time_periods:
            sell_conditions = []
            sell_conditions.append(df['candle_count'].ge(self.startup_candle_count))
            sell_conditions.append(df['volume'].ge(self.min_candle_vol * 1.4))
            sell_conditions.append(df[f"volume_{i}"].ge(self.min_candle_vol * i * 0.8))
            if i == max(time_periods):
                sell_conditions.append(df['streak_max'].ge(i))
            else:
                sell_conditions.append(df['streak_max'].eq(i))
            sell_conditions.append(df[f"close_corr_{i}"].gt(0.90))
            sell_conditions.append(df[f"close_corr_{i-1}"].lt(df[f"close_corr_{i}"]))
            sell_conditions.append(df['uppertail'].ge(1.02))

            sell = reduce(lambda x, y: x & y, sell_conditions)
            df.loc[sell, 'sell_tag'] = 'tail_sell+' + df['streak_max'].astype(str)

        for i in time_periods:
            sell_conditions = []
            sell_conditions.append(df['candle_count'].ge(self.startup_candle_count))
            sell_conditions.append(df['volume'].ge(self.min_candle_vol * 1.2))
            if i == max(time_periods):
                sell_conditions.append(df['streak_min'].le(-i))
            else:
                sell_conditions.append(df['streak_min'].eq(-i))
            sell_conditions.append(df[f"close_corr_{i}"].le(-0.50))
            sell_conditions.append(df[f"close_corr_{i-1}"].lt(df[f"close_corr_{i}"]))
            sell_conditions.append(df['close_change'].le(-0.01))

            sell = reduce(lambda x, y: x & y, sell_conditions)
            df.loc[sell, 'sell_tag'] = 'sell' + df['streak_min'].astype(str)

        for i in time_periods:
            sell_conditions = []
            sell_conditions.append(df['candle_count'].ge(self.startup_candle_count))
            sell_conditions.append(df['volume'].ge(self.min_candle_vol * 1.2))
            if i == min(time_periods):
                sell_conditions.append(df['streak_min'].between(-i, -2))
            if i == max(time_periods):
                sell_conditions.append(df['streak_min'].le(-i))
            else:
                sell_conditions.append(df['streak_min'].eq(-i))
            sell_conditions.append(df[f"close_corr_{i}"].le(0))
            sell_conditions.append(df[f"close_corr_{i-1}"].le(df[f"close_corr_{i}"]))
            sell_conditions.append(df['close_change'].le(0))

            sell = reduce(lambda x, y: x & y, sell_conditions)
            df.loc[sell, 'stop_tag'] = 'stop' + df['streak_min'].astype(str)

        df.loc[:, 'sell'] = False
        return df

    def custom_sell(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        candle_1 = df.iloc[-1]
        sell_tag, stop_tag, current_rate = candle_1['sell_tag'], candle_1['stop_tag'], candle_1['close']
        current_profit = (current_rate - trade.open_rate) / trade.open_rate
        if (sell_tag == '' and stop_tag == '') or (-0.04 <= current_profit <= 0.015):
            return None

        if hasattr(trade, 'buy_tag') and trade.buy_tag is not None:
            buy_tag = trade.buy_tag
        trade_open_date = timeframe_to_prev_date(self.timeframe, trade.open_date_utc)
        df_trade: DataFrame = df[df['date'] >= trade_open_date]
        candles_between = df_trade.index[-1] - df_trade.index[0]
        if current_profit < -0.04 and 'stop' in stop_tag:
            if candle_1['buy']:
                log.info(f"custom_sell: stop for pair {pair} with profit {current_profit}, stop_tag {stop_tag} and buy_tag {buy_tag} on candle {candle_1['date']} is cancelled because of buy signal {candle_1['buy_tag']}.")
                return None
            else:
                n = int("".join(filter(str.isdigit, stop_tag)))
                if n <= candles_between + 1:
                    self.stoploss_count += 1
                    log.info(f"custom_sell: stoploss # {self.stoploss_count} for pair {pair} with loss {round(current_profit, 2)}, stop_tag {stop_tag} and buy_tag {buy_tag} on candle {candle_1['date']}.")
                    return f"{stop_tag} ({buy_tag})"

        if current_profit > 0.015 and 'sell' in sell_tag:
            if candle_1['buy']:
                log.info(f"custom_sell: sell for pair {pair} with profit {current_profit}, sell_tag {sell_tag} and buy_tag {buy_tag} on candle {candle_1['date']} is cancelled.")
                return None
            else:
                n = int("".join(filter(str.isdigit, sell_tag)))
                if n <= candles_between:
                    return f"{sell_tag} ({buy_tag})"

        return None

    def bot_loop_start(self, **kwargs) -> None:
        self.min_candle_vol = self.config['stake_amount']
    
        if self.config['runmode'].value not in ('live', 'dry_run'):
            if self.write_to_txt:
                with open(self.f_buys, 'w') as f:
                    print('pair;date open;trade open rate;buy tags;close;mfi;cmf;close_corr_', file=f)
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
                indicator.append(candle_1[f"mfi_{period}"])
                indicator.append(candle_1[f"close_corr_{period}"])
                indicator.append(candle_2[f"close_corr_{period}"])
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

def cmf_enh(df: DataFrame, length: int) -> Series:
    max_tail = 1.05
    hi = df['high'].where(df['uppertail'] <= max_tail, max_tail * df['close'].where(df['positive'], df['open']))
    lo = df['low'].where(df['lowertail'] <= max_tail, df['open'].where(df['positive'], df['close']) / max_tail)
    mfv = ((df['close'] - lo) - (hi - df['close'])) / (hi - lo) * df['volume']

    return mfv.rolling(window=length, min_periods=length).sum() / df['volume'].rolling(window=length, min_periods=length).sum()

def mfi_enh(df: DataFrame, length: int) -> Series:
    max_tail = 1.05
    hi = df['high'].where(df['uppertail'] <= max_tail, max_tail * df['close'].where(df['positive'], df['open']))
    lo = df['low'].where(df['lowertail'] <= max_tail, df['open'].where(df['positive'], df['close']) / max_tail)
    hlc3 = (hi + lo + df['close']) / 3
    mf: Series = hlc3 * df['volume']
    mfp = mf.where(hlc3.gt(hlc3.shift(1)), 0).rolling(window=length, min_periods=length).sum()
    mfn = mf.where(hlc3.lt(hlc3.shift(1)), 0).rolling(window=length, min_periods=length).sum()

    return 100 * (mfp / (mfp + mfn))


class CR1(CR):
    b = __name__.lower()
    if b not in ('cr'):
    # to check each time_period use confirm_trade_entry
        buy_period = int("".join(filter(str.isdigit, b)))

class ViN3(CR1):
    pass
