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
    startup_candle_count: int = 72

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
        s = (1, 2, 3)
        for i in s:
            df['updown'] = np.where(df_closechange.rolling(window=i, min_periods=1).sum().gt(0), 1, np.where(df_closechange.rolling(window=i, min_periods=1).sum().lt(0), -1, 0))
            df[f"streak_{i}"] = df['updown'].groupby((df['updown'].ne(df['updown'].shift(1))).cumsum()).cumsum()
        df['streak_s_min'] = df[[f"streak_{i}" for i in s]].min(axis=1)
        df['streak_s_min_change'] = df['close'] / df['close'].to_numpy()[df.index.to_numpy() + np.where(df['streak_s_min'].lt(0), df['streak_s_min'].to_numpy(), 0)]
        df['streak_s_max'] = df[[f"streak_{i}" for i in s]].max(axis=1)
        df['streak_s_max_change'] = df['close'] / df['close'].to_numpy()[df.index.to_numpy() - np.where(df['streak_s_max'].gt(0), df['streak_s_max'].to_numpy(), 0)]
        df.drop(columns=[f"streak_{i}" for i in s], inplace=True)
        i = 16
        df['updown'] = np.where(df_closechange.rolling(window=i, min_periods=1).sum().gt(0), 1, np.where(df_closechange.rolling(window=i, min_periods=1).sum().lt(0), -1, 0))
        df[f"streak_b"] = df['updown'].groupby((df['updown'].ne(df['updown'].shift(1))).cumsum()).cumsum()
        df.drop(columns=['updown'], inplace=True)

        df[f"candle_count_{self.startup_candle_count}"] = df['volume'].rolling(window=self.startup_candle_count, min_periods=self.startup_candle_count).count()

        df = self.populate_indicators_buy(df, metadata)
        df = self.populate_indicators_sell(df, metadata)
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

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                            time_in_force: str, current_time: datetime, **kwargs) -> bool:
        # df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        # candle_1: Series = df.iloc[-1]
        # buy_candle_date = candle_1['date']
        # log.info(f"confirm_trade_entry: Buy for pair {pair} with on candle {buy_candle_date} with close_corr {candle_1['close_corr_99']}")
        # try:
        #     buy_tags = self.custom_buy_info[buy_candle_date][pair]
        #     pairs = len(self.dp.current_whitelist())
        #     max_concurrent_buy_signals = int(pairs * 0.08)
        #     if  max_concurrent_buy_signals > 0:
        #         buy_info = self.custom_buy_info[buy_candle_date]
        #         buy_signal_count = buy_info['buy_signals']
        #         if buy_signal_count > max_concurrent_buy_signals:
                    # log.info(f"confirm_trade_entry: Buy for pair {pair} with buy tag {buy_tags} on candle {buy_candle_date} is cancelled. There are {buy_signal_count} concurrent buy signals (max = {max_concurrent_buy_signals}).")
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
    lookback_range = range(12, 33)

    def populate_indicators_buy(self, df: DataFrame, metadata: dict) -> DataFrame:
        ef = df['close'].reset_index()
        for i in self.lookback_range:
            j = i * 2
            df[f"pctchange_{i}"] = df['close'].pct_change(periods=i)
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
                df['volume'].ge(self.min_candle_vol),
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
        buy_tag = trade.buy_tag
        if trade_len <= 2:
            if current_profit < -0.18:
                log.info(f"custom_sell: immediate stoploss for pair {pair} with loss {current_profit:.2n} and trade len {trade_len} on candle {candle_1['date']}.")
                return f"immediate stoploss {buy_tag} ({len(buy_tag) // 3 + 1})"
            else:
                return None
        if trade_len < self.sideways_candles and -0.015 < current_profit < 0.015:
            return None
        # do not sell if price is more above lowest price compared with below highest price
        i = trade_len
        j = i // 2
        ef = df_trade[['close', 'hlc3_adj', 'volume']].reset_index()
        vwrs_sell = vwrs(ef, length=i).iat[-1]
        vwrs_sell_p = vwrs(ef, length=i-1).iat[-2]
        close_corr_i = ef['index'].rolling(window=i, min_periods=i).corr(ef['close'], method='spearman').iat[-1]
        close_corr_i_p = ef['index'].rolling(window=i-1, min_periods=i-1).corr(ef['close'], method='spearman').iat[-2]
        close_corr_j = ef['index'].rolling(window=j, min_periods=j).corr(ef['close'], method='spearman').iat[-1]
        close_corr_ij_diff = close_corr_i - close_corr_j
        close_offset = 0.06 - current_profit / 80 + i / 8000
        vwrs_offset = 100 - i / 10
        # close_change = ef['close'].iat[-1] / ef['close'].iat[-2]
        # if i == 48: # current_profit > 2.9:
            # print(f"pair: {pair} i: {i} vwrs: {vwrs_sell} vwrs prev: {vwrs_sell_p} profit: {current_profit} cc_i: {close_corr_i} cc_j: {close_corr_j} close_corr: {close_corr_ij_diff}")
# pair: BTCST/USDT i: 48 vwrs: 87.39963613666005 vwrs prev: 80.59229611672298 profit: 0.7785269709543567 cc_i: 0.8247342416653087 cc_j: 0.7820012921245723 close_corr: 0.042732949540736476
# pair: BTCST/USDT i: 296 vwrs: 70.6825062612173 vwrs prev: 68.25388902037324 profit: 3.3838174273858916 cc_i: 0.8131438028747624 cc_j: 0.7590573325077318 close_corr: 0.054086470367030626
# pair: BTCST/USDT i: 297 vwrs: 66.8376860617758 vwrs prev: 70.6825062612173 profit: 3.071576763485477 cc_i: 0.8003955956679617 cc_j: 0.7428646204322812 close_corr: 0.057530975235680515
# pair: BTCST/USDT i: 402 vwrs: 58.87263579365934 vwrs prev: 57.96015607190177 profit: 2.9683609958506225 cc_i: 0.9096999490172314 cc_j: 0.894806587637213 close_corr: 0.0148933613800184
# pair: BTCST/USDT i: 404 vwrs: 57.33083386314991 vwrs prev: 57.971860532739896 profit: 2.91753112033195 cc_i: 0.9110616191765116 cc_j: 0.8959639132515144 close_corr: 0.015097705924997262
# pair: BTCST/USDT i: 405 vwrs: 57.63673827222537 vwrs prev: 57.33083386314991 profit: 2.946058091286307 cc_i: 0.9117132094273102 cc_j: 0.8961977837248652 close_corr: 0.015515425702444907
# pair: BTCST/USDT i: 406 vwrs: 58.44885825165831 vwrs prev: 57.63673827222537 profit: 3.2022821576763483 cc_i: 0.9121376088880336 cc_j: 0.8978056188066058 close_corr: 0.014331990081427826
# pair: BTCST/USDT i: 407 vwrs: 57.77085886980963 vwrs prev: 58.44885825165831 profit: 3.109958506224066 cc_i: 0.9126771987138557 cc_j: 0.8982248159832055 close_corr: 0.014452382730650193
# pair: BTCST/USDT i: 408 vwrs: 57.29654131511428 vwrs prev: 57.77085886980963 profit: 2.978215767634855 cc_i: 0.9133129054751903 cc_j: 0.8994507195198589 close_corr: 0.013862185955331396
# pair: BTCST/USDT i: 409 vwrs: 56.92294208695377 vwrs prev: 57.29654131511428 profit: 2.941908713692946 cc_i: 0.9139604050646777 cc_j: 0.8995072192676636 close_corr: 0.014453185797014045
# pair: BTCST/USDT i: 423 vwrs: 57.12074645671926 vwrs prev: 56.89872465357725 profit: 2.930497925311203 cc_i: 0.9219235363866242 cc_j: 0.8911315350311495 close_corr: 0.030792001355474663
# pair: BTCST/USDT i: 425 vwrs: 57.11104848488262 vwrs prev: 56.96286623697871 profit: 2.9403526970954355 cc_i: 0.9230589766688897 cc_j: 0.8910238645028629 close_corr: 0.032035112166026836
# pair: BTCST/USDT i: 437 vwrs: 57.061423786491794 vwrs prev: 56.97841214040318 profit: 2.937759336099585 cc_i: 0.9291281711372972 cc_j: 0.885345915754932 close_corr: 0.04378225538236513
        if current_profit < -0.08 + i / 800 and vwrs_sell > 18 and close_corr_i < -0.8 - current_profit * 4 + i / 800 and close_corr_j < -0.4 - current_profit * 4 + i / 800:
            log.info(f"custom_sell: corr stop for pair {pair} with loss {current_profit:.2n} and trade len {trade_len} on candle {candle_1['date']}.")
            return f"corr stop {buy_tag} ({len(buy_tag) // 3 + 1})"
        if current_profit > 0.015 and vwrs_sell > vwrs_sell_p and vwrs_sell > vwrs_offset and close_corr_i < close_corr_i_p and close_corr_i < 0.9 and close_corr_j < 0.8 and -close_offset < close_corr_ij_diff < close_offset:
            log.info(f"custom_sell: corr sell for pair {pair} with profit {current_profit:.2n} and trade len {trade_len} on candle {candle_1['date']}.")
            return f"corr sell {buy_tag} ({len(buy_tag) // 3 + 1})"
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
            df['streak_s_max'].ge(3)
        ]
        sell = reduce(lambda x, y: x & y, sell_conditions)
        df.loc[sell, 'sell_tag'] = 'pct+' + df['streak_s_max'].astype(str)

        time_periods = range(3, self.lookback_candles + 1)
        for i in time_periods:
            sell_conditions = [
                df['volume'].ge(self.min_candle_vol * 1.4),
                df[f"volume_{i}"].ge(self.min_candle_vol * i * 0.8),
                df['streak_s_max'].ne(df['streak_min']),
                df['streak_min'].ne(-1),
                df[f"close_corr_{i}"].between(0.75, 0.95),
                df[f"close_corr_{i-1}"].lt(df[f"close_corr_{i}"])
            ]
            if i == max(time_periods):
                sell_conditions.append(df['streak_s_max'].ge(i))
            else:
                sell_conditions.append(df['streak_s_max'].eq(i))
            sell = reduce(lambda x, y: x & y, sell_conditions)
            df.loc[sell, 'sell_tag'] = 'corr+' + df['streak_s_max'].astype(str)

        for i in time_periods:
            sell_conditions = [
                df['volume'].ge(self.min_candle_vol * 1.4),
                df[f"volume_{i}"].ge(self.min_candle_vol * i * 0.8),
                df['streak_min'].ne(-1),
                df[f"close_corr_{i}"].gt(0.75),
                df['uppertail'].ge(1.02)
            ]
            if i == max(time_periods):
                sell_conditions.append(df['streak_s_max'].ge(i))
            else:
                sell_conditions.append(df['streak_s_max'].eq(i))
            sell = reduce(lambda x, y: x & y, sell_conditions)
            df.loc[sell, 'sell_tag'] = 'tail+' + df['streak_s_max'].astype(str)

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
