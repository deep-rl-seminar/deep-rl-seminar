import gym
import gym.spaces
from gym.utils import seeding
import enum
import numpy as np

from . import data

DEFAULT_BARS_COUNT = 10
DEFAULT_COMMISSION_PERC = 0.1


class Actions(enum.Enum):
    # 行動パターン
    Skip = 0
    Buy = 1
    Close = 2


class State:
    # 初期化
    def __init__(self, bars_count, commission_perc, reset_on_close, reward_on_close=True, volumes=True):
        assert isinstance(bars_count, int)
        assert bars_count > 0
        assert isinstance(commission_perc, float)
        assert commission_perc >= 0.0
        assert isinstance(reset_on_close, bool)
        assert isinstance(reward_on_close, bool)
        self.bars_count = bars_count
        self.commission_perc = commission_perc
        self.reset_on_close = reset_on_close
        self.reward_on_close = reward_on_close
        self.volumes = volumes

    # pricesとoffsetを保存
    def reset(self, prices, offset):
        assert isinstance(prices, data.Prices)
        assert offset >= self.bars_count-1
        self.have_position = False
        self.open_price = 0.0
        self._prices = prices
        self._offset = offset

    # high, low, closeのデータのサイズを返す
    @property
    def shape(self):
        # [h, l, c] * bars + position_flag + rel_profit (since open)
        if self.volumes:
            return (4 * self.bars_count + 1 + 1, )
        else:
            return (3*self.bars_count + 1 + 1, )

    # 列ベクトルの形で過去の時刻でのデータを取得する。resetとstopから呼ばれる。
    # bars_countがtrain_modelでは10, train_model_convでは50となる。train_modelからしか呼ばれない。
    def encode(self):
        """
        Convert current state into numpy array.
        """
        res = np.ndarray(shape=self.shape, dtype=np.float32)
        shift = 0
        for bar_idx in range(-self.bars_count+1, 1):
            # high, low, closeを入れる
            res[shift] = self._prices.high[self._offset + bar_idx]
            shift += 1
            res[shift] = self._prices.low[self._offset + bar_idx]
            shift += 1
            res[shift] = self._prices.close[self._offset + bar_idx]
            shift += 1
            # volume
            if self.volumes:
                res[shift] = self._prices.volume[self._offset + bar_idx]
                shift += 1
        # position
        res[shift] = float(self.have_position)
        shift += 1
        # profit
        if not self.have_position:
            res[shift] = 0.0
        else:
            res[shift] = (self._cur_close() - self.open_price) / self.open_price
        return res

    # 現在のcloseを計算
    # data.py の load_relative() では closeの値をopenとの差に対する比率にしているので計算。
    def _cur_close(self):
        """
        Calculate real close price for the current bar
        """
        open = self._prices.open[self._offset]
        rel_close = self._prices.close[self._offset]
        return open * (1.0 + rel_close)

    # 各ステップごとの処理
    def step(self, action):
        """
        Perform one step in our price, adjust offset, check for the end of prices
        and handle position change
        :param action:
        :return: reward, done
        """
        assert isinstance(action, Actions)
        reward = 0.0
        done = False # エピソード終了かどうか
        close = self._cur_close()
        # 手数料を支払って買う
        if action == Actions.Buy and not self.have_position:
            self.have_position = True
            self.open_price = close
            reward -= self.commission_perc
        # 手数料を払って売る
        elif action == Actions.Close and self.have_position:
            reward -= self.commission_perc
            done |= self.reset_on_close
            # Trueに設定してある。報酬の計算。
            if self.reward_on_close:
                reward += 100.0 * (close - self.open_price) / self.open_price
            self.have_position = False
            self.open_price = 0.0

        self._offset += 1
        prev_close = close
        close = self._cur_close()
        done |= self._offset >= self._prices.close.shape[0]-1
        # 売る時、終値に対して報酬を計算する。
        if self.have_position and not self.reward_on_close:
            reward += 100.0 * (close - prev_close) / prev_close

        return reward, done

# train_model_convで利用される。
class State1D(State):
    # データをmatrixで初期化
    """
    State with shape suitable for 1D convolution
    """
    @property
    def shape(self):
        if self.volumes:
            return (6, self.bars_count)
        else:
            return (5, self.bars_count)

    # Stateとは異なり、matrixの形でデータを保持する
    # train_model_convで利用され、vars_count = 50
    def encode(self):
        res = np.zeros(shape=self.shape, dtype=np.float32)
        ofs = self.bars_count-1
        res[0] = self._prices.high[self._offset-ofs:self._offset+1]
        res[1] = self._prices.low[self._offset-ofs:self._offset+1]
        res[2] = self._prices.close[self._offset-ofs:self._offset+1]
        if self.volumes:
            res[3] = self._prices.volume[self._offset-ofs:self._offset+1]
            dst = 4
        else:
            dst = 3
        if self.have_position:
            res[dst] = 1.0
            res[dst+1] = (self._cur_close() - self.open_price) / self.open_price
        return res


class StocksEnv(gym.Env):
    # gym.Envのために必要
    metadata = {'render.modes': ['human']}

    # prices : { "[open|high|low|close]", "値段" }
    # bars_count : observationを通過したbarの数。train_modelでは10、train_model_convでは50
    # commission : 手数料。default : 0.1%
    # reset_on_close : True  >> 売ってエピソードを終了するかどうかを毎回Agantが選択できる。
    #                  False >> そのまま年の終わりまで続く
    # conv_1d :  state_1dのモデルを利用。
    #            True  >> 一行にHighやLowのデータがある、2次元的なMatrixデータ。
    #            False >> 一列に[High,Low,Close,Volume]がたくさんある、ベクトル的なデータ。
    # random_ofs_on_reset : 途中の時刻から始まるかどうか
    # reward_on_close : closeしたときの報酬だけを受け取るかどうか
    # volumes : observationを切り替えるかどうか
    def __init__(self, prices, bars_count=DEFAULT_BARS_COUNT,
                 commission=DEFAULT_COMMISSION_PERC, reset_on_close=True, state_1d=False,
                 random_ofs_on_reset=True, reward_on_close=False, volumes=False):
        assert isinstance(prices, dict)
        self._prices = prices
        if state_1d:
            self._state = State1D(bars_count, commission, reset_on_close, reward_on_close=reward_on_close,
                                  volumes=volumes)
        else:
            self._state = State(bars_count, commission, reset_on_close, reward_on_close=reward_on_close,
                                volumes=volumes)
        self.action_space = gym.spaces.Discrete(n=len(Actions))
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=self._state.shape, dtype=np.float32)
        self.random_ofs_on_reset = random_ofs_on_reset
        self.seed()

    # offsetから再開する
    def reset(self):
        # make selection of the instrument and it's offset. Then reset the state
        self._instrument = self.np_random.choice(list(self._prices.keys()))
        prices = self._prices[self._instrument]
        bars = self._state.bars_count
        if self.random_ofs_on_reset:
            offset = self.np_random.choice(prices.high.shape[0]-bars*10) + bars
        else:
            offset = bars
        self._state.reset(prices, offset)
        return self._state.encode()

    # state クラスのラッパー。 observation, reward, 終了フラグ, オフセットを返す。
    def step(self, action_idx):
        action = Actions(action_idx)
        reward, done = self._state.step(action)
        obs = self._state.encode()
        info = {"instrument": self._instrument, "offset": self._state._offset}
        return obs, reward, done, info

    # デバッグ用のhandeler
    def render(self, mode='human', close=False):
        pass

    def close(self):
        pass

    # envをたくさん生成した時のseedが同じにならないように設定
    def seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed)
        seed2 = seeding.hash_seed(seed1 + 1) % 2 ** 31
        return [seed1, seed2]

    # CSVデータのあるオブジェクトを取得
    @classmethod
    def from_dir(cls, data_dir, **kwargs):
        prices = {file: data.load_relative(file) for file in data.price_files(data_dir)}
        return StocksEnv(prices, **kwargs)
