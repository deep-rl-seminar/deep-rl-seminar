
# 強化学習ゼミ 5章 実装パート

## 自己紹介
---
* 川島 丸生
* proton( @proton_1602) <font color="Gainsboro">ちなみに、1602は電気素量の上4桁ですよ。</font>
* 研究室は、AY, trok, nemr, mnmt, kwhr当たり？まだ未定。仮第一はAY。
* 機械学習は去年の春のゼミ以降ちょこちょこやってたりする。バイトも一応機械学習関係
* 強化学習は何もやってこなかったので初めてです。間違ってる、怪しい所がある時はちゃんと指摘してほしい。

## 5章のメモ
<font color="Gainsboro">英弱がわざわざ英語の本を何度も必要ない所まで読み返さなくて済むように書いたメモ</font>

***
5章はp122~141(p99~118)  
* FrozenLakeでの例(p122~124),常に最善の報酬を得る貪欲法だとうまく行かないときもあるよっていう、ただの導入
* ベルマン方程式(p125~127), 将来の即時報酬和を再帰的な式を利用して過去の即時報酬を使うことで代用and高速化?, 確率を導入することで確率的な報酬等に対応できるよ、すごいね  
    * p125: 決定的方策, p126~127:確率的方策の導入、一般的なベルマン方程式, p127:(状態)行動価値Qの導入
    * p125: $ V_0(a=a_i)$はなんか状態価値のくせに方策引数?にしてて変な感じだが、決定的方策$\pi$を取った時の$V_\pi(s_0)$と同じ?
    * p126: $S$は$s_0$から遷移可能な状態集合、$r_{s,a}$は行動aにより状態sに遷移した時の即時報酬、$V_s$は状態sの状態価値。$A$は状態$s_0$から取りうる行動集合  
* 簡単な例によるQ学習の説明,価値反復法(p128~129), 行動に対する状態遷移を確率的にしてる。
    * p129: Q値,行動の評価を学習して、行動の評価が最大になる行動を選択する方策を取るValueベース。遷移確率とかが既知な完全なモデルベースは珍しいので、予測していく必要もあるよ。
* ループ等を含む一般的なマルコフ過程におけるV,Qの求め方(p129~131), この方法は状態空間が離散的で十分小さい必要があるが、ニューラルネットワークを用いたQ学習を用いるとなんとかなる(後章)、遷移確率を普通は知らないからある状態推移と行動時の報酬を覚えておく必要もある。
    1. $V_i, Q_{s,a}$の初期化(0)
    2. 全ての状態$s$についての状態価値$V_s$、全ての状態$s$,行動$a$についての行動価値$Q_{s,a}$をベルマン方程式で更新  
    $V_s \leftarrow \max_\alpha \sum_{s'}p_{a,s\rightarrow s'}(r_{s,a}+\gamma V_{s'})$  
    $Q_{s,a} \leftarrow \sum_{s'}p_{a,s\rightarrow s'}(r_{s,a}+\gamma \max Q_{s',a'})$
    3. 十分な回数か差分が一定値以下になるまで2をループ 
* FrozenLakeでの例(p132~139)(コード有)
    * 01_frozenlake_v_iteration.py p132~137: 価値反復法、クロスエントロピー法に比べてエピソードが完全に終了する必要がなく、各施行で推定,更新ができるので効率的で収束が早い。FrozenLakeは終了時のみ報酬が入るのでマップが広いとちょっと時間かかる。"FrozenLake8x8-v0"に変更することで比較したのが図9
    * 02_frozenlake_q_iteration.py p138~139: Q学習、01では状態価値Vを表に持っていたが、行動価値Qをテーブルに持つ。これによりメモリの使用量は増えるが更新が分かりやすくなる。Vの更新よりも、Qの更新のほうが一般的ではあるが、Actor-Critic法等ではVの更新を用いる。
* まとめ(p140), RLで重要なV,Q,ベルマン方程式と値反復による改善を学んだ。次はディープQネットワーク

## コードと解説(?)

## FrozenLakeのルール
---
* 4x4の盤面を移動する．(FrozenLake8x8-v0では8x8)
* Sが開始地点で，Gがゴール．
* Hが落とし穴でゲーム失敗で，Fは床で移動できる．
* 隣接4方向に移動可能
* 現在の位置とゲームオーバーかどうかが分かる．
* 意図した方向に進める確率は1/3で，それ以外だと90度直角に進む．
* env.reset()してもマップは変わらない．

### 01_frozenlake_v_iteration.py
---


```python
#01_frozenlake_v_iteration.py
import gym
import collections
from tensorboardX import SummaryWriter

ENV_NAME = "FrozenLake-v0"
#ENV_NAME = "FrozenLake8x8-v0"
GAMMA = 0.9
TEST_EPISODES = 20


#Agentクラスの定義、読めばだいたいわかる
class Agent:
    def __init__(self):
        self.env = gym.make(ENV_NAME)
        self.state = self.env.reset()
        self.rewards = collections.defaultdict(float)
        self.transits = collections.defaultdict(collections.Counter)
        self.values = collections.defaultdict(float)

    def play_n_random_steps(self, count):
        #count回だけ遷移する、途中で終了すれば初期化して再開
        #クロスエントロピー法と異なりエピソードが終了しなくともよい
        for _ in range(count):
            #action_space.sample()で可能な行動をランダムに返してくれる
            action = self.env.action_space.sample()
            
            #stepメソッドに行動を渡すと、状態,報酬,終了判定,デバッグ用情報を返す。
            #ex: observation,reward,done,info=env.step(action)
            new_state, reward, is_done, _ = self.env.step(action)
            
            #報酬表(Reward table)の更新、keyは状態,行動,次状態でvalueは即時報酬
            self.rewards[(self.state, action, new_state)] = reward
            
            #遷移表(Transitions table)の更新
            #ある状態sと行動aにより状態s'に行った回数を記録することで、
            #未知の遷移確率を推定する。
            self.transits[(self.state, action)][new_state] += 1
            
            #終了していた場合はresetして初期状態のobservationを返す
            #そうじゃないならstepにより得られた新状態を入れる。
            self.state = self.env.reset() if is_done else new_state

    #遷移表で遷移確率を近似して行動価値Qを計算する。
    #計算には状態価値Vを用いる。
    def calc_action_value(self, state, action):
        target_counts = self.transits[(state, action)]
        total = sum(target_counts.values())
        action_value = 0.0
        for tgt_state, count in target_counts.items():
            reward = self.rewards[(state, action, tgt_state)]
            action_value += (count / total) * (reward + GAMMA * self.values[tgt_state])
        return action_value

    #行動価値Qが最も高くなる行動aを返す
    def select_action(self, state):
        best_action, best_value = None, None
        for action in range(self.env.action_space.n):
            action_value = self.calc_action_value(state, action)
            if best_value is None or best_value < action_value:
                best_value = action_value
                best_action = action
        return best_action

    #推定された行動価値Qに基づいた方策πで
    #1エピソード実行し累計報酬を返す
    #学習後の描画をしたいので少し変更してます
    def play_episode(self, env,moniter=0):
        total_reward = 0.0
        state = env.reset()
        cnt = 0
        while True:
            cnt += 1
            action = self.select_action(state)
            new_state, reward, is_done, _ = env.step(action)
            self.rewards[(state, action, new_state)] = reward
            self.transits[(state, action)][new_state] += 1
            total_reward += reward
            if moniter:
                env.render()
            if is_done:
                break
            state = new_state
        if moniter:
            return total_reward, cnt
        else: 
            return total_reward

    #方策はQが最大になる行動として、状態価値Vの表を更新する。
    #行動価値Qを用いているが展開すればVのベルマン最適方程式になってる
    def value_iteration(self):
        for state in range(self.env.observation_space.n):
            state_values = [self.calc_action_value(state, action)
                            for action in range(self.env.action_space.n)]
            self.values[state] = max(state_values)


if __name__ == "__main__":
    test_env = gym.make(ENV_NAME)
    agent = Agent()
    writer = SummaryWriter(comment="-v-iteration")

    iter_no = 0
    best_reward = 0.0
    while True:
        iter_no += 1
        #100回ランダムに動いて探索、表埋め
        agent.play_n_random_steps(100)
        agent.value_iteration()

        #TEST_EPISODES回エピソードを実行して平均を取る
        #平均が一定値を超えていたら学習終了
        reward = 0.0
        for _ in range(TEST_EPISODES):
            reward += agent.play_episode(test_env)
        reward /= TEST_EPISODES
        writer.add_scalar("reward", reward, iter_no)
        if reward > best_reward:
            print("Best reward updated %.3f -> %.3f" % (best_reward, reward))
            best_reward = reward
        if reward > 0.80:
            print("Solved in %d iterations!" % iter_no)
            break
    writer.close()

```

    Best reward updated 0.000 -> 0.150
    Best reward updated 0.150 -> 0.200
    Best reward updated 0.200 -> 0.300
    Best reward updated 0.300 -> 0.450
    Best reward updated 0.450 -> 0.600
    Best reward updated 0.600 -> 0.750
    Best reward updated 0.750 -> 0.800
    Best reward updated 0.800 -> 0.900
    Solved in 14 iterations!


#### 学習前と学習後の比較
---
* 実行するとランダムな方策と学習後の方策での、1エピソードの動きが見れます。
(環境構築済みのpcでやらないとエラーが出て履歴が消えちゃうよ)
* 緑本みたくVテーブルの図示とかやりたかったけど余りやる意味を感じなかった。暇が有ったらやるかもしれない


```python
test_env.reset()
cnt = 0
total_reward = 0.0
while True:
    cnt += 1
    action=test_env.action_space.sample()
    observation,reward,done,info = test_env.step(action)
    total_reward += reward
    test_env.render()
    if done:
        print("---randam_policy(1 episode)---")
        print("play: {}, total_reward:{}".format(cnt,total_reward))
        break
```

      (Left)
    [41mS[0mFFF
    FHFH
    FFFH
    HFFG
      (Left)
    [41mS[0mFFF
    FHFH
    FFFH
    HFFG
      (Down)
    SFFF
    [41mF[0mHFH
    FFFH
    HFFG
      (Up)
    SFFF
    F[41mH[0mFH
    FFFH
    HFFG
    ---randam_policy(1 episode)---
    play: 4, total_reward:0.0



```python
state = test_env.reset()
total_reward, cnt = agent.play_episode(test_env,moniter=1)
print("---opt_policy(1 episode)---")
print("play: {}, total_reward:{}".format(iter_no,total_reward))
```

      (Left)
    [41mS[0mFFF
    FHFH
    FFFH
    HFFG
      (Left)
    [41mS[0mFFF
    FHFH
    FFFH
    HFFG
      (Left)
    SFFF
    [41mF[0mHFH
    FFFH
    HFFG
      (Left)
    [41mS[0mFFF
    FHFH
    FFFH
    HFFG
      (Left)
    [41mS[0mFFF
    FHFH
    FFFH
    HFFG
      (Left)
    [41mS[0mFFF
    FHFH
    FFFH
    HFFG
      (Left)
    [41mS[0mFFF
    FHFH
    FFFH
    HFFG
      (Left)
    [41mS[0mFFF
    FHFH
    FFFH
    HFFG
      (Left)
    SFFF
    [41mF[0mHFH
    FFFH
    HFFG
      (Left)
    [41mS[0mFFF
    FHFH
    FFFH
    HFFG
      (Left)
    [41mS[0mFFF
    FHFH
    FFFH
    HFFG
      (Left)
    [41mS[0mFFF
    FHFH
    FFFH
    HFFG
      (Left)
    [41mS[0mFFF
    FHFH
    FFFH
    HFFG
      (Left)
    SFFF
    [41mF[0mHFH
    FFFH
    HFFG
      (Left)
    SFFF
    FHFH
    [41mF[0mFFH
    HFFG
      (Up)
    SFFF
    FHFH
    F[41mF[0mFH
    HFFG
      (Down)
    SFFF
    FHFH
    FF[41mF[0mH
    HFFG
      (Left)
    SFFF
    FHFH
    FFFH
    HF[41mF[0mG
      (Down)
    SFFF
    FHFH
    FFFH
    HFF[41mG[0m
    ---opt_policy(1 episode)---
    play: 14, total_reward:1.0


### 02_frozenlake_q_iteration.py
---


```python
#02_frozenlake_q_iteration.py
import gym
import collections
from tensorboardX import SummaryWriter

ENV_NAME = "FrozenLake-v0"
#ENV_NAME = "FrozenLake8x8-v0"
GAMMA = 0.9
TEST_EPISODES = 20


#diff取ればわかるけど01とほとんど違いない
class Agent:
    def __init__(self):
        self.env = gym.make(ENV_NAME)
        self.state = self.env.reset()
        self.rewards = collections.defaultdict(float)
        self.transits = collections.defaultdict(collections.Counter)
        self.values = collections.defaultdict(float)

    def play_n_random_steps(self, count):
        for _ in range(count):
            action = self.env.action_space.sample()
            new_state, reward, is_done, _ = self.env.step(action)
            self.rewards[(self.state, action, new_state)] = reward
            self.transits[(self.state, action)][new_state] += 1
            self.state = self.env.reset() if is_done else new_state

    #calc_action_value 消去
            
    def select_action(self, state):
        best_action, best_value = None, None
        for action in range(self.env.action_space.n):
            #直接Qテーブルを保持しているのでcalc_action_valueは必要ない　
            action_value = self.values[(state, action)]
            if best_value is None or best_value < action_value:
                best_value = action_value
                best_action = action
        return best_action

        #学習後の描画をしたいので少し変更してます
    def play_episode(self, env,moniter=0):
        total_reward = 0.0
        state = env.reset()
        cnt = 0
        while True:
            cnt += 1
            action = self.select_action(state)
            new_state, reward, is_done, _ = env.step(action)
            self.rewards[(state, action, new_state)] = reward
            self.transits[(state, action)][new_state] += 1
            total_reward += reward
            if moniter:
                env.render()
            if is_done:
                break
            state = new_state
        if moniter:
            return total_reward, cnt
        else: 
            return total_reward
    
    #Qテーブル更新
    def value_iteration(self):
        for state in range(self.env.observation_space.n):
            for action in range(self.env.action_space.n):
                action_value = 0.0
                target_counts = self.transits[(state, action)]
                total = sum(target_counts.values())
                for tgt_state, count in target_counts.items():
                    reward = self.rewards[(state, action, tgt_state)]
                    best_action = self.select_action(tgt_state)
                    action_value += (count / total) * (reward + GAMMA * self.values[(tgt_state, best_action)])
                self.values[(state, action)] = action_value


if __name__ == "__main__":
    test_env = gym.make(ENV_NAME)
    agent = Agent()
    writer = SummaryWriter(comment="-q-iteration")

    iter_no = 0
    best_reward = 0.0
    while True:
        iter_no += 1
        agent.play_n_random_steps(100)
        agent.value_iteration()

        reward = 0.0
        for _ in range(TEST_EPISODES):
            reward += agent.play_episode(test_env)
        reward /= TEST_EPISODES
        writer.add_scalar("reward", reward, iter_no)
        if reward > best_reward:
            print("Best reward updated %.3f -> %.3f" % (best_reward, reward))
            best_reward = reward
        if reward > 0.80:
            print("Solved in %d iterations!" % iter_no)
            break
    writer.close()

```

    Best reward updated 0.000 -> 0.250
    Best reward updated 0.250 -> 0.600
    Best reward updated 0.600 -> 0.650
    Best reward updated 0.650 -> 0.700
    Best reward updated 0.700 -> 0.850
    Solved in 15 iterations!


#### 学習前と学習後の比較
---
* 実行するとランダムな方策と学習後の方策での、1エピソードの動きが見れます。
(環境構築済みのpcでやらないとエラーが出て履歴が消えちゃうよ)


```python
test_env.reset()
cnt = 0
total_reward = 0.0
while True:
    cnt += 1
    action=test_env.action_space.sample()
    observation,reward,done,info = test_env.step(action)
    total_reward += reward
    test_env.render()
    if done:
        print("---randam_policy(1 episode)---")
        print("play: {}, total_reward:{}".format(cnt,total_reward))
        break
```

      (Right)
    S[41mF[0mFF
    FHFH
    FFFH
    HFFG
      (Up)
    S[41mF[0mFF
    FHFH
    FFFH
    HFFG
      (Left)
    [41mS[0mFFF
    FHFH
    FFFH
    HFFG
      (Left)
    [41mS[0mFFF
    FHFH
    FFFH
    HFFG
      (Right)
    SFFF
    [41mF[0mHFH
    FFFH
    HFFG
      (Right)
    SFFF
    F[41mH[0mFH
    FFFH
    HFFG
    ---randam_policy(1 episode)---
    play: 6, total_reward:0.0



```python
state = test_env.reset()
total_reward, cnt = agent.play_episode(test_env,moniter=1)
print("---opt_policy(1 episode)---")
print("play: {}, total_reward:{}".format(iter_no,total_reward))
```

      (Left)
    [41mS[0mFFF
    FHFH
    FFFH
    HFFG
      (Left)
    [41mS[0mFFF
    FHFH
    FFFH
    HFFG
      (Left)
    [41mS[0mFFF
    FHFH
    FFFH
    HFFG
      (Left)
    SFFF
    [41mF[0mHFH
    FFFH
    HFFG
      (Left)
    [41mS[0mFFF
    FHFH
    FFFH
    HFFG
      (Left)
    [41mS[0mFFF
    FHFH
    FFFH
    HFFG
      (Left)
    [41mS[0mFFF
    FHFH
    FFFH
    HFFG
      (Left)
    SFFF
    [41mF[0mHFH
    FFFH
    HFFG
      (Left)
    SFFF
    [41mF[0mHFH
    FFFH
    HFFG
      (Left)
    [41mS[0mFFF
    FHFH
    FFFH
    HFFG
      (Left)
    [41mS[0mFFF
    FHFH
    FFFH
    HFFG
      (Left)
    [41mS[0mFFF
    FHFH
    FFFH
    HFFG
      (Left)
    [41mS[0mFFF
    FHFH
    FFFH
    HFFG
      (Left)
    SFFF
    [41mF[0mHFH
    FFFH
    HFFG
      (Left)
    SFFF
    FHFH
    [41mF[0mFFH
    HFFG
      (Up)
    SFFF
    FHFH
    [41mF[0mFFH
    HFFG
      (Up)
    SFFF
    [41mF[0mHFH
    FFFH
    HFFG
      (Left)
    SFFF
    [41mF[0mHFH
    FFFH
    HFFG
      (Left)
    [41mS[0mFFF
    FHFH
    FFFH
    HFFG
      (Left)
    [41mS[0mFFF
    FHFH
    FFFH
    HFFG
      (Left)
    [41mS[0mFFF
    FHFH
    FFFH
    HFFG
      (Left)
    [41mS[0mFFF
    FHFH
    FFFH
    HFFG
      (Left)
    SFFF
    [41mF[0mHFH
    FFFH
    HFFG
      (Left)
    SFFF
    FHFH
    [41mF[0mFFH
    HFFG
      (Up)
    SFFF
    [41mF[0mHFH
    FFFH
    HFFG
      (Left)
    SFFF
    [41mF[0mHFH
    FFFH
    HFFG
      (Left)
    SFFF
    FHFH
    [41mF[0mFFH
    HFFG
      (Up)
    SFFF
    [41mF[0mHFH
    FFFH
    HFFG
      (Left)
    SFFF
    [41mF[0mHFH
    FFFH
    HFFG
      (Left)
    SFFF
    FHFH
    [41mF[0mFFH
    HFFG
      (Up)
    SFFF
    FHFH
    [41mF[0mFFH
    HFFG
      (Up)
    SFFF
    [41mF[0mHFH
    FFFH
    HFFG
      (Left)
    SFFF
    [41mF[0mHFH
    FFFH
    HFFG
      (Left)
    [41mS[0mFFF
    FHFH
    FFFH
    HFFG
      (Left)
    [41mS[0mFFF
    FHFH
    FFFH
    HFFG
      (Left)
    [41mS[0mFFF
    FHFH
    FFFH
    HFFG
      (Left)
    [41mS[0mFFF
    FHFH
    FFFH
    HFFG
      (Left)
    [41mS[0mFFF
    FHFH
    FFFH
    HFFG
      (Left)
    [41mS[0mFFF
    FHFH
    FFFH
    HFFG
      (Left)
    SFFF
    [41mF[0mHFH
    FFFH
    HFFG
      (Left)
    [41mS[0mFFF
    FHFH
    FFFH
    HFFG
      (Left)
    [41mS[0mFFF
    FHFH
    FFFH
    HFFG
      (Left)
    [41mS[0mFFF
    FHFH
    FFFH
    HFFG
      (Left)
    SFFF
    [41mF[0mHFH
    FFFH
    HFFG
      (Left)
    [41mS[0mFFF
    FHFH
    FFFH
    HFFG
      (Left)
    [41mS[0mFFF
    FHFH
    FFFH
    HFFG
      (Left)
    [41mS[0mFFF
    FHFH
    FFFH
    HFFG
      (Left)
    SFFF
    [41mF[0mHFH
    FFFH
    HFFG
      (Left)
    SFFF
    FHFH
    [41mF[0mFFH
    HFFG
      (Up)
    SFFF
    FHFH
    F[41mF[0mFH
    HFFG
      (Down)
    SFFF
    FHFH
    FF[41mF[0mH
    HFFG
      (Left)
    SFFF
    FHFH
    F[41mF[0mFH
    HFFG
      (Down)
    SFFF
    FHFH
    [41mF[0mFFH
    HFFG
      (Up)
    SFFF
    FHFH
    [41mF[0mFFH
    HFFG
      (Up)
    SFFF
    [41mF[0mHFH
    FFFH
    HFFG
      (Left)
    SFFF
    [41mF[0mHFH
    FFFH
    HFFG
      (Left)
    SFFF
    FHFH
    [41mF[0mFFH
    HFFG
      (Up)
    SFFF
    FHFH
    F[41mF[0mFH
    HFFG
      (Down)
    SFFF
    FHFH
    FF[41mF[0mH
    HFFG
      (Left)
    SFFF
    FHFH
    FFFH
    HF[41mF[0mG
      (Down)
    SFFF
    FHFH
    FFFH
    H[41mF[0mFG
      (Right)
    SFFF
    FHFH
    FFFH
    H[41mF[0mFG
      (Right)
    SFFF
    FHFH
    FFFH
    H[41mF[0mFG
      (Right)
    SFFF
    FHFH
    FFFH
    HF[41mF[0mG
      (Down)
    SFFF
    FHFH
    FFFH
    HF[41mF[0mG
      (Down)
    SFFF
    FHFH
    FFFH
    H[41mF[0mFG
      (Right)
    SFFF
    FHFH
    FFFH
    HF[41mF[0mG
      (Down)
    SFFF
    FHFH
    FFFH
    HF[41mF[0mG
      (Down)
    SFFF
    FHFH
    FFFH
    H[41mF[0mFG
      (Right)
    SFFF
    FHFH
    FFFH
    H[41mF[0mFG
      (Right)
    SFFF
    FHFH
    FFFH
    H[41mF[0mFG
      (Right)
    SFFF
    FHFH
    F[41mF[0mFH
    HFFG
      (Down)
    SFFF
    FHFH
    FFFH
    H[41mF[0mFG
      (Right)
    SFFF
    FHFH
    FFFH
    HF[41mF[0mG
      (Down)
    SFFF
    FHFH
    FFFH
    HF[41mF[0mG
      (Down)
    SFFF
    FHFH
    FFFH
    HFF[41mG[0m
    ---opt_policy(1 episode)---
    play: 15, total_reward:1.0


なんか追加して実装しようかと思ったけど、ここまでの範囲だとFrozenLakeみたいな問題しか解けなさそうなのでやめました。


```python

```
