# 4章 The Cross-Entropy Method

> 実装パートの[@akky_eeic](https://twitter.com/akky_eeic)です。
> 実装なのですが、何も実装しておりません。
> フォークしたので許してください...

クロスエントロピー法の章ですが、ドキュメント[(torch.nn.CrossEntropyLoss)](https://pytorch.org/docs/stable/nn.html#torch.nn.CrossEntropyLoss)を見てもらうと、

> It is useful when training a classification problem with C classes.


と書いてあるように、クラス分類のタスクにおいてクロスエントロピー法は有効な方法となっています。
KLダイバージェンスとかいろいろありますよね...

とりあえず、実装を見てお気持ちを理解していきましょう。

##  Cross-Entropy on CartPole

実装は[`Chapter04/01_cartpole.py`](https://github.com/deep-rl-seminar/deep-rl-seminar/blob/Akky/Akky/chapter04/01_cartpole.py)にあります。
追っていきましょう。

既知の方もいるかもしれませんが、復習も兼ねてDNNを見ていきましょう。
```
class Net(nn.Module):
    def __init__(self, obs_size, hidden_size, n_actions):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )

    def forward(self, x):
        return self.net(x)
```

`__main__()`関数で`net = Net(obs_size, HIDDEN_SIZE, n_actions)`を呼んでいるんですが、CartPole-v0において、

- obs_sizeは obs = [カート位置, カート速度, ポール角度(rad), ポール角速度]で4
- n_actionsは action =[左, 右]の2値なので2

となっています。(actionは0なら-1, 1なら+1に動きます)
で、`net.forward(x)`を呼ぶことで、
inputを $\boldsymbol{x}$, outputを $\boldsymbol{y}$ とすると

- `nn.Linear`を $\boldsymbol{W_1}$
- `nn.ReLU()`を $f$
- `nn.Linear`を $\boldsymbol{W_2}$

とすると、
$$ \boldsymbol{y} = \boldsymbol{W_2}(f(\boldsymbol{W_1x}))$$
という感じになっています。
この出力を確率分布として扱いたいので、**softmax**をかけていきます。
このネットワークをうまく学習するために、トレーニングにはロス関数として**Cross-Entropy Loss**を用います。

ここまででとりあえず28行目まではOKでしょう。


次に、`iterate_batches()`を見ていくのですが、...
44行目で`action() = np.random.choice(len(act_probs), p=act_probs)`
によって、 現在のobservationの状態 $s$, $a$ から $s'$ を推定するときに、確率過程を仮定している(?)っぽい。
これぐらいで大丈夫でしょう。

あとは、main文で学習しています。

終了です。

## Cross-Entropy on FrozenLake

実装は[`chapter04/02_frozenlake_native.py`](https://github.com/deep-rl-seminar/deep-rl-seminar/blob/Akky/Akky/chapter04/02_frozenlake_naive.py)にあります。
追っていきましょう。

FrozenLakeの実装は、上記のものとほとんど変えずに行えます。
[diff](https://github.com/deep-rl-seminar/deep-rl-seminar/blob/Akky/Akky/chapter04/diff_01cartpole_02frozenlake.txt)を見てもらうとわかるんですが、ほぼ変わってません。
強いて言えば、

- actionが上下左右の4つあるので、ワンホットベクトルで扱う。

ということぐらいですね。

で、これを走らせてみると、まぁうまく行きません。
微調整を加えることで、タスクを溶けるようにがんばります。[chapter04/03_frozenlake_tweaked.py](https://github.com/deep-rl-seminar/deep-rl-seminar/blob/Akky/Akky/chapter04/03_frozenlake_tweaked.py)

[diff](https://github.com/deep-rl-seminar/deep-rl-seminar/blob/Akky/Akky/chapter04/diff_02_03.txt)を取ってみるとわかるのですが、めっちゃ変わってて追うのがアレなので、かいつまんで話すと、

- バッチサイズを増やす
- 割引率を適用
- eliteなエピソードを長期間残しておく
- 学習率を下げる
- たくさん学習時間をかける

ということが書いてあります。

確率的な状態遷移が発生しないような`Chapter04/04_frozenlake_nonslippery.py`にもこの微調整を適用すると、めっちゃすごいです。

以上になります。
