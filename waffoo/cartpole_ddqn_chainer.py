import numpy as np
import gym
import argparse
import chainer
import chainerrl
import chainer.links as L
import chainer.functions as F


class QFunction(chainer.Chain):
    def __init__(self, obs_size, n_actions, n_hidden=50):
        super().__init__()
        with self.init_scope():
            self.l0 = L.Linear(obs_size, n_hidden)
            self.l1 = L.Linear(n_hidden, n_hidden)
            self.l2 = L.Linear(n_hidden, n_actions)

    def __call__(self, x, test=False):
        h = F.tanh(self.l0(x))
        h = F.tanh(self.l1(h))
        return chainerrl.action_value.DiscreteActionValue(self.l2(h))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CartPole Demo')
    parser.add_argument('-t', dest='times', type=int, default=10000)
    parser.add_argument('-f', dest='filepath', type=str, default='')
    args = parser.parse_args()

    env = gym.make('CartPole-v0')

    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n

    q_func = QFunction(obs_size, n_actions)
    # q_func.to_gpu()

    opt = chainer.optimizers.Adam(eps=1e-2)
    opt.setup(q_func)

    gamma = 0.95

    def sample():
        return np.random.randint(n_actions)

    explorer = chainerrl.explorers.LinearDecayEpsilonGreedy(
        1.0, 0.1, args.times, random_action_func=sample)

    replay_buffer = chainerrl.replay_buffer.ReplayBuffer(capacity=10**6)

    def phi(x): return x.astype(np.float32, copy=False)

    agent = chainerrl.agents.DoubleDQN(q_func, opt, replay_buffer, gamma, explorer, replay_start_size=500,
                                       update_interval=1, target_update_interval=100, phi=phi)

    if args.filepath:
        agent.load(args.filepath)

        for i in range(args.times):
            total_reward = 0
            total_step = 0
            obs = env.reset()

            done = False
            while not done:
                env.render()
                action = agent.act(obs)
                obs, reward, done, _ = env.step(action)
                total_reward += reward
                total_step += 1

            print('Episode'+str(i), 'Step:',
                  total_step, 'Reward:', total_reward)
    else:
        chainerrl.experiments.train_agent_with_evaluation(
            agent, env, steps=args.times, eval_n_runs=1, max_episode_len=200, eval_interval=1000,
            outdir='results')
