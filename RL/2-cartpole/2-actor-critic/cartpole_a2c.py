import sys
import gym
import pylab
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.losses import categorical_crossentropy, SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K

EPISODES = 1000


class Actor(Model):
    def __init__(self, num_action):
        super(Actor, self).__init__()
        self.layer1 = Dense(24, activation='relu', kernel_initializer='he_uniform')
        self.logits = Dense(num_action, activation='softmax', kernel_initializer='he_uniform')

    def call(self, state):
        x = self.layer1(state)
        logits = self.logits(x)
        return logits


class Critic(Model):
    def __init__(self):
        super(Critic, self).__init__()
        self.layer1 = Dense(24, activation='relu', kernel_initializer='he_uniform')
        self.layer2 = Dense(24, activation='relu', kernel_initializer='he_uniform')
        self.value = Dense(1, kernel_initializer='he_uniform')

    def call(self, state):
        x = self.layer1(state)
        x = self.layer2(x)
        value = self.value(x)
        return value


# 카트폴 예제에서의 액터-크리틱(A2C) 에이전트
class A2CAgent:
    def __init__(self, state_size, action_size):
        self.render = False
        self.load_model = False
        # 상태와 행동의 크기 정의
        self.state_size = state_size
        self.action_size = action_size
        self.value_size = 1

        # 액터-크리틱 하이퍼파라미터
        self.discount_factor = 0.99
        self.actor_lr = 0.001
        self.critic_lr = 0.005

        # 정책신경망과 가치신경망 생성
        self.actor = Actor(self.action_size)
        self.actor_opt = Adam(learning_rate=self.actor_lr)

        self.critic = Critic()
        self.critic_opt = Adam(learning_rate=self.critic_lr)

        if self.load_model:
            self.actor.load_weights("./save_model/cartpole_actor_trained.h5")
            self.critic.load_weights("./save_model/cartpole_critic_trained.h5")

    # 정책신경망의 출력을 받아 확률적으로 행동을 선택
    def get_action(self, state):
        policy = self.actor(tf.convert_to_tensor(state[None, :], dtype=tf.float32))
        action = tf.squeeze(tf.random.categorical(policy, 1), axis=-1)
        return np.array(action)[0]

    def compute_advantages(self, states, rewards, dones):
        last_state = states[-1]
        if dones[-1] == True:
            reward_sum = 0
        else:
            reward_sum = self.critic(tf.convert_to_tensor(last_state[None, :], dtype=tf.float32))
        discounted_rewards = []
        for reward in rewards[::-1]:
            reward_sum = reward + self.discount_factor * reward_sum
            discounted_rewards.append(reward_sum)
        discounted_rewards.reverse()
        values = self.critic(tf.convert_to_tensor(np.vstack(states), dtype=tf.float32))
        advantages = discounted_rewards - values
        return advantages


    # 정책신경망을 업데이트하는 함수
    def actor_loss(self, states, actions, advantages):
        policy = self.actor(tf.convert_to_tensor(np.vstack(states), dtype=tf.float32))

        # entropy = categorical_crossentropy(policy, policy, from_logits=False)
        ce_loss = SparseCategoricalCrossentropy(from_logits=False)

        log_pi = ce_loss(actions, policy)
        policy_loss = log_pi * np.array(advantages)
        policy_loss = tf.reduce_mean(policy_loss)

        return policy_loss

    # 가치신경망을 업데이트하는 함수
    def critic_loss(self, states, rewards, dones):
        last_state = states[-1]
        if dones[-1] == True:
            reward_sum = 0
        else:
            reward_sum = self.critic(tf.convert_to_tensor(last_state[None, :], dtype=tf.float32))
        discounted_rewards = []
        for reward in rewards[::-1]:
            reward_sum = reward + self.discount_factor * reward_sum
            discounted_rewards.append(reward_sum)
        discounted_rewards.reverse()
        discounted_rewards = tf.convert_to_tensor(np.array(discounted_rewards)[None, :], dtype=tf.float32)
        values = self.critic(tf.convert_to_tensor(np.vstack(states), dtype=tf.float32))
        error = tf.square(values - discounted_rewards) * 0.5
        error = tf.reduce_mean(error)
        return error

    # 각 타임스텝마다 정책신경망과 가치신경망을 업데이트
    def train(self, states, actions, rewards, next_states, dones):
        critic_variable = self.critic.trainable_variables
        with tf.GradientTape() as tape_critic:
            tape_critic.watch(critic_variable)
            critic_loss = self.critic_loss(states, rewards, dones)

        # gradient descent will be applied automatically
        critic_grads = tape_critic.gradient(critic_loss, critic_variable)
        self.critic_opt.apply_gradients(zip(critic_grads, critic_variable))

        advantages = self.compute_advantages(states, rewards, dones)
        actor_variable = self.actor.trainable_variables
        with tf.GradientTape() as tape_actor:
            tape_actor.watch(actor_variable)
            actor_loss = self.actor_loss(states, actions, advantages)

        actor_grads = tape_actor.gradient(actor_loss, actor_variable)
        self.actor_opt.apply_gradients(zip(actor_grads, actor_variable))


if __name__ == "__main__":
    # CartPole-v1 환경, 최대 타임스텝 수가 500
    env = gym.make('CartPole-v1')

    t_end = 500
    train_size = 20

    states = []
    actions = []
    rewards = []
    next_states = []
    dones = []

    # 환경으로부터 상태와 행동의 크기를 받아옴
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # 액터-크리틱(A2C) 에이전트 생성
    agent = A2CAgent(state_size, action_size)

    scores, episodes = [], []

    for e in range(EPISODES):
        done = False
        score = 0
        state = env.reset()
        # state = np.reshape(state, [1, state_size])

        for t in range(t_end):
            if agent.render:
                env.render()

            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            # next_state = np.reshape(next_state, [1, state_size])

            if t == t_end:
                done = True

            # 에피소드가 중간에 끝나면 -100 보상
            reward = reward if not done or score == 499 else -100

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)

            score += reward
            state = next_state

            if len(states) == train_size or done:
                agent.train(states, actions, rewards, next_states, dones)
                states = []
                actions = []
                rewards = []
                next_states = []
                dones = []

            if done:
                # 에피소드마다 학습 결과 출력
                score = score if score == 500.0 else score + 100
                scores.append(score)
                episodes.append(e)
                pylab.plot(episodes, scores, 'b')
                pylab.savefig("./save_graph/cartpole_a2c.png")
                print("episode:", e, "  score:", score)

                # 이전 10개 에피소드의 점수 평균이 490보다 크면 학습 중단
                if np.mean(scores[-min(10, len(scores)):]) > 490:
                    agent.actor.save_weights("./save_model/cartpole_actor.h5")
                    agent.critic.save_weights(
                        "./save_model/cartpole_critic.h5")
                    sys.exit()

                break

    K.clear_session()
    del agent.model