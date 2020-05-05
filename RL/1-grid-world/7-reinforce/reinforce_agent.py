import copy
import pylab
import numpy as np
import tensorflow as tf
from environment import Env
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

EPISODES = 2500


class Reinforce(Model):
    def __init__(self, num_action):
        super(Reinforce, self).__init__()
        self.layer1 = Dense(24, activation='relu')
        self.layer2 = Dense(24, activation='relu')
        self.logits = Dense(num_action, activation='softmax')

    def call(self, state):
        x = self.layer1(state)
        x = self.layer2(x)
        logits = self.logits(x)
        return logits


# 그리드월드 예제에서의 REINFORCE 에이전트
class ReinforceAgent:
    def __init__(self):
        self.load_model = False
        # 가능한 모든 행동 정의
        self.action_space = [0, 1, 2, 3, 4]
        # 상태와 행동의 크기 정의
        self.action_size = len(self.action_space)
        self.state_size = 15
        self.discount_factor = 0.99
        self.learning_rate = 0.001

        self.model = Reinforce(self.action_size)
        self.optimizer = Adam(lr=self.learning_rate)
        self.states, self.actions, self.rewards = [], [], []

        if self.load_model:
            self.model.load_weights('./save_model/reinforce_trained.h5')


    # 정책신경망을 업데이트 하기 위한 오류함수와 훈련함수의 생성
    def actor_loss(self, states, actions, rewards):
        policy = self.model(tf.convert_to_tensor(np.vstack(states), dtype=tf.float32))

        # 크로스 엔트로피 오류함수 계산
        action_prob = tf.math.reduce_sum(actions * policy, axis=1)
        cross_entropy = tf.math.log(action_prob) * rewards
        loss = -tf.math.reduce_sum(cross_entropy)
        return loss

    # 정책신경망으로 행동 선택
    def get_action(self, state):
        policy = self.model(tf.convert_to_tensor(state[None, :], dtype=tf.float32))[0]
        return np.array(tf.squeeze(tf.random.categorical(policy, 1), axis=1))[0]

    # 반환값 계산
    def discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.discount_factor + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    # 한 에피소드 동안의 상태, 행동, 보상을 저장
    def append_sample(self, state, action, reward):
        self.states.append(state[0])
        self.rewards.append(reward)
        act = np.zeros(self.action_size)
        act[action] = 1
        self.actions.append(act)

    # 정책신경망 업데이트
    def train_model(self):
        discounted_rewards = np.float32(self.discount_rewards(self.rewards))
        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= np.std(discounted_rewards)

        actor_variable = self.model.trainable_variables
        with tf.GradientTape() as tape:
            tape.watch(actor_variable)
            actor_loss = self.actor_loss(self.states, self.actions, discounted_rewards)

        actor_grads = tape.gradient(actor_loss, actor_variable)
        self.optimizer.apply_gradients(zip(actor_grads, actor_variable))

        self.states, self.actions, self.rewards = [], [], []


if __name__ == "__main__":
    # 환경과 에이전트의 생성
    env = Env()
    agent = ReinforceAgent()

    global_step = 0
    scores, episodes = [], []

    for e in range(EPISODES):
        done = False
        score = 0
        # env 초기화
        state = env.reset()
        state = np.reshape(state, [1, 15])

        while not done:
            global_step += 1
            # 현재 상태에 대한 행동 선택
            action = agent.get_action(state)
            # 선택한 행동으로 환경에서 한 타임스탭 진행 후 샘플 수집
            next_state, reward, done = env.step(action)
            next_state = np.reshape(next_state, [1, 15])

            agent.append_sample(state, action, reward)
            score += reward
            state = copy.deepcopy(next_state)

            if done:
                # 에피소드마다 정책신경망 업데이트
                agent.train_model()
                scores.append(score)
                episodes.append(e)
                score = round(score, 2)
                print("episode:", e, "  score:", score, "  time_step:",
                      global_step)

        # 100 에피소드마다 학습 결과 출력 및 모델 저장
        if e % 100 == 0:
            pylab.plot(episodes, scores, 'b')
            pylab.savefig("./save_graph/reinforce.png")
            agent.model.save_weights("./save_model/reinforce.h5")

    K.clear_session()
    del agent.model