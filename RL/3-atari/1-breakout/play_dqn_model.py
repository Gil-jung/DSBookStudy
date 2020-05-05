from tensorflow.keras.layers import Conv2D, Dense, Flatten
from tensorflow.keras.models import Model
from skimage.transform import resize
from skimage.color import rgb2gray
from tensorflow.keras import backend as K
import tensorflow as tf
import numpy as np
import random
import gym

EPISODES = 50000


# 상태가 입력, 큐함수가 출력인 인공신경망 생성
class DQN(Model):
    def __init__(self, num_action):
        super(DQN, self).__init__()
        self.conv1 = Conv2D(32, (8, 8), strides=(4, 4), activation='relu')
        self.conv2 = Conv2D(64, (4, 4), strides=(2, 2), activation='relu')
        self.conv3 = Conv2D(64, (3, 3), strides=(1, 1), activation='relu')
        self.dense1 = Dense(512, activation='relu')
        self.dense2 = Dense(num_action)

    def call(self, state):
        x = self.conv1(state)
        x = self.conv2(x)
        x = self.conv3(x)
        x = Flatten()(x)
        x = self.dense1(x)
        values = self.dense2(x)
        return values


# 브레이크아웃에서의 DQN 에이전트
class DQNAgent:
    def __init__(self, action_size):
        self.render = True
        self.load_model = True
        # 상태와 행동의 크기 정의
        self.state_size = (84, 84, 4)
        self.action_size = action_size
        self.no_op_steps = 20
        self.epsilon = 0.01

        self.model = DQN(self.action_size)

        if self.load_model:
            self.model.load_weights("./save_model/breakout_dqn.h5")

    # 입실론 탐욕 정책으로 행동 선택
    def get_action(self, history):
        history = history / 255.0
        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_value = self.model(tf.convert_to_tensor(history, dtype=tf.float32))
            return np.argmax(q_value[0])

    # 학습속도를 높이기 위해 흑백화면으로 전처리
    def pre_processing(self, observe):
        processed_observe = np.int_(resize(rgb2gray(observe), (84, 84), mode='constant') * 255)
        return processed_observe


if __name__ == "__main__":
    # 환경과 DQN 에이전트 생성
    env = gym.make('BreakoutDeterministic-v4')
    agent = DQNAgent(action_size=3)

    scores, episodes, global_step = [], [], 0

    for e in range(EPISODES):
        done = False
        dead = False

        step, score, start_life = 0, 0, 5
        observe = env.reset()

        for _ in range(random.randint(1, agent.no_op_steps)):
            observe, _, _, _ = env.step(1)

        state = agent.pre_processing(observe)
        history = np.stack((state, state, state, state), axis=2)
        history = np.reshape([history], (1, 84, 84, 4))

        while not done:
            if agent.render:
                env.render()
            step += 1

            # 바로 전 4개의 상태로 행동을 선택
            action = agent.get_action(history)
            # 1: 정지, 2: 왼쪽, 3: 오른쪽
            if action == 0:
                real_action = 1
            elif action == 1:
                real_action = 2
            else:
                real_action = 3

            if dead:
                real_action = 1
                dead = False

            # 선택한 행동으로 환경에서 한 타임스텝 진행
            observe, reward, done, info = env.step(real_action)
            # 각 타임스텝마다 상태 전처리
            next_state = agent.pre_processing(observe)
            next_state = np.reshape([next_state], (1, 84, 84, 1))
            next_history = np.append(next_state, history[:, :, :, :3], axis=3)

            if start_life > info['ale.lives']:
                dead = True
                start_life = info['ale.lives']

            score += reward

            history = next_history

            if done:
                print("episode:", e, "  score:", score)

    K.clear_session()
    del agent.model