from tensorflow.keras.layers import Conv2D, Dense, Flatten
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import Model
from tensorflow.keras.losses import Huber
from skimage.transform import resize
from skimage.color import rgb2gray
from collections import deque
from tensorflow.keras import backend as K
from atari_wrappers import wrap_dqn
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
        self.render = False
        self.load_model = False
        # 상태와 행동의 크기 정의
        self.state_size = (84, 84, 4)
        self.action_size = action_size
        # DQN 하이퍼파라미터
        self.epsilon = 1.
        self.epsilon_start, self.epsilon_end = 1.0, 0.1
        self.exploration_steps = 1000000.
        self.epsilon_decay_step = (self.epsilon_start - self.epsilon_end) \
                                  / self.exploration_steps
        self.batch_size = 32
        self.train_start = 1000
        self.update_target_rate = 10000
        self.discount_factor = 0.99
        # 리플레이 메모리, 최대 크기 400000
        self.memory = deque(maxlen=400000)
        self.no_op_steps = 30
        # 모델과 타겟모델을 생성하고 타겟모델 초기화
        self.model = DQN(self.action_size)
        self.target_model = DQN(self.action_size)
        self.update_target_model()

        # 텐서보드 설정
        self.log_dir = 'summary\\breakout_dqn'
        self.train_summary_writer = tf.summary.create_file_writer(self.log_dir)
        self.reward_board = tf.keras.metrics.Mean('reward_board', dtype=tf.float32)
        self.Q_board = tf.keras.metrics.Mean('Q_board', dtype=tf.float32)
        self.duration_board = tf.keras.metrics.Mean('duration_board', dtype=tf.float32)
        self.loss_board = tf.keras.metrics.Mean('loss_board', dtype=tf.float32)

        if self.load_model:
            self.model.load_weights("./save_model/breakout_dqn.h5")

    # 타겟 모델을 모델의 가중치로 업데이트
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # 입실론 탐욕 정책으로 행동 선택
    def get_action(self, history):
        # history = history / 255.0
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_value = self.model(tf.convert_to_tensor(history, dtype=tf.float32))
            return np.argmax(q_value[0])

    # 샘플 <s, a, r, s'>을 리플레이 메모리에 저장
    def append_sample(self, history, action, reward, next_history, dead):
        self.memory.append((history, action, reward, next_history, dead))

    # 리플레이 메모리에서 무작위로 추출한 배치로 모델 학습
    def train_model(self):
        if self.epsilon > self.epsilon_end:
            self.epsilon -= self.epsilon_decay_step

        mini_batch = random.sample(self.memory, self.batch_size)

        history = np.zeros((self.batch_size, self.state_size[0],
                            self.state_size[1], self.state_size[2]))
        next_history = np.zeros((self.batch_size, self.state_size[0],
                                 self.state_size[1], self.state_size[2]))
        action, reward, dead = [], [], []

        for i in range(self.batch_size):
            history[i] = mini_batch[i][0] / 255.
            next_history[i] = mini_batch[i][3] / 255.
            action.append(mini_batch[i][1])
            reward.append(mini_batch[i][2])
            dead.append(mini_batch[i][4])

        target = self.model(tf.convert_to_tensor(history, dtype=tf.float32))
        target_value = self.target_model(tf.convert_to_tensor(next_history, dtype=tf.float32))

        target = np.array(target)
        target_value = np.array(target_value)

        for i in range(self.batch_size):
            if dead[i]:
                target[i][action[i]] = reward[i]
            else:
                target[i][action[i]] = reward[i] + self.discount_factor * (np.amax(target_value[i]))

        self.model.compile(loss=Huber(), optimizer=RMSprop(learning_rate=0.00025, epsilon=0.01))
        logs = self.model.fit(history, target, epochs=1, verbose=0)

        return logs

    # 각 에피소드 당 학습 정보를 기록
    def setup_summary(self, total_reward, avg_q_max, duration, avg_loss, e):
        self.reward_board(total_reward)
        self.Q_board(avg_q_max)
        self.duration_board(duration)
        self.loss_board(avg_loss)

        with self.train_summary_writer.as_default():
            tf.summary.scalar('reward', total_reward, step=e)
            tf.summary.scalar('Q', avg_q_max, step=e)
            tf.summary.scalar('duration', duration, step=e)
            tf.summary.scalar('loss', avg_loss, step=e)

    # 학습속도를 높이기 위해 흑백화면으로 전처리
    def pre_processing(self, observe):
        processed_observe = np.int_(resize(rgb2gray(observe), (84, 84), mode='constant') * 255)
        return processed_observe


if __name__ == "__main__":
    # 환경과 DQN 에이전트 생성
    env = wrap_dqn(gym.make('BreakoutDeterministic-v4'))
    agent = DQNAgent(action_size=3)

    scores, episodes, global_step = [], [], 0

    for e in range(EPISODES):
        done = False
        dead = False

        step, score, start_life = 0, 0, 5
        avg_q_max, avg_loss = 0, 0
        state = env.reset()

        for _ in range(random.randint(1, agent.no_op_steps)):
            observe, _, _, _ = env.step(1)

        while not done:
            if agent.render:
                env.render()
            global_step += 1
            step += 1

            # 바로 전 4개의 상태로 행동을 선택
            action = agent.get_action(state)
            # 1: 정지, 2: 왼쪽, 3: 오른쪽
            if action == 0:
                real_action = 1
            elif action == 1:
                real_action = 2
            else:
                real_action = 3

            # 선택한 행동으로 환경에서 한 타임스텝 진행
            next_state, reward, done, info = env.step(real_action)
            avg_q_max += np.amax(agent.model(tf.convert_to_tensor(state, dtype=tf.float32))[0])

            # 타깃모델 초기화
            if global_step == 1:
                _ = agent.target_model(tf.convert_to_tensor(next_state, dtype=tf.float32))[0]

            if start_life > info['ale.lives']:
                dead = True
                start_life = info['ale.lives']

            reward = np.clip(reward, -1., 1.)
            # 샘플 <s, a, r, s'>을 리플레이 메모리에 저장 후 학습
            agent.append_sample(state, action, reward, next_state, dead)

            if len(agent.memory) >= agent.train_start:
                logs = agent.train_model()
                avg_loss = np.sum(logs.history['loss'])

            # 일정 시간마다 타겟모델을 모델의 가중치로 업데이트
            if global_step % agent.update_target_rate == 0:
                agent.update_target_model()

            score += reward

            if dead:
                dead = False
            else:
                state = next_state

            if done:
                # 각 에피소드 당 학습 정보를 기록
                if global_step > agent.train_start:
                    agent.setup_summary(score, avg_q_max/step, step, avg_loss/step, e)

                print("episode:", e, "  score:", score, "  memory length:",
                      len(agent.memory), "  epsilon:", agent.epsilon,
                      "  global_step:", global_step, "  average_q:",
                      avg_q_max / float(step), "  average loss:",
                      avg_loss / float(step))

        # 1000 에피소드마다 모델 저장
        if e % 1000 == 0:
            agent.model.save_weights("./save_model/breakout_dqn.h5")

    K.clear_session()
    del agent.model
    del agent.target_model