import numpy as np
import torch
import logging
from bidding_train_env.common.utils import normalize_state, normalize_reward, save_normalize_dict
from bidding_train_env.baseline.iql.replay_buffer import ReplayBuffer
from bidding_train_env.baseline.bcq.bcq import BCQ
import sys
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s] [%(name)s] [%(filename)s(%(lineno)d)] [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

STATE_DIM = 16


def train_bcq_model():
    """
    Train the BCQ model.
    """
    train_data_path = "./data/traffic/training_data_rlData_folder/training_data_all-rlData.parquet"
    training_data = pd.read_parquet(train_data_path)
    STATE_DIM = len(training_data['state'].iloc[0])

    is_normalize = True
    if is_normalize:
        normalize_dic = normalize_state(training_data, STATE_DIM, normalize_indices=[13, 14, 15])
        training_data['reward'] = normalize_reward(training_data, "reward_continuous")
        save_normalize_dict(normalize_dic, "saved_model/BCQtest")

    # Build replay buffer
    replay_buffer = ReplayBuffer()
    add_to_replay_buffer(replay_buffer, training_data, is_normalize)
    print(len(replay_buffer.memory))

    # Train model
    model = BCQ(state_dim=STATE_DIM)
    train_model_steps(model, replay_buffer)

    # Save model
    # model.save_net("saved_model/BCQtest")
    model.save_jit("saved_model/BCQtest")

    # Test trained model
    test_trained_model(model, replay_buffer)


def add_to_replay_buffer(replay_buffer, training_data, is_normalize):
    for row in training_data.itertuples():
        state, action, reward, next_state, done = row.state if not is_normalize else row.normalize_state, row.action, row.reward if not is_normalize else row.normalize_reward, row.next_state if not is_normalize else row.normalize_nextstate, row.done
        # ! 去掉了所有的done==1的数据
        if done != 1:
            replay_buffer.push(np.array(state), np.array([action]), np.array([reward]), np.array(next_state),
                               np.array([done]))
        else:
            replay_buffer.push(np.array(state), np.array([action]), np.array([reward]), np.zeros_like(state),
                               np.array([done]))


def train_model_steps(model, replay_buffer, step_num=100, batch_size=100):
    for i in range(step_num):
        states, actions, rewards, next_states, terminals = replay_buffer.sample(batch_size)
        q_loss, a_loss = model.step(states, actions, rewards, next_states, terminals)
        logger.info(f'Step: {i} Q_loss: {q_loss} A_loss: {a_loss}')


def test_trained_model(model, replay_buffer):
    for i in range(100):
        states, actions, rewards, next_states, terminals = replay_buffer.sample(1)
        pred_actions = np.expand_dims(model.take_actions(states), 0)
        actions = actions.cpu().detach().numpy()
        tem = np.concatenate((actions, pred_actions), axis=1)
        print("concate:", tem)


def run_bcq():
    print(sys.path)
    """
    Run BCQ model training and evaluation.
    """
    train_bcq_model()


if __name__ == '__main__':
    run_bcq()
