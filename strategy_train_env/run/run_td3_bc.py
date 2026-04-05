import numpy as np
import torch
import logging
from bidding_train_env.common.utils import normalize_state, normalize_reward, save_normalize_dict
from bidding_train_env.baseline.iql.replay_buffer import ReplayBuffer
from bidding_train_env.baseline.td3_bc.td3_bc import TD3_BC
import sys
import pandas as pd
import ast
import pickle
# Configure logging
logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s] [%(name)s] [%(filename)s(%(lineno)d)] [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

STATE_DIM = 16


def train_td3_bc_model():
    """
    Train the td3_bc model.
    """
    train_data_path = "./data/traffic/training_data_rlData_folder/training_data_all-rlData.csv"
    training_data = pd.read_csv(train_data_path)

    def safe_literal_eval(val):
        if pd.isna(val):
            return val  # 如果是NaN，返回NaN
        try:
            return ast.literal_eval(val)
        except (ValueError, SyntaxError):
            try:
                return eval(val, {"__builtins__": {}}, {"np": np})
            except Exception:
                return val  # 如果解析出错，返回原值

    # 使用apply方法应用上述函数
    training_data["state"] = training_data["state"].apply(safe_literal_eval)
    training_data["next_state"] = training_data["next_state"].apply(safe_literal_eval)
    STATE_DIM = len(training_data['state'].iloc[0])

    is_normalize = True
    if is_normalize:
        normalize_dic = normalize_state(training_data, STATE_DIM, normalize_indices=[13, 14, 15])
        training_data['reward'] = normalize_reward(training_data, "reward_continuous")
        save_normalize_dict(normalize_dic, "saved_model/TD3_bctest")

    # Build replay buffer
    replay_buffer = ReplayBuffer()
    add_to_replay_buffer(replay_buffer, training_data, is_normalize)
    print(len(replay_buffer.memory))

    # Train model
    model = TD3_BC(dim_obs=STATE_DIM)
    train_model_steps(model, replay_buffer)

    # Save model
    # model.save_net("saved_model/TD3_bctest")
    model.save_jit("saved_model/TD3_bctest")

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
        pred_actions = model.take_actions(torch.tensor(states,dtype=torch.float))
        actions = actions.cpu().detach().numpy()
        tem = np.concatenate((actions, pred_actions), axis=1)
        print("concate:",tem)

def run_td3_bc():
    print(sys.path)
    """
    Run td3_BC model training and evaluation.
    """
    train_td3_bc_model()

if __name__ == '__main__':
    run_td3_bc()
