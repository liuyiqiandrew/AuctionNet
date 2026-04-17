# Strategy Training Environment

Auto-bidding strategy training module with data processing, strategy training, and offline evaluation.

## Data

Periods 7-26 are used for **training**, period 27 for **testing**.

Unzip the parquet data archive into `data/traffic/`:
```
data/traffic/
    period-7.parquet
    period-8.parquet
    ...
    period-27.parquet
```

Generate trajectory data (processes periods 7-26 only):
```bash
python bidding_train_env/train_data_generator/train_data_generator.py
```
This produces `data/traffic/training_data_rlData_folder/training_data_all-rlData.parquet`.

## Training

All commands run from `strategy_train_env/`.

| Algorithm | Command |
|-----------|---------|
| IQL | `python main/main_iql.py` |
| BC | `python main/main_bc.py` |
| BCQ | `python main/main_bcq.py` |
| CQL | `python main/main_cql.py` |
| TD3+BC | `python main/main_td3_bc.py` |
| OnlineLP | `python main/main_onlineLp.py` |
| Decision Transformer | `python main/main_decision_transformer.py` |

Trained models are saved to `saved_model/<algo>test/`.

## Evaluation

To select which trained strategy to evaluate, edit `bidding_train_env/strategy/__init__.py`:
```python
from .iql_bidding_strategy import IqlBiddingStrategy as PlayerBiddingStrategy
```

Run offline evaluation on period 27:
```bash
python main/main_test.py
```
