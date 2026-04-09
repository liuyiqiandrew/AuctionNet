# Overview
This is an auto-bidding strategy training module to help users implement and evaluate their bidding strategies. This framework includes three modules: data processing, strategy training, and offline evaluation. Several industry-proven baseline strategies, such as reinforcement learning-based bidding, online linear programming-based bidding and generative models, are included in the framework. Users can utilize this framework to develop a well-trained auto-bidding strategy based on the training dataset. Users can also rely on the provided framework for a basic offline assessment to evaluate the strategy.




## Data Processing
Download the raw data on ad opportunities granularity and place it in the biddingTrainENv/data/ folder.
The directory structure under data should be:
```
NeurIPS_Auto_Bidding_General_Track_Baseline
|── data
    |── traffic
        |── period-7.csv
        |── period-8.csv
        |── period-9.csv
        |── period-10.csv
        |── period-11.csv
        |── period-12.csv
        |── period-13.csv
        |── period-14.csv
        |── period-15.csv
        |── period-16.csv
        |── period-17.csv
        |── period-18.csv
        |── period-19.csv
        |── period-20.csv
        |── period-21.csv
        |── period-22.csv
        |── period-23.csv
        |── period-24.csv
        |── period-25.csv
        |── period-26.csv
        |── period-27.csv
        
```

Run this script to convert the raw data on ad opportunities granularity into trajectory data required for model training.
```
python  bidding_train_env/train_data_generator/train_data_generator.py
```

## strategy training
### reinforcement learning-based bidding

#### PPO (Proximal Policy Optimization) Model
Use `main_ppo.py` for both training and evaluation of the PPO bidding strategy. 

⚠️ NOTE: PPO is evaluated online, do not use offline evaluation for PPO. 
```
  python main/main_ppo.py                    # full train + eval                              
  python main/main_ppo.py --mode train --num-iterations 5  # quick test                          
  python main/main_ppo.py --mode eval        # eval saved model                                  
  python main/main_ppo.py --mode plot        # regenerate plots 
```

#### IQL(Implicit Q-learning) Model
Load the training data and train the IQL bidding strategy.
```
python main/main_iql.py 
```
Use the IqlBiddingStrategy as the PlayerBiddingStrategy for evaluation.
```
bidding_train_env/strategy/__init__.py
from .iql_bidding_strategy import IqlBiddingStrategy as PlayerBiddingStrategy
```
#### BC(behavior cloning) Model
Load the training data and train the BC bidding strategy.
```
python main/main_bc.py 
```
Use the BcBiddingStrategy as the PlayerBiddingStrategy for evaluation.
```
bidding_train_env/strategy/__init__.py
from .bc_bidding_strategy import BcBiddingStrategy as PlayerBiddingStrategy
```

#### BCQ  Model
Load the training data and train the BCQ bidding strategy.
```
python main/main_bcq.py 
```
Use the BcqBiddingStrategy as the PlayerBiddingStrategy for evaluation.
```
bidding_train_env/strategy/__init__.py
# from .bcq_bidding_strategy import BcqBiddingStrategy as PlayerBiddingStrategy
```
#### IQL  Model
Load the training data and train the CQL bidding strategy.
```
python main/main_cql.py 
```
Use the CqlBiddingStrategy as the PlayerBiddingStrategy for evaluation.
```
bidding_train_env/strategy/__init__.py
# from .cql_bidding_strategy import CqlBiddingStrategy as PlayerBiddingStrategy
```

#### TD3_BC  Model
Load the training data and train the TD3_BC bidding strategy.
```
python main/main_td3_bc.py 
```
Use the TD3_BCBiddingStrategy as the PlayerBiddingStrategy for evaluation.
```
bidding_train_env/strategy/__init__.py
from .td3_bc_bidding_strategy import TD3_BCBiddingStrategy as PlayerBiddingStrategy
```


### online linear programming-based bidding
#### OnlineLp Model
Load the training data and train the OnlineLp bidding strategy.
```
python main/main_onlineLp.py 
```
Use the OnlineLpBiddingStrategy as the PlayerBiddingStrategy for evaluation.
```
bidding_train_env/strategy/__init__.py
# from .onlinelp_bidding_strategy import OnlineLpBiddingStrategy as PlayerBiddingStrategy
```
### Generative Model
#### Decision-Transformer
Load the training data and train the DT bidding strategy.
```
python main/main_decision_transformer.py 
```
Use the DtBiddingStrategy as the PlayerBiddingStrategy for evaluation.
```
bidding_train_env/strategy/__init__.py
from .dt_bidding_strategy import DtBiddingStrategy as PlayerBiddingStrategy
```




## offline evaluation
Load the raw data on ad opportunities granularity to construct an offline evaluation environment for assessing the bidding strategy offline.
```
python main/main_test.py
```

