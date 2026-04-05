import sys

sys.path.append("./strategy_train_env")
import gin
from run.run_test import run_test
import torch
import numpy as np

torch.manual_seed(1)
np.random.seed(1)


@gin.configurable
def main():
    gin_file = ["./config/test.gin"]
    gin.parse_config_files_and_bindings(gin_file, None)
    all_results = []
    for player_index in range(7):
        result = run_test(player_index=player_index)
        formated_result = {
            'score': float(result["score"]),
            'reward': int(result["reward"]),
            'playerIndex': player_index,
            'category': int(result["category"]),
            'win_pv_ratio': result["win_pv_ratio"],
            'budget_consumer_ratio': result["budget_consumer_ratio"],
            'second_price_ratio': result["second_price_ratio"],
            'cpa_exceedance_Rate': min(result["cpa_exceedance_Rate"], 49999),
            'last_compete_tick_index': int(result["last_compete_tick_index"])
        }
        all_results.append(formated_result)

    collect_fields = ["score", "category", "reward", "win_pv_ratio", "budget_consumer_ratio", "second_price_ratio",
                      "last_compete_tick_index", "cpa_exceedance_Rate"]
    rank_score_field = "score"
    analysis_info = {}
    rank_score = 0.0
    for result in all_results:
        for key in collect_fields:
            if key == rank_score_field:
                rank_score += float(result[key])
            else:
                if key not in analysis_info.keys():
                    analysis_info[key] = []
                analysis_info[key].append(
                    float(format(result[key], '.3f')) if isinstance(result[key], float) else result[key])
    print("rank_score:", rank_score)
    print("analysis_info", analysis_info)


if __name__ == "__main__":
    main()
