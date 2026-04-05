for p in {0..6}
do
    for i in {0..1}
    do
        python plot_eval_log.py \
            --input data/log/player_${p}_episode_${i}.csv
    done
done