python pretrain.py Di_COT ecg2 -p configs/ecg2config.yml -s 1 --evaluate supervised
python pretrain.py Di_COT harth -p configs/harthconfig.yml -s 1 --evaluate supervised
python pretrain.py Di_COT pamap2 -p configs/pamap2config.yml -s 1 --evaluate supervised
python pretrain.py Di_COT skoda -p configs/skodaconfig.yml -s 1 --evaluate supervised
python pretrain.py Di_COT sleepm -p configs/sleepmconfig.yml -s 1 --evaluate supervised
python pretrain.py Di_COT wisdm2 -p configs/wisdm2config.yml -s 1 --evaluate supervised


python launch_all.py

python evaluate_all.py
python evaluate_ML.py

python evaluate_transfer.py
python evaluate_clustering.py