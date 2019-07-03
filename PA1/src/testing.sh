python train.py --lr 0.001 --momentum 0.9 --num_hidden 3 --sizes 250,200,150 --activation relu --loss ce --opt adam --batch_size 20 --epochs 10 --anneal true --save_dir ../save_dir/best/ --expt_dir ../expt_dir/ --train train.csv --val valid.csv --test test.csv --input_dim 75 --pretrain true --state 1 --testing true --lambd 0.001
python train.py --lr 0.001 --momentum 0.9 --num_hidden 3 --sizes 350,350,350 --activation relu --loss ce --opt adam --batch_size 20 --epochs 6 --anneal true --save_dir ../save_dir/best/ --expt_dir ../expt_dir/ --train train.csv --val valid.csv --test test.csv --lambd 0 --input_dim 100 --pretrain true --state 2 --testing true