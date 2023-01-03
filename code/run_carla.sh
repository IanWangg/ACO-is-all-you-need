python -u main_label_moco_carla_new.py -a resnet34 --mlp -j 16 --lr 0.003 \
	--batch-size 256 --moco-k 40960 --dist-url 'tcp://localhost:10001' \
	--gpu 5 --world-size 1 --rank 0 \
	--action-seq-length 1 --exp-name 'moco_interval10'

# python main_label_moco_carla_new.py -a resnet34 --mlp -j 16 --lr 0.003 \
	# --batch-size 256 --moco-k 40960 --dist-url 'tcp://localhost:10001' \
	# --multiprocessing-distributed --world-size 1 --rank 0 