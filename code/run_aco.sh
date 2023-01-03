python main_label_moco.py -a resnet34 --mlp -j 16 --lr 0.003 \
	--batch-size 256 --moco-k 40960 --dist-url 'tcp://localhost:10001' \
	--gpu 2 --world-size 1 --rank 0 
