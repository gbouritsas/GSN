#!/bin/bash
for i in {1..10}
do
	python -m main_HIV --weight_decay=3e-6 --L=4 --type_net="simple" --hidden_dim=60 --out_dim=60 --residual=True --edge_feat=False --readout=mean --in_feat_dropout=0.0 --dropout=0.3 --graph_norm=False --batch_norm=True --aggregators="mean max min dir0-av dir1-av dir2-av dir3-av" --scalers="identity" --dataset ogbg-molhiv --gpu_id 0 --config "configs/molecules_graph_classification_DGN_HIV.json" --epochs=200 --init_lr=0.01 --lr_reduce_factor=0.5 --lr_schedule_patience=20 --min_lr=0.0001 --id_scope local --k 6 --id_type cycle_graph --directions 'subgraphs'
done

