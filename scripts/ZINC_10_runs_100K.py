import argparse
import sys
sys.path.append('../')
import main
import utils_parsing as parse
import torch

scores_list = list()
num_seeds = 10
for i in range(num_seeds):

    parser = argparse.ArgumentParser()

    # set seeds to ensure reproducibility
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--split_seed', type=int, default=0)
    parser.add_argument('--np_seed', type=int, default=0)

    # this specifies the folds for cross-validation
    parser.add_argument('--fold_idx', type=parse.str2list2int, default=[0,1,2,3,4,5,6,7,8,9])
    parser.add_argument('--onesplit', type=parse.str2bool, default=False)

    # set multiprocessing to true in order to do the precomputation in parallel
    parser.add_argument('--multiprocessing', type=parse.str2bool, default=False)
    parser.add_argument('--num_processes', type=int, default=64)

    ###### data loader parameters
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--num_threads', type=int, default=1)

    ###### these are to select the dataset:
    # - dataset can be bionformatics or social and states the class;
    # - name is for the specific problem itself
    parser.add_argument('--dataset', type=str, default='bioinformatics')
    parser.add_argument('--dataset_name', type=str, default='MUTAG')
    parser.add_argument('--split', type=str, default='given')
    parser.add_argument('--root_folder', type=str, default='/vol/deform/gbouritsas/datasets')

    ######  set degree_as_tag to True to use the degree as node features;
    # set retain_features to True to keep the existing features as well;
    parser.add_argument('--degree_as_tag', type=parse.str2bool, default=False)
    parser.add_argument('--retain_features', type=parse.str2bool, default=True)


    ###### used only for ogb to reproduce the different configurations,
    # i.e. additional features (full) or not (simple), virtual node or not (vn: True)
    parser.add_argument('--features_scope', type=str, default="full")
    parser.add_argument('--vn', type=parse.str2bool, default=False)
    # denotes the aggregation used by the virtual node
    parser.add_argument('--vn_pooling', type=str, default='sum')
    parser.add_argument('--input_vn_encoder', type=str, default='one_hot_encoder')
    parser.add_argument('--d_out_vn_encoder', type=int, default=None)
    parser.add_argument('--d_out_vn', type=int, default=None)

    ###### substructure-related parameters:
    # - id_type: substructure family
    # - induced: graphlets vs motifs
    # - edge_automorphism: induced edge automorphism or line graph edge automorphism (slightly larger group than the induced edge automorphism)
    # - k: size of substructures that are used; e.g. k=3 means three nodes
    # - id_scope: local vs global --> GSN-e vs GSN-v
    parser.add_argument('--id_type', type=str, default='cycle_graph')
    parser.add_argument('--induced', type=parse.str2bool, default=False)
    parser.add_argument('--edge_automorphism', type=str, default='induced')
    parser.add_argument('--k', type=parse.str2list2int, default=[3])
    parser.add_argument('--id_scope', type=str, default='local')
    parser.add_argument('--custom_edge_list', type=parse.str2ListOfListsOfLists2int, default=None)
    parser.add_argument('--directed', type=parse.str2bool, default=False)
    parser.add_argument('--directed_orbits', type=parse.str2bool, default=False)

    ###### encoding args: different ways to encode discrete data

    parser.add_argument('--id_encoding', type=str, default='one_hot_unique')
    parser.add_argument('--degree_encoding', type=str, default='one_hot_unique')


    # binning and minmax encoding parameters. NB: not used in our experimental evaluation
    parser.add_argument('--id_bins', type=parse.str2list2int, default=None)
    parser.add_argument('--degree_bins', type=parse.str2list2int, default=None)
    parser.add_argument('--id_strategy', type=str, default='uniform')
    parser.add_argument('--degree_strategy', type=str, default='uniform')
    parser.add_argument('--id_range', type=parse.str2list2int, default=None)
    parser.add_argument('--degree_range', type=parse.str2list2int, default=None)


    parser.add_argument('--id_embedding', type=str, default='one_hot_encoder')
    parser.add_argument('--d_out_id_embedding', type=int, default=None)
    parser.add_argument('--degree_embedding', type=str, default='one_hot_encoder')
    parser.add_argument('--d_out_degree_embedding', type=int, default=None)

    parser.add_argument('--input_node_encoder', type=str, default='None')
    parser.add_argument('--d_out_node_encoder', type=int, default=None)
    parser.add_argument('--edge_encoder', type=str, default='None')
    parser.add_argument('--d_out_edge_encoder', type=int, default=None)


    # sum or concatenate embeddings when multiple discrete features available
    parser.add_argument('--multi_embedding_aggr', type=str, default='sum')

    # only used for the GIN variant: creates a dummy variable for self loops (e.g. edge features or edge counts)
    parser.add_argument('--extend_dims', type=parse.str2bool, default=True)

    ###### model to be used and architecture parameters, in particular
    # - d_h: is the dimension for internal mlps, set to None to
    #   make it equal to d_out
    # - final_projection: is for jumping knowledge, specifying
    #   which layer is accounted for in the last model stage, if
    #   the list has only one element, that that value gets applied
    #   to all the layers
    # - jk_mlp: set it to True to use an MLP after each jk layer, otherwise a linear layer will be used

    parser.add_argument('--model_name', type=str, default='GSN_sparse')

    parser.add_argument('--random_features', type=parse.str2bool, default=False)
    parser.add_argument('--num_mlp_layers', type=int, default=2)
    parser.add_argument('--d_h', type=int, default=None)
    parser.add_argument('--activation_mlp', type=str, default='relu')
    parser.add_argument('--bn_mlp', type=parse.str2bool, default=True)

    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--d_msg', type=int, default=None)
    parser.add_argument('--d_out', type=int, default=16)
    parser.add_argument('--bn', type=parse.str2bool, default=True)
    parser.add_argument('--dropout_features', type=float, default=0)
    parser.add_argument('--activation', type=str, default='relu')
    parser.add_argument('--train_eps', type=parse.str2bool, default=False)
    parser.add_argument('--aggr', type=str, default='add')
    parser.add_argument('--flow', type=str, default='source_to_target')

    parser.add_argument('--final_projection', type=parse.str2list2bool, default=[True])
    parser.add_argument('--jk_mlp', type=parse.str2bool, default=False)
    parser.add_argument('--residual', type=parse.str2bool, default=False)

    parser.add_argument('--readout', type=str, default='sum')

    ###### architecture variations:
    # - msg_kind: gin (extends gin with structural identifiers),
    #             general (general formulation with MLPs - eq 3,4 of the main paper)
    #             ogb (extends the architecture used in ogb with structural identifiers)
    # - inject*: passes the relevant variable to deeper layers akin to skip connections.
    #            If set to False, then the variable is used only as input to the first layer
    parser.add_argument('--msg_kind', type=str, default='general')
    parser.add_argument('--inject_ids', type=parse.str2bool, default=False)
    parser.add_argument('--inject_degrees', type=parse.str2bool, default=False)
    parser.add_argument('--inject_edge_features', type=parse.str2bool, default=True)

    ###### optimisation parameters
    parser.add_argument('--shuffle', type=parse.str2bool, default=True)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=300)
    parser.add_argument('--num_iters', type=int, default=None)
    parser.add_argument('--num_iters_test', type=int, default=None)
    parser.add_argument('--eval_frequency', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--regularization', type=float, default=0)
    parser.add_argument('--scheduler', type=str, default='StepLR')
    parser.add_argument('--scheduler_mode', type=str, default='min')    
    parser.add_argument('--min_lr', type=float, default=0.0)
    parser.add_argument('--decay_steps', type=int, default=50)
    parser.add_argument('--decay_rate', type=float, default=0.5)
    parser.add_argument('--patience', type=int, default=20)


    ###### training parameters: task, loss, metric
    parser.add_argument('--regression', type=parse.str2bool, default=False)
    parser.add_argument('--loss_fn', type=str, default='CrossEntropyLoss')
    parser.add_argument('--prediction_fn', type=str, default='multi_class_accuracy')

    ######  folders to save results
    parser.add_argument('--results_folder', type=str, default='temp')
    parser.add_argument('--checkpoint_file', type=str, default='checkpoint')

    ######  general (mode, gpu, logging)
    parser.add_argument('--mode', type=str, default='train')

    parser.add_argument('--resume', type=parse.str2bool, default=False)
    parser.add_argument('--GPU', type=parse.str2bool, default=True)
    parser.add_argument('--device_idx', type=int, default=0)
    parser.add_argument('--wandb', type=parse.str2bool, default=True)
    parser.add_argument('--wandb_realtime', type=parse.str2bool, default=False)
    parser.add_argument('--wandb_project', type=str, default="gsn_project")
    parser.add_argument('--wandb_entity', type=str, default="anonymous")

    ######  misc
    parser.add_argument('--isomorphism_eps', type=float, default=1e-2)
    parser.add_argument('--return_scores', action='store_true')


    
    args = ['--seed', '{}'.format(i), '--onesplit', 'True', '--dataset', 'chemical', '--dataset_name', 'ZINC', '--id_type', 'cycle_graph', '--induced', 'False', '--k', '8', '--id_scope', 'global', '--id_encoding', 'one_hot_unique', '--id_embedding', 'one_hot_encoder', '--input_node_encoder', 'one_hot_encoder', '--edge_encoder', 'one_hot_encoder', '--model_name', 'GSN_edge_sparse', '--msg_kind', 'general', '--num_layers', '4 ', '--d_out', '64', '--dropout_features', '0', '--final_projection', 'False', '--jk_mlp', 'True', '--readout', 'sum', '--batch_size', '128', '--num_epochs', '1000', '--lr', '1e-3 ', '--scheduler', 'ReduceLROnPlateau', '--decay_rate', '0.5', '--patience', '5', '--min_lr', '1e-5 ', '--regression', 'True', '--loss_fn', 'L1Loss', '--prediction_fn', 'L1Loss', '--mode', 'train', '--device_idx', '0', '--wandb', 'False', '--return_scores', '--inject_ids', 'False', '--random_features', 'False', '--degree_embedding', 'None']


    
    
    
    args = parser.parse_args(args)

    scores = main.main(vars(args))
    scores_list.append(scores)

last_tests = [scores['last_test_mean'] for scores in scores_list]
print('mean test MAE: {} ± {} (number of seeds: {})'.format(torch.mean(torch.FloatTensor(last_tests)), torch.std(torch.FloatTensor(last_tests)),num_seeds))
