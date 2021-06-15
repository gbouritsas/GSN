## ---- imports -----

import argparse
import utils_parsing as parse
import os

import random
import copy
import json

import torch
import numpy as np

from torch_geometric.data import DataLoader
from torch_geometric.data import Data

from utils import process_arguments, prepare_dataset
from utils_data_prep import separate_data, separate_data_given_split
from utils_encoding import encode

from train_test_funcs import train, test_isomorphism, test, test_ogb, setup_optimization, resume_training

from models_graph_classification_mlp import MLPSubstructures
from models_graph_classification import GNNSubstructures
from models_graph_classification_ogb_original import GNN_OGB

from ogb.graphproppred import Evaluator


## ---- main function -----

def main(args):
    
    
    ## ----------------------------------- argument processing
    
    args, extract_ids_fn, count_fn, automorphism_fn, loss_fn, prediction_fn, perf_opt = process_arguments(args)
    evaluator = Evaluator(args['dataset_name']) if args['dataset'] == 'ogb' else None

    
    ## ----------------------------------- infrastructure

    torch.manual_seed(args['seed'])
    torch.cuda.manual_seed(args['seed'])
    torch.cuda.manual_seed_all(args['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args['np_seed'])
    os.environ['PYTHONHASHSEED'] = str(args['seed'])
    random.seed(args['seed'])
    print('[info] Setting all random seeds {}'.format(args['seed']))
    
    torch.set_num_threads(args['num_threads'])
    if args['GPU']:
        device = torch.device("cuda:"+str(args['device_idx']) if torch.cuda.is_available() else "cpu")
        print('[info] Training will be performed on {}'.format(torch.cuda.get_device_name(args['device_idx'])))
    else:
        device = torch.device("cpu")
        print('[info] Training will be performed on cpu')

    if args['wandb']:
        import wandb
        wandb.init(sync_tensorboard=False, project=args['wandb_project'], reinit = False, config = args, entity=args['wandb_entity'])
        print('[info] Monitoring with wandb')
        
        
    ## ----------------------------------- datasets: prepare and preprocess (count or load subgraph counts)
    
    path = os.path.join(args['root_folder'], args['dataset'], args['dataset_name'])
    subgraph_params = {'induced': args['induced'], 
                       'edge_list': args['custom_edge_list'],
                       'directed': args['directed'],
                       'directed_orbits': args['directed_orbits']}
    graphs_ptg, num_classes, orbit_partition_sizes = prepare_dataset(path, 
                                                                     args['dataset'],
                                                                     args['dataset_name'], 
                                                                     args['id_scope'], 
                                                                     args['id_type'], 
                                                                     args['k'], 
                                                                     args['regression'],
                                                                     extract_ids_fn, 
                                                                     count_fn,
                                                                     automorphism_fn,
                                                                     args['multiprocessing'],
                                                                     args['num_processes'],
                                                                     **subgraph_params)
    
    
    # OGB-specifics: different feature collections
    if args['dataset'] == 'ogb':

        if args['features_scope'] == 'simple':  # only retain the top two node/edge features
            print('[info] (OGB) Using simple node and edge features')
            simple_graphs = []
            for graph in graphs_ptg:
                new_data = Data()
                for attr in graph.__iter__():
                    name, value = attr
                    setattr(new_data, name, value)
                setattr(new_data, 'x', graph.x[:,:2])
                setattr(new_data, 'edge_features', graph.edge_features[:,:2])
                simple_graphs.append(new_data)
            graphs_ptg = simple_graphs
        else:
            print('[info] (OGB) Using full node and edge features')
        

    ## ----------------------------------- node and edge feature dimensions

    if graphs_ptg[0].x.dim()==1:
        num_features = 1
    else:
        num_features = graphs_ptg[0].num_features
        
    if hasattr(graphs_ptg[0], 'edge_features'):
        if graphs_ptg[0].edge_features.dim()==1:
            num_edge_features = 1
        else:
            num_edge_features  = graphs_ptg[0].edge_features.shape[1]
    else:
        num_edge_features = None

    if args['dataset'] == 'chemical' and args['dataset_name'] == 'ZINC':
        d_in_node_encoder, d_in_edge_encoder = torch.load(os.path.join(path, 'processed', 'num_feature_types.pt'))
        d_in_node_encoder, d_in_edge_encoder = [d_in_node_encoder], [d_in_edge_encoder]
    else:
        d_in_node_encoder = [num_features]
        d_in_edge_encoder = [num_edge_features]
        
    
    ## ----------------------------------- encode ids and degrees (and possibly edge features)

    degree_encoding = args['degree_encoding'] if args['degree_as_tag'][0] else None
    id_encoding = args['id_encoding'] if args['id_encoding'] != 'None' else None
    encoding_parameters = {
        'ids': {
            'bins': args['id_bins'],
            'strategy': args['id_strategy'],
            'range': args['id_range'],
        },
        'degree': {
            'bins': args['degree_bins'],
            'strategy': args['degree_strategy'],
            'range': args['degree_range']}}
    
    print("Encoding substructure counts and degree features... ", end='')
    graphs_ptg, encoder_ids, d_id, encoder_degrees, d_degree = encode(graphs_ptg, 
                                                                      id_encoding,
                                                                      degree_encoding,
                                                                      **encoding_parameters)
    print("Done.")
    
    assert args['mode'] in ['isomorphism_test', 'train','test'], "Unknown mode. Supported options are 'isomorphism_test', 'train','test'"
    
    ## ----------------------------------- graph isomorphism testing
    ##
    ## We use GSN with random weights, so no training is performed
    ##
    
    if args['mode'] == 'isomorphism_test':

        eps = args['isomorphism_eps']
        loader = DataLoader(graphs_ptg,
                            batch_size=args['batch_size'],
                            shuffle=False,
                            worker_init_fn=random.seed(args['seed']),
                            num_workers=args['num_workers'])
        
        model = GNNSubstructures(
            in_features=num_features, 
            out_features=num_classes, 
            encoder_ids=encoder_ids,
            d_in_id=d_id,
            in_edge_features=num_edge_features,
            d_in_node_encoder=d_in_node_encoder, 
            d_in_edge_encoder=d_in_edge_encoder,
            encoder_degrees=encoder_degrees,
            d_degree=d_degree,
            **args)
        
        model = model.to(device)
        print("Instantiated model:\n{}".format(model))
        
        # count model params
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("[info] Total number of parameters is: {}".format(params))

        mm, num_not_distinguished = test_isomorphism(loader, model, device, eps=eps)

        print('Total pairs: {}'.format(len(mm)))
        print('Number of non-isomorphic pairs that are not distinguised: {}'.format(num_not_distinguished))
        print('Failure Percentage: {:.2f}%'.format(100 * num_not_distinguished / len(mm)))

        if args['wandb']:
            wandb.run.summary['num_not_distinguished'] = num_not_distinguished
            wandb.run.summary['total pairs'] = len(mm)
            wandb.run.summary['failure_percentage'] = num_not_distinguished / len(mm)

        return

    
    ## ----------------------------------- training
    ##
    ## Unified training code for all the datasets.
    ## Please use args['onesplit'] = True if cross-validation is not required.
    ##

    print("Training starting now...")
    train_losses_folds = []; train_accs_folds = []
    test_losses_folds = []; test_accs_folds = []
    val_losses_folds = []; val_accs_folds = []

    results_folder_init = os.path.join(path, 'results', args['results_folder'])

    fold_idxs = [-1] if args['onesplit'] else args['fold_idx']
    for fold_idx in fold_idxs:
        
        print('############# FOLD NUMBER {:01d} #############'.format(fold_idx))
            
        # prepare result folder
        results_folder = os.path.join(results_folder_init, str(fold_idx), args['model_name'])
        if not os.path.exists(results_folder):
            os.makedirs(results_folder)

        # prepare folder for model checkpoints
        checkpoint_path = os.path.join(results_folder, 'checkpoints')
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)

        # save parameters of the training job
        with open(os.path.join(results_folder, 'params.json'), 'w') as fp:
            saveparams = copy.deepcopy(args)
            json.dump(saveparams, fp)

        # split data into training/validation/test
        if args['split'] == 'random':  # use a random split
            dataset_train, dataset_test = separate_data(graphs_ptg, args['split_seed'], fold_idx)
            dataset_val = None
        elif args['split'] == 'given':  # use a precomputed split
            dataset_train, dataset_test, dataset_val = separate_data_given_split(graphs_ptg, path, fold_idx)

        # instantiate data loaders
        loader_train = DataLoader(dataset_train,
                                  batch_size=args['batch_size'], 
                                  shuffle=args['shuffle'],
                                  worker_init_fn=random.seed(args['seed']), 
                                  num_workers=args['num_workers'])
        loader_test = DataLoader(dataset_test,
                                 batch_size=args['batch_size'], 
                                 shuffle=False, 
                                 worker_init_fn=random.seed(args['seed']), 
                                 num_workers=args['num_workers'])
        if dataset_val is not None:
            loader_val = DataLoader(dataset_val,
                                    batch_size=args['batch_size'], 
                                    shuffle=False,
                                    worker_init_fn=random.seed(args['seed']), 
                                    num_workers=args['num_workers'])
        else:
            loader_val = None
             
        # instantiate model
        if args['model_name'] == 'MLP':
            Model = MLPSubstructures
        else:
            if args['dataset'] == 'ogb':
                Model = GNN_OGB
            else:
                Model = GNNSubstructures
            
        model = Model(
            in_features=num_features, 
            out_features=num_classes, 
            encoder_ids=encoder_ids,
            d_in_id=d_id,
            in_edge_features=num_edge_features,
            d_in_node_encoder=d_in_node_encoder, 
            d_in_edge_encoder=d_in_edge_encoder,
            encoder_degrees=encoder_degrees,
            d_degree=d_degree,
            **args)
        model = model.to(device)
        print("Instantiated model:\n{}".format(model))
        
        # count model params
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("[info] Total number of parameters is: {}".format(params))

        if args['mode'] == 'train':
            
            # optimizer and lr scheduler
            optimizer, scheduler = setup_optimization(model, **args)
            
            # logging
            if args['wandb']:
                wandb.watch(model)
            
            checkpoint_filename = os.path.join(checkpoint_path, args['checkpoint_file'] + '.pth.tar')
            if args['resume']:
                start_epoch = resume_training(checkpoint_filename, model, optimizer, scheduler, device)    
            else:
                start_epoch = 0
                
            # train (!)
            metrics = train(
                loader_train,
                loader_test, 
                model,
                optimizer,
                loss_fn,
                loader_val=loader_val,
                prediction_fn=prediction_fn,
                evaluator=evaluator,
                scheduler=scheduler,
                min_lr=args['min_lr'],
                fold_idx=fold_idx,
                start_epoch=start_epoch, 
                n_epochs=args['num_epochs'],
                n_iters=args['num_iters'],
                n_iters_test=args['num_iters_test'],
                eval_freq=args['eval_frequency'], 
                checkpoint_file=checkpoint_filename,
                wandb_realtime=args['wandb_realtime'] and args['wandb'])

            # log results of training
            train_losses, train_accs, test_losses, test_accs, val_losses, val_accs = metrics
            train_losses_folds.append(train_losses)
            train_accs_folds.append(train_accs)
            test_losses_folds.append(test_losses)
            test_accs_folds.append(test_accs)
            val_losses_folds.append(val_losses)
            val_accs_folds.append(val_accs)
            best_idx = perf_opt(val_accs) if loader_val is not None else perf_opt(test_accs)
            print("Training complete!")
            print("\tbest train accuracy {:.4f}\n\tbest test accuracy {:.4f}".format(train_accs[best_idx], test_accs[best_idx]))
            
        elif args['mode'] == 'test': 

            checkpoint_filename = os.path.join(checkpoint_path, args['checkpoint_file'] + '.pth.tar')
            print('Loading checkpoint from file {}... '.format(checkpoint_filename), end='')
            checkpoint_dict = torch.load(checkpoint_filename, map_location=device)
            model.load_state_dict(checkpoint_dict['model_state_dict'])
            print('Done.')

            if args['dataset'] == 'ogb':
                _, train_acc = test_ogb(loader_train, model, loss_fn, device, evaluator)
                _, test_acc = test_ogb(loader_test, model, loss_fn, device, evaluator)
            else:
                _, train_acc = test(loader_train, model, loss_fn, device, prediction_fn)
                _, test_acc = test(loader_test, model, loss_fn, device, prediction_fn)

            train_accs_folds.append(train_acc)
            test_accs_folds.append(test_acc)
            
            
            if dataset_val is not None:
                if args['dataset'] == 'ogb':
                    _, val_acc = test(loader_val, model, loss_fn, device, prediction_fn)
                else:
                    _, val_acc = test(loader_val, model, loss_fn, device, prediction_fn)
                val_accs_folds.append(val_acc)

            print("Evaluation complete!")
            if dataset_val is not None:
                print("\ttrain accuracy {:.4f}\n\ttest accuracy {:.4f}\n\tvalidation accuracy {:.4f}".format(train_acc, test_acc, val_acc))
            else:
                print("\ttrain accuracy {:.4f}\n\ttest accuracy {:.4f}".format(train_acc, test_acc))
                

        else:

            raise NotImplementedError('Mode {} is not currently supported.'.format(args['mode']))

            
    # log metrics 
    if args['mode'] == 'train':

        train_accs_folds = np.array(train_accs_folds)
        test_accs_folds = np.array(test_accs_folds)
        train_losses_folds = np.array(train_losses_folds)
        test_losses_folds = np.array(test_losses_folds)

        train_accs_mean = np.mean(train_accs_folds, 0)
        train_accs_std = np.std(train_accs_folds, 0)
        test_accs_mean = np.mean(test_accs_folds, 0)
        test_accs_std = np.std(test_accs_folds, 0)
        
        train_losses_mean = np.mean(train_losses_folds, 0)
        test_losses_mean = np.mean(test_losses_folds, 0)
        
        if val_losses_folds[0] is not None:
            val_accs_folds = np.array(val_accs_folds)
            val_losses_folds = np.array(val_losses_folds)
            val_accs_mean = np.mean(val_accs_folds, 0)
            val_accs_std = np.std(val_accs_folds, 0)
            val_losses_mean = np.mean(val_losses_folds, 0)
        
        best_index = perf_opt(test_accs_mean) if val_losses_folds[0] is None else perf_opt(val_accs_mean)
        
        if not args['wandb_realtime'] and args['wandb']:

            for epoch in range(len(train_accs_mean)):

                # log scores for each fold in the current epoch
                for fold_idx in fold_idxs:
                    log_corpus = {
                               'train_accs_fold_'+str(fold_idx): train_accs_folds[fold_idx, epoch],
                               'train_losses_fold_'+str(fold_idx): train_losses_folds[fold_idx, epoch],
                               'test_accs_fold_'+str(fold_idx): test_accs_folds[fold_idx,epoch],
                               'test_losses_fold_'+str(fold_idx): test_losses_folds[fold_idx,epoch]}
                    if val_losses_folds[0] is not None:
                        log_corpus['val_accs_fold_'+str(fold_idx)] = val_accs_folds[fold_idx, epoch]
                        log_corpus['val_losses_fold_'+str(fold_idx)] = val_losses_folds[fold_idx, epoch]
                    wandb.log(log_corpus, step=epoch)
                
                # log epoch score means across folds
                log_corpus = {
                           'train_accs_mean': train_accs_mean[epoch],
                           'train_accs_std': train_accs_std[epoch],
                           'test_accs_mean': test_accs_mean[epoch],
                           'test_accs_std': test_accs_std[epoch],
                           'train_losses_mean': train_losses_mean[epoch],
                           'test_losses_mean': test_losses_mean[epoch]}
                if val_losses_folds[0] is not None:
                    log_corpus['val_accs_mean'] = val_accs_mean[epoch]
                    log_corpus['val_accs_std'] = val_accs_std[epoch]
                    log_corpus['val_losses_mean'] = val_losses_mean[epoch]
                wandb.log(log_corpus, step=epoch)

        if args['wandb']:
            
            wandb.run.summary['best_epoch_val'] = best_index

            wandb.run.summary['best_train_mean'] = train_accs_mean[best_index]
            wandb.run.summary['best_train_std'] = train_accs_std[best_index]
            wandb.run.summary['best_train_loss_mean'] = train_losses_mean[best_index]
            wandb.run.summary['last_train_std'] = train_accs_std[-1]
            wandb.run.summary['last_train_mean'] = train_accs_mean[-1]
            
            wandb.run.summary['best_test_mean'] = test_accs_mean[best_index]
            wandb.run.summary['best_test_std'] = test_accs_std[best_index]
            wandb.run.summary['best_test_loss_mean'] = test_losses_mean[best_index]
            
            wandb.run.summary['last_test_std'] = test_accs_std[-1]
            wandb.run.summary['last_test_mean'] = test_accs_mean[-1]

            if val_losses_folds[0] is not None:

                wandb.run.summary['best_validation_std'] = val_accs_std[best_index]
                wandb.run.summary['best_validation_mean'] = val_accs_mean[best_index]
                wandb.run.summary['best_validation_loss_mean'] = val_losses_mean[best_index]
                
                wandb.run.summary['last_validation_std'] = val_accs_std[-1]
                wandb.run.summary['last_validation_mean'] = val_accs_mean[-1]
                wandb.run.summary['performance_at_best_epoch'] = val_accs_mean[best_index]

            else:

                wandb.run.summary['performance_at_best_epoch'] = test_accs_mean[best_index]
                
        print("Best train mean: {:.4f} +/- {:.4f}".format(train_accs_mean[best_index], train_accs_std[best_index]))
        print("Best test mean: {:.4f} +/- {:.4f}".format(test_accs_mean[best_index], test_accs_std[best_index]))
        
        if args['return_scores']:
            scores = dict()
            scores['best_train_mean'] = train_accs_mean[best_index]
            scores['best_train_std'] = train_accs_std[best_index]
            scores['last_train_std'] = train_accs_std[-1]
            scores['last_train_mean'] = train_accs_mean[-1]
            scores['best_test_mean'] = test_accs_mean[best_index]
            scores['best_test_std'] = test_accs_std[best_index]
            scores['last_test_std'] = test_accs_std[-1]
            scores['last_test_mean'] = test_accs_mean[-1]
            if val_losses_folds[0] is not None:
                scores['best_validation_std'] = val_accs_std[best_index]
                scores['best_validation_mean'] = val_accs_mean[best_index]
                scores['last_validation_std'] = val_accs_std[-1]
                scores['last_validation_mean'] = val_accs_mean[-1]
      
    if args['mode'] == 'test' and not args['onesplit']:

        train_acc_mean = np.mean(train_accs_folds)
        test_acc_mean = np.mean(test_accs_folds)
        train_acc_std = np.std(train_accs_folds)
        test_acc_std = np.std(test_accs_folds)
        
        print("Train accuracy: {:.4f} +/- {:.4f}".format(train_acc_mean, train_acc_std))
        print("Test accuracy: {:.4f} +/- {:.4f}".format(test_acc_mean, test_acc_std))
        
        if dataset_val is not None:
            val_acc_mean = np.mean(val_accs_folds)
            val_acc_std = np.std(val_accs_folds)
            print("Validation accuracy: {:.4f} +/- {:.4f}".format(val_acc_mean, val_acc_std))
            
    if args['mode'] == 'train' and args['return_scores']:
        return scores
    else:
        return None


if __name__ == '__main__':   
   
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
    parser.add_argument('--root_folder', type=str, default='./datasets')
    
    ######  set degree_as_tag to True to use the degree as node features;
    # set retain_features to True to keep the existing features as well;
    parser.add_argument('--degree_as_tag', type=parse.str2bool, default=False)
    parser.add_argument('--retain_features', type=parse.str2bool, default=False)
    

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
    
    parser.add_argument('--random_features',  type=parse.str2bool, default=False)
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

    args = parser.parse_args()
    print(args)
    main(vars(args))
