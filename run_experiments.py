import json
import os
import sys
from class_resolver.contrib import torch
import pykeen
from pykeen.datasets.inductive.ilp_teru import InductiveNELL, InductiveFB15k237, InductiveWN18RR
from pykeen.evaluation.rank_based_evaluator import RankBasedEvaluator
from pykeen.stoppers import EarlyStopper
from pykeen.trackers import TensorBoardResultTracker
from pykeen.training import SLCWATrainingLoop
from pykeen.training.callbacks import TrackerTrainingCallback
from pykeen.triples import TriplesFactory
from torch.optim import Adam, Adagrad
from RESHUFFLE_GNN import RESHUFFLE_Node_GNN
from RESHUFFLE_Layer import RESHUFFLE_Layer
from RESHUFFLE_Interaction import RESHUFFLE_Interaction
from Utils import analyze_checkpoints


def parse_kwargs(**kwargs):
    if 'config_dir' in kwargs.keys() and 'config_name' in kwargs.keys():
        config_path = kwargs['config_dir'] + kwargs['config_name'] + ".json"

        with open(config_path, "r") as f:
            config = json.loads(f.read())
    else:
        raise Exception("No config input file specified.")

    log_dir = '/logs'
    if 'log_dir' in kwargs.keys():
        log_dir = '/' + kwargs['log_dir']

    gpu = None
    if 'gpu' in kwargs.keys():
        gpu = kwargs['gpu']
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu

    if 'seeds' in kwargs.keys():
        seeds = kwargs['seeds'].split(',')
        seeds = [int(i) for i in seeds]
    else:
        seeds = [1, 2, 3]

    if 'config_name' in kwargs.keys():
        experiment_name = kwargs['config_name']

    if 'test' in kwargs.keys():
        if kwargs['test'] == 'true':
            test = True
        elif kwargs['test'] == 'false':
            test = False
        else:
            raise Exception(
                "Invalid value %s for parameter test. Parameter test may only be <true> or <false>." % kwargs[
                    'test'])
    else:
        raise Exception("Parameter test needs to be specified.")

    if 'train' in kwargs.keys():
        if kwargs['train'] == 'true':
            train = True
        elif kwargs['train'] == 'false':
            train = False
        else:
            raise Exception(
                "Invalid value %s for parameter train. Parameter train may only be <true> or <false>." % kwargs[
                    'train'])
    else:
        raise Exception(
            "Parameter <train> needs to be specified.")

    return log_dir, config_path, config, experiment_name, seeds, train, test, gpu


def parse_config(config):
    if config["optimizer"] == "Adam":
        optimizer = Adam
    elif config["optimizer"] == "Adagrad":
        optimizer = Adagrad
    else:
        raise Exception("Optimizer %s unknown." % config["optimizer"])

    if config['dataset'] == "InductiveNELL":
        dataset = InductiveNELL(version=config['dataset_version'],
                                create_inverse_triples=config['dataset_kwargs']['create_inverse_triples'], force=True)
    elif config['dataset'] == "InductiveFB15k237":
        dataset = InductiveFB15k237(version=config['dataset_version'],
                                    create_inverse_triples=config['dataset_kwargs']['create_inverse_triples'],
                                    force=True)
    elif config['dataset'] == "InductiveWN18RR":
        dataset = InductiveWN18RR(version=config['dataset_version'],
                                  create_inverse_triples=config['dataset_kwargs']['create_inverse_triples'], force=True)
    else:
        raise Exception('Dataset unknown.')

    dataset._inductive_validation = TriplesFactory.from_path(
        path=dataset.inductive_validation_path,
        entity_to_id=dataset._transductive_training.entity_to_id,
        relation_to_id=dataset._transductive_training.relation_to_id,
        create_inverse_triples=False,
        load_triples_kwargs=dataset.load_triples_kwargs,
    )

    return optimizer, dataset


def evaluate_and_save_final_result(seed, model, dataset, config, experiment_dir, experiment_name, current_config):
    analysis_path = experiment_dir + '/complete_results/' + config["dataset"] + '/' + experiment_name
    res = analyze_checkpoints(seed, model, dataset, config, analysis_path)

    final_res = {'MR': res['both.realistic.arithmetic_mean_rank'],
                 'MRR': res['both.realistic.inverse_harmonic_mean_rank'],
                 'Hits@1': res['both.realistic.hits_at_1'],
                 'Hits@3': res['both.realistic.hits_at_3'],
                 'Hits@10': res['both.realistic.hits_at_10']}

    for parameter in current_config.keys():
        final_res[parameter] = current_config[parameter]

    final_res_dir = experiment_dir + '/short_results' + '/' + config["dataset"] + "/" + experiment_name
    file_name = 'test_result_seed_%s.json' % seed
    final_res_path = os.path.join(final_res_dir, file_name)

    if not os.path.exists(final_res_dir):
        os.makedirs(final_res_dir)
    with open(final_res_path, 'w') as fp:
        json.dump(final_res, fp)


def main(**kwargs):
    log_dir, config_path, config, experiment_name_config, seeds, train, test, gpu = parse_kwargs(**kwargs)
    experiment_dir = './Benchmarking'

    exp_id = 0
    current_config = {}
    for num_layers in config['model_kwargs']['num_layers']:
        current_config['num_layers'] = num_layers
        for margin in config['loss_kwargs']['margin']:
            current_config['margin'] = margin
            loss_kwargs = config['loss_kwargs'].copy()
            loss_kwargs['margin'] = margin
            for lr in config["optimizer_kwargs"]["lr"]:
                current_config['lr'] = lr
                for l in config['model_kwargs']['l']:
                    current_config['l'] = l
                    for k in config['model_kwargs']['k']:
                        current_config['k'] = k
                        exp_id += 1
                        experiment_name = experiment_name_config + '_expId' + str(exp_id)
                        for seed in seeds:
                            optimizer, dataset = parse_config(config)

                            experiment_name_seed = '/' + experiment_name + '_seed_' + str(seed)
                            reshuffle_layer = RESHUFFLE_Layer(
                                input_dim=l,
                                output_dim=l,
                                relation_row_function=config['model_kwargs'][
                                    'relation_row_function'] if 'relation_row_function' in config[
                                    'model_kwargs'].keys() else None,
                                activation=torch.nn.ReLU,
                                dropout=config['model_kwargs']['dropout'],
                                aggregation_mode=config['model_kwargs']['aggregation_mode'] if 'aggregation_mode' in
                                                                                               config[
                                                                                                   'model_kwargs'].keys() else None,
                            )

                            model = RESHUFFLE_Node_GNN(
                                manual_seed=config['model_kwargs']['manual_seed'] if 'manual_seed' in config[
                                    'model_kwargs'].keys() else None,
                                triples_factory=dataset.transductive_training,
                                inference_factory=dataset.inductive_inference,
                                interaction=RESHUFFLE_Interaction,
                                l=l,
                                k=k,
                                loss=config['loss'],
                                loss_kwargs=loss_kwargs,
                                relation_row_function=config['model_kwargs'][
                                    'relation_row_function'] if 'relation_row_function' in config[
                                    'model_kwargs'].keys() else None,
                                random_seed=seed,
                                gnn_encoder=[
                                    reshuffle_layer
                                    for _ in range(num_layers)
                                ],
                            )

                            if gpu is not None:
                                model.to("cuda")

                            optimizer = optimizer(params=model.parameters(), lr=lr)

                            result_tracker = TensorBoardResultTracker(
                                experiment_path=experiment_dir + log_dir + "/" + config[
                                    "dataset"] + experiment_name_seed,
                                experiment_name=experiment_name)

                            training_loop = SLCWATrainingLoop(
                                triples_factory=dataset.transductive_training,
                                model=model,
                                optimizer=optimizer,
                                negative_sampler=config['negative_sampler'],
                                negative_sampler_kwargs=dict(
                                    num_negs_per_pos=config['negative_sampler_kwargs']['num_negs_per_pos']),
                                mode="training",  # set the mode, as training nodes are different from test nodes
                                result_tracker=result_tracker,
                            )

                            valid_evaluator = RankBasedEvaluator(
                                mode="training",  # any validation node occurs in the train graph
                            )

                            if config['stopper'] == 'early':
                                early_stopper = EarlyStopper(
                                    model=model,
                                    training_triples_factory=dataset.transductive_training,
                                    evaluation_triples_factory=dataset.inductive_validation,
                                    frequency=config['stopper_kwargs']['frequency'],
                                    patience=config['stopper_kwargs']['patience'],
                                    relative_delta=config['stopper_kwargs']['relative_delta'],
                                    result_tracker=result_tracker,
                                    evaluation_batch_size=config['evaluator_kwargs']['batch_size'],
                                    evaluator=valid_evaluator,
                                )
                            else:
                                raise Exception('Stopper %s unknown.' % config['stopper'])

                            if train:
                                try:
                                    training_loop.train(
                                        triples_factory=dataset.transductive_training,
                                        stopper=early_stopper,
                                        num_epochs=config['training_kwargs']['num_epochs'],
                                        batch_size=config['training_kwargs']['batch_size'],
                                        checkpoint_directory=experiment_dir + '/checkpoints' + "/" + config[
                                            "dataset"] + experiment_name_seed,
                                        checkpoint_frequency=config['training_kwargs']['checkpoint_frequency'],
                                        checkpoint_name=config['training_kwargs']['checkpoint_name'],
                                        checkpoint_on_failure=config['training_kwargs']['checkpoint_on_failure'],
                                        callbacks=[TrackerTrainingCallback],
                                    )
                                except MemoryError:
                                    print("There was a memory error for this configuration.")

                            if test:
                                evaluate_and_save_final_result(seed, model, dataset, config, experiment_dir,
                                                               experiment_name, current_config)


if __name__ == '__main__':
    pykeen.datasets.inductive.ilp_teru.FB_INDUCTIVE_VALIDATION_URL = "{base_url}/fb237_{version}/valid.txt"
    pykeen.datasets.inductive.ilp_teru.WN_INDUCTIVE_VALIDATION_URL = "{base_url}/WN18RR_{version}/valid.txt"
    pykeen.datasets.inductive.ilp_teru.NELL_INDUCTIVE_VALIDATION_URL = "{base_url}/nell_{version}/valid.txt"

    main(**dict(arg.split('=') for arg in sys.argv[1:]))  # kwargs
