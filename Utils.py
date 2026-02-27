import os
import json
from torch import nn, squeeze, square
from pykeen.evaluation import SampledRankBasedEvaluator


def analyze_checkpoints(seed, model, dataset, config, analysis_path):
    test_evaluator = SampledRankBasedEvaluator(
        mode="testing",
        evaluation_factory=dataset.inductive_testing,
        additional_filter_triples=dataset.inductive_inference.mapped_triples,
        num_negatives=50
    )

    test_results = test_evaluator.evaluate(
        model=model,
        mapped_triples=dataset.inductive_testing.mapped_triples,
        additional_filter_triples=[
            dataset.inductive_inference.mapped_triples,
        ],
        batch_size=config['evaluator_kwargs']['batch_size'],
    )

    # Save Results:
    print("[Info] Saving results!")
    test_results = test_results.to_flat_dict()

    test_path = os.path.join(analysis_path, 'test_result_seed_' + str(seed) + '.json')

    if not os.path.exists(analysis_path):
        os.makedirs(analysis_path)
    with open(test_path, 'w') as fp:
        json.dump(test_results, fp)

    return test_results


def relation_multiplication(r, e):
    m = e @ r
    return squeeze(m, dim=-1)


def preprocess_relation_matrix(x_r, relation_row_function=None):
    if relation_row_function == "softmax" or relation_row_function is None:
        softmax = nn.Softmax(dim=(len(x_r.shape) - 2))
        x = softmax(x_r)
    elif relation_row_function == "square":
        x = square(x_r)
    else:
        raise Exception("Unknown row-activation function.")

    if len(x_r.shape) == 3:
        return x[:, :-1, :]
    elif len(x_r.shape) == 2:
        return x[:-1, :]
    else:
        raise Exception("The relation matrix should always be either 2- or 3-dimensional.")
