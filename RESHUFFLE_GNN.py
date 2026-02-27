import torch
from torch import nn
from typing import Iterable, Optional, Tuple, cast
from pykeen.typing import HeadRepresentation, InductiveMode, RelationRepresentation, TailRepresentation
from pykeen.utils import get_edge_index
from RESHUFFLE import RESHUFFLE_Node
from Utils import preprocess_relation_matrix


class RESHUFFLE_Node_GNN(RESHUFFLE_Node):
    def __init__(
            self,
            *,
            manual_seed: int = None,
            gnn_encoder: Optional[Iterable[nn.Module]] = None,
            relation_row_function=None,
            **kwargs,
    ) -> None:
        """
        Initialize the model.

        :param manual_seed:
            a seed to make experiments reproducible
        :param gnn_encoder:
            an iterable of message passing layers.
        :param relation_row_function:
            can be 'square' or 'softmax'. Specifies the function to be applied row-wise over the relation matrices.
        """
        super().__init__(**kwargs)
        self.manual_seed = manual_seed
        self.relation_row_function = relation_row_function

        train_factory, inference_factory, validation_factory, test_factory = (
            kwargs.get("triples_factory"),
            kwargs.get("inference_factory"),
            kwargs.get("validation_factory"),
            kwargs.get("test_factory"),
        )
        self.gnn_encoder = nn.ModuleList(gnn_encoder)

        # Saving edge indices for all the supplied splits
        assert train_factory is not None, "train_factory must be a valid triples factory"
        self.register_buffer(name="training_edge_index", tensor=get_edge_index(triples_factory=train_factory))
        self.register_buffer(name="training_edge_type", tensor=train_factory.mapped_triples[:, 1])

        if inference_factory is not None:
            inference_edge_index = get_edge_index(triples_factory=inference_factory)
            inference_edge_type = inference_factory.mapped_triples[:, 1]

            self.register_buffer(name="validation_edge_index", tensor=inference_edge_index)
            self.register_buffer(name="validation_edge_type", tensor=inference_edge_type)
            self.register_buffer(name="testing_edge_index", tensor=inference_edge_index)
            self.register_buffer(name="testing_edge_type", tensor=inference_edge_type)
        else:
            assert (
                    validation_factory is not None and test_factory is not None
            ), "Validation and test factories must be triple factories"
            self.register_buffer(
                name="validation_edge_index", tensor=get_edge_index(triples_factory=validation_factory)
            )
            self.register_buffer(name="validation_edge_type", tensor=validation_factory.mapped_triples[:, 1])
            self.register_buffer(name="testing_edge_index", tensor=get_edge_index(triples_factory=test_factory))
            self.register_buffer(name="testing_edge_type", tensor=test_factory.mapped_triples[:, 1])

    def reset_parameters_(self):
        """Reset the GNN encoder explicitly in addition to other params."""
        super().reset_parameters_()
        if getattr(self, "gnn_encoder", None) is not None:
            for layer in self.gnn_encoder:
                if hasattr(layer, "reset_parameters"):
                    layer.reset_parameters()

    def _get_representations(
            self,
            h: Optional[torch.LongTensor],
            r: Optional[torch.LongTensor],
            t: Optional[torch.LongTensor],
            mode: Optional[InductiveMode] = None,
    ) -> Tuple[HeadRepresentation, RelationRepresentation, TailRepresentation]:
        """Get representations for head, relation and tails, in canonical shape with a GNN encoder."""
        entity_representations = self._get_entity_representations_from_inductive_mode(mode=mode)

        # Extract entity and relation representations
        x_e, x_r = entity_representations[0](), self.relation_representations[0]()

        # Random initialization of constant node features
        with torch.no_grad():
            if self.manual_seed is not None:
                torch.manual_seed(self.manual_seed)
            tmp = torch.normal(0, 1, size=x_e.size())
            clone = x_e.clone()
            clone[tmp > 0.5] = 1
            clone[tmp <= 0.5] = 0
            x_e = clone
            x_e.requires_grad = False

        x_r = preprocess_relation_matrix(x_r, self.relation_row_function)

        # Perform message passing and get updated states
        for layer in self.gnn_encoder:
            x_e, _ = layer(
                x_e=x_e,
                x_r=x_r,
                edge_index=getattr(self, f"{mode}_edge_index"),
                edge_type=getattr(self, f"{mode}_edge_type"),
            )

        # Use updated entity and relation states to extract requested IDs
        hh, rr, tt = [
            x_e[h] if h is not None else x_e,
            x_r[r] if r is not None else x_r,
            x_e[t] if t is not None else x_e,
        ]

        return cast(
            Tuple[HeadRepresentation, RelationRepresentation, TailRepresentation],
            tuple(x[0] if len(x) == 1 else x for x in (hh, rr, tt)),
        )
