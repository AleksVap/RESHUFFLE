"""A wrapper which combines an interaction function with the randomly initialized entity representations."""

from typing import Any, ClassVar, Mapping, Optional
from class_resolver import HintOrType
from pykeen.constants import DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE
from pykeen.models import InductiveERModel
from pykeen.triples.triples_factory import CoreTriplesFactory
from pykeen.nn import (
    Interaction,
    Embedding,
)


class RESHUFFLE_Node(InductiveERModel):
    hpo_default: ClassVar[Mapping[str, Any]] = dict(
        embedding_dim=DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE,
    )

    def __init__(
            self,
            *,
            triples_factory: CoreTriplesFactory,
            inference_factory: CoreTriplesFactory,
            interaction: HintOrType[Interaction],
            validation_factory: Optional[CoreTriplesFactory] = None,
            test_factory: Optional[CoreTriplesFactory] = None,
            l: int,
            k: int,
            **kwargs,
    ) -> None:
        """
        Initialize the model.

        :param triples_factory:
            the triples factory of training triples. Must have create_inverse_triples set to True.
        :param inference_factory:
            the triples factory of inference triples. Must have create_inverse_triples set to True.
        :param validation_factory:
            the triples factory of validation triples. Must have create_inverse_triples set to True.
        :param test_factory:
            the triples factory of testing triples. Must have create_inverse_triples set to True.
        :param l:
            the embedding dimension, where the entities embedding shape is (k,l) and relation embedding shape is (l,l).
        :param k:
            the embedding dimension, where the entities embedding shape is (k,l) and relation embedding shape is (l,l).
        :param interaction:
            the interaction module, or a hint for it.
        :param kwargs:
            additional keyword-based arguments passed to :meth:`InductiveERModel.__init__`

        :raises ValueError:
            if the triples factory does not create inverse triples
        """
        if not triples_factory.create_inverse_triples:
            raise ValueError(
                "The provided triples factory does not create inverse triples.",
            )

        if validation_factory is None:
            validation_factory = inference_factory

        super().__init__(
            triples_factory=triples_factory,
            interaction=interaction,
            entity_representations=Embedding,
            entity_representations_kwargs=dict(shape=(k, l),),
            relation_representations=Embedding(max_id=triples_factory.num_relations, shape=(l+1, l)),
            validation_factory=validation_factory,
            testing_factory=test_factory,
            **kwargs,
        )
