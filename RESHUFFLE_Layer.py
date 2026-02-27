from typing import Tuple
from pykeen.nn.representation import CompGCNLayer
from torch import FloatTensor, LongTensor, maximum
from Utils import relation_multiplication, preprocess_relation_matrix


class RESHUFFLE_Layer(CompGCNLayer):

    def __init__(
            self,
            input_dim=None,
            aggregation_mode=None,
            relation_row_function=None,
            **kwargs,
    ):
        super().__init__(input_dim=input_dim + 1, **kwargs)
        self.relation_row_function = relation_row_function
        self.aggregation_mode = aggregation_mode

    def message(
            self,
            x_e: FloatTensor,
            x_r: FloatTensor,
            edge_index: LongTensor,
            edge_type: LongTensor
    ) -> FloatTensor:
        """
        Perform message passing.

        :param x_e: shape: (num_entities, k, l)
            The entity representations.
        :param x_r: shape: (2 * num_relations, l, l)
            The relation representations (including inverse relations).
        :param edge_index: shape: (2, num_edges)
            The edge index, pairs of source and target entity for each triple.
        :param edge_type: shape (num_edges,)
            The edge type, i.e., relation ID, for each triple.

        :return:
            The updated entity representations.
        """
        # split edge into source and target nodes
        source, target = edge_index

        # transformation
        m = relation_multiplication(x_r[edge_type], x_e[source])

        # message aggregation
        if self.aggregation_mode is None or self.aggregation_mode[0:3] == 'max':
            a_m = x_e.new_zeros(m.shape[0], m.shape[1], m.shape[2]).scatter_reduce(dim=0, index=target.unsqueeze(
                dim=-1).unsqueeze(dim=-1).expand(m.shape[0], m.shape[1], m.shape[2]), src=m, reduce="amax")
            a_m = a_m[:x_e.shape[0]]
        else:
            raise Exception('Aggregation mode unknown.')

        # dropout
        a_m = self.drop(a_m)

        return a_m

    def forward(
            self,
            x_e: FloatTensor,
            x_r: FloatTensor,
            edge_index: LongTensor,
            edge_type: LongTensor,
    ) -> Tuple[FloatTensor, FloatTensor]:
        r"""
        Update entity and relation representations.

        :param x_e: shape: (num_entities, k, l)
            The entity representations.
        :param x_r: shape: (2 * num_relations, l, l)
            The relation representations (including inverse relations).
        :param edge_index: shape: (2, num_edges)
            The edge index, pairs of source and target entity for each triple.
        :param edge_type: shape (num_edges,)
            The edge type, i.e., relation ID, for each triple.
        """
        # prepare for inverse relations
        edge_type = 2 * edge_type

        # aggregate entity representations
        if self.aggregation_mode == 'max_self_and_selfLoop':
            r_loop = preprocess_relation_matrix(self.w_loop, self.relation_row_function)
            x_e = maximum(
                maximum(x_e, x_e @ r_loop),
                maximum(
                    self.message(x_e=x_e, x_r=x_r, edge_index=edge_index, edge_type=edge_type),
                    self.message(x_e=x_e, x_r=x_r, edge_index=edge_index.flip(0), edge_type=edge_type + 1)
                ),
            )
        elif self.aggregation_mode == 'max_self_noLoop':
            x_e = maximum(
                x_e,
                maximum(
                    self.message(x_e=x_e, x_r=x_r, edge_index=edge_index, edge_type=edge_type),
                    self.message(x_e=x_e, x_r=x_r, edge_index=edge_index.flip(0), edge_type=edge_type + 1)
                ),
            )
        else:
            raise Exception('Aggregation mode unknown.')

        return x_e, x_r
