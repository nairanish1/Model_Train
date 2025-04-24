import torch
from torch import nn, Tensor
import e3nn
from e3nn import o3
from typing import Dict, List, Optional, Tuple, Union

def get_edges_for_l3_and_L(l3, L, optimize_speed=True):
    import numpy as np
    import networkx as nx

    edges = []
    for l1 in range(L + 1):
        for l2 in range(l1, L + 1):
            if l3 >= np.abs(l1 - l2) and l3 <= l1 + l2:
                if optimize_speed:
                    edges.append((l1, l2, (2 * l1 + 1) * (2 * l2 + 1)))
                else:
                    edges.append((l1, l2, 1))

    G = nx.Graph()
    G.add_weighted_edges_from(edges)

    MST = nx.minimum_spanning_tree(G, weight="weight")

    # Add self-connections
    for l in range(L + 1):
        if l3 <= l + l:
            MST.add_edge(l, l)

    # Organize connections with l1 >= l2
    edges = [(max(edge), min(edge)) for edge in MST.edges]

    # Sort connections
    edges = list(sorted(edges))

    return edges

def get_efficient_connections(
    L_in: int, L_out: Union[int, Tuple[int, int]]
) -> Dict[int, Dict[int, List[int]]]:
    if isinstance(L_out, int):
        L_out = (0, L_out)

    connections = {}
    for l3 in range(L_out[0], L_out[1] + 1):
        edges = get_edges_for_l3_and_L(l3, L_in)
        for edge in edges:
            l1, l2 = edge[0], edge[1]
            if l1 not in connections:
                connections[l1] = {}
            if l2 not in connections[l1]:
                connections[l1][l2] = []
            connections[l1][l2].append(l3)
    return connections

class TP_nonlinearity(torch.nn.Module):
    """
    Implements an SO(3) tensor product of a tensor with itself.
    """
    def __init__(
        self,
        irreps_in: o3.Irreps,
        w3j_matrices: Dict[int, Tensor],
        filter_ir_out: Optional[List[Union[str, o3.Irrep]]] = None,
        ls_rule: str = "full",
        channel_rule: str = "full",
        filter_symmetric: bool = True,
    ):
        super().__init__()

        self.irreps_in = irreps_in
        self.filter_symmetric = filter_symmetric
        self.w3j_matrices = w3j_matrices

        # Instead of using irr.ir.p we iterate and safely unpack each element.
        self._check_parity(irreps_in)

        # Use the available list of degrees from the Irreps object.
        self.all_ls = sorted(list(set(irreps_in.ls)))
        
        assert ls_rule in ["full", "elementwise", "efficient"]
        self.ls_rule = ls_rule

        assert channel_rule in ["full", "elementwise"]
        self.channel_rule = channel_rule

        if filter_ir_out is not None:
            filter_ir_out = [o3.Irrep(ir) if isinstance(ir, str) else ir for ir in filter_ir_out]

        if ls_rule in ["full", "elementwise"]:
            out = []
            for item1 in irreps_in:
                if isinstance(item1, tuple):
                    mul1, ir1 = item1
                else:
                    mul1, ir1 = 1, item1
                for item2 in irreps_in:
                    if isinstance(item2, tuple):
                        mul2, ir2 = item2
                    else:
                        mul2, ir2 = 1, item2
                    if filter_symmetric and ir2.l < ir1.l:
                        continue

                    if ls_rule == "elementwise" and ir1 != ir2:
                        continue

                    for ir_out in ir1 * ir2:
                        if filter_ir_out is not None and ir_out not in filter_ir_out:
                            continue

                        if channel_rule == "full":
                            out.append((mul1 * mul2, ir_out))
                        elif channel_rule == "elementwise":
                            assert mul1 == mul2
                            out.append((mul1, ir_out))

            self.irreps_out = o3.Irreps(out).sort().irreps.simplify()
            self.ls_out = [ir.ir.l for ir in self.irreps_out]
            self.set_ls_out = set(self.ls_out)
        elif ls_rule == "efficient":
            ls_in = [item[1].l if isinstance(item, tuple) else item.l for item in irreps_in]
            assert ls_in[0] == 0
            for i in range(1, len(ls_in)):
                assert ls_in[i] == ls_in[i - 1] + 1

            L_in = ls_in[-1]

            ls_out = [ir.l for ir in filter_ir_out]
            for i in range(1, len(ls_out)):
                assert ls_out[i] == ls_out[i - 1] + 1

            L_out = (ls_out[0], ls_out[-1])

            self.connections = get_efficient_connections(L_in, L_out)

            l3_mul_counts = {}
            for item1 in irreps_in:
                if isinstance(item1, tuple):
                    mul1, ir1 = item1
                else:
                    mul1, ir1 = 1, item1
                if ir1.l in self.connections:
                    for item2 in irreps_in:
                        if isinstance(item2, tuple):
                            mul2, ir2 = item2
                        else:
                            mul2, ir2 = 1, item2
                        if ir2.l in self.connections[ir1.l]:
                            for l3 in self.connections[ir1.l][ir2.l]:
                                if l3 not in l3_mul_counts:
                                    l3_mul_counts[l3] = 0
                                if channel_rule == "full":
                                    l3_mul_counts[l3] += mul1 * mul2
                                elif channel_rule == "elementwise":
                                    assert mul1 == mul2
                                    l3_mul_counts[l3] += mul1

            out = []
            for l3 in l3_mul_counts:
                out.append((l3_mul_counts[l3], "%de" % (l3)))
            self.irreps_out = o3.Irreps(out).sort().irreps.simplify()
            self.ls_out = [ir.ir.l for ir in self.irreps_out]
            self.set_ls_out = set(self.ls_out)

    def _check_parity(self, irreps):
        # Ensure that each element, whether it is a tuple or an Irrep, has parity 1.
        for item in irreps:
            if isinstance(item, tuple):
                _, ir = item
            else:
                ir = item
            assert ir.p == 1, f"Expected parity 1 but got parity {ir.p}"

    def forward(self, x: Dict[int, Tensor]) -> Dict[int, Tensor]:
        # Only iterate over the degrees available in x.
        available_ls = sorted(x.keys())
        output = {l3: [] for l3 in self.ls_out}

        if self.ls_rule in ["full", "elementwise"]:
            for l1 in available_ls:
                for l2 in available_ls:
                    if self.ls_rule == "elementwise" and l1 != l2:
                        continue
                    if self.filter_symmetric and l2 < l1:
                        continue

                    possible_ls = list(range(abs(l1 - l2), l1 + l2 + 1))
                    output_ls = [l for l in possible_ls if l in self.set_ls_out]
                    if len(output_ls) > 0:
                        #print(f"Processing l1={l1}, l2={l2}, output_ls={output_ls}")
                        if self.channel_rule == "full":
                            outer_product = torch.einsum("bim,bjn->bijmn", x[l1], x[l2])
                            #print(f"Outer product shape: {outer_product.shape}")
                            op_shape = outer_product.shape
                            outer_product = outer_product.reshape(op_shape[0], op_shape[1]*op_shape[2], op_shape[3], op_shape[4])
                        elif self.channel_rule == "elementwise":
                            outer_product = torch.einsum("bim,bin->bimn", x[l1], x[l2])
                            #print(f"Elementwise outer product shape: {outer_product.shape}")
                        for l3 in output_ls:
                            #print(f"Processing l3={l3}")
                            output[l3].append(
                                torch.einsum("mnM,bimn->biM", self.w3j_matrices[(l1, l2, l3)], outer_product)
                            )
        elif self.ls_rule == "efficient":
            for l1 in self.connections:
                for l2 in self.connections[l1]:
                    print(f"Efficient processing l1={l1}, l2={l2}")
                    if self.channel_rule == "full":
                        outer_product = torch.einsum("bim,bjn->bijmn", x[l1], x[l2])
                        print(f"Efficient outer product shape: {outer_product.shape}")
                        op_shape = outer_product.shape
                        outer_product = outer_product.reshape(op_shape[0], op_shape[1]*op_shape[2], op_shape[3], op_shape[4])
                    elif self.channel_rule == "elementwise":
                        outer_product = torch.einsum("bim,bin->bimn", x[l1], x[l2])
                        print(f"Efficient elementwise outer product shape: {outer_product.shape}")
                    for l3 in self.connections[l1][l2]:
                        print(f"Efficient processing l3={l3}")
                        output[l3].append(
                            torch.einsum("mnM,bimn->biM", self.w3j_matrices[(l1, l2, l3)], outer_product)
                        )
        for l3 in self.ls_out:
            if output[l3]:
                output[l3] = torch.cat(output[l3], axis=1)
            else:
                batch = next(iter(x.values())).shape[0]
                output[l3] = torch.zeros(batch, 0, device=next(iter(x.values())).device)
        return output
