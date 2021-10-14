import re
from typing import Optional

import numpy as np
import pandas as pd


def sort_lineages(lineage_list, pattern=re.compile(r"[A-Z](\.\d+)*$")):
    assert len(lineage_list) == len(
        set(lineage_list)
    ), "Arg lineage_list must contain unique lineages!"
    # extract lineages that follow the specific pattern
    identifier = [lineage for lineage in lineage_list if pattern.match(lineage)]
    max_levels = max([len(lineage.split(".")) for lineage in identifier])

    # all other identifier
    other_identifier = [
        lineage for lineage in lineage_list if not pattern.match(lineage)
    ]

    identifier_levels = []
    for lineage in identifier:
        levels = lineage.split(".")
        while len(levels) < max_levels:
            levels = levels + ["0"]

        identifier_levels.append(levels)

    for i in reversed(range(max_levels)):
        identifier_levels.sort(key=lambda x: int(x[i]) if x[i].isdigit() else x[i])

    sorted_identifier = []
    for lineage in identifier_levels:
        sorted_identifier.append(".".join([i for i in lineage if i != "0"]))

    return sorted_identifier, other_identifier



def alias(lineage, aliases, anti_alias=False):
    for a, v in aliases.items():
        if not a in ["A", "B"]:
            if not anti_alias:
                a, v = v, a
            lineage_alias = re.sub("^"+re.sub("\.","\.", a)+"(?=\\.)", v, lineage)
    return lineage_alias


def alias_lineages(lineage_list, aliases, anti_alias=False):
    assert (
        len([a for a in aliases.values() if a in lineage_list]) == 0
    ), "Aliases are not allowed to map to lineages in lineage_list!"

    assert len(list(aliases.values())) == len(
        set(aliases.values())
    ), "Two or more aliases are not allowed to map to the same lineage!"

    return [alias(lineage) for lineage in lineage_list]

def merge_lineages(
    lineage_identifier,
    lineage_counts,
    cutoff=100,
    skip=[],
    pattern=re.compile(r"[A-Z](\.\d+)*$"),
):
    def depth(x):
        return len(x.split("."))

    def parent(x):
        return ".".join(x.split(".")[:-1])

    cluster_dict = {
        k: [v, [k]]
        for k, v in sorted(
            zip(lineage_identifier, lineage_counts),
            key=lambda x: depth(x[0]),
            reverse=True,
        )
    }

    iteration = 0
    while True:
        remove_lineages = []
        for lineage, (lineage_count, cluster) in cluster_dict.items():
            if not pattern.match(lineage):
                continue

            parent_lineage = parent(lineage)
            lineage_cutoff = lineage_count < cutoff
            parent_exists = len(parent_lineage) > 0 and parent_lineage in cluster_dict

            if lineage_cutoff and parent_exists and lineage not in skip:
                cluster_dict[parent_lineage][0] += cluster_dict[lineage][0]
                cluster_dict[parent_lineage][1] += cluster_dict[lineage][1]
                remove_lineages.append(lineage)

        if len(remove_lineages) == 0:
            break
        else:
            for lineage in remove_lineages:
                # print(f'Pruning lineage {lineage}')
                del cluster_dict[lineage]
        iteration += 1

    merging_indices = {}
    merged_lineages = []
    droped_lineages = []

    sorted_identifier, other_identifier = sort_lineages(cluster_dict.keys())

    for lineage in sorted_identifier + other_identifier:
        (lineage_count, cluster) = cluster_dict[lineage]
        # drop lineages with zero counts
        if lineage_count == 0:
            droped_lineages.append(lineage)
            continue
        merged_lineages.append(lineage)
        merging_indices[lineage_identifier.index(lineage)] = [
            lineage_identifier.index(i) for i in cluster
        ]

    return merged_lineages, merging_indices, droped_lineages


def aggregate_tensor(lineage_tensor, cluster):
    lineage_red = np.zeros(
        (lineage_tensor.shape[0], lineage_tensor.shape[1], len(cluster))
    )
    for i, (k, v) in enumerate(cluster.items()):
        lineage_red[..., i] = lineage_tensor[..., v].sum(-1)

    return lineage_red


def preprocess_lineage_tensor(
    lineage_list: list,
    lineage_tensor: np.ndarray,
    aliases: Optional[dict] = None,
    vocs: list = [],
    cutoff: int = 100,
    refractory: bool = False,
):
    """
    Preprocesses the lineage tensor.

    :param lineage_list: A list of all lineages.
    :param lineage_tensor: The lineage tensor (shape: (num_location, num_time, num_lineages).
    :param aliases: A dictionary with 1:1 mappings of lineages that shall be renamed.
    :param vocs: Variants of concerns, not to be merged.
    :return: The list of merged lineages and the reduced lineage tensor.
    """
    if aliases:
        alias_list = alias_lineages(lineage_list, aliases)
    else:
        alias_list = lineage_list

    lineage_counts = np.nansum(lineage_tensor, axis=(0, 1))

    # lineages of current interest
    if refractory:
        refractory = pd.DataFrame(
            np.nansum(lineage_tensor, axis=(0)), columns=alias_list
        ).iloc[-1]
        refractory = refractory.index[refractory > 0].tolist()
        vocs += refractory

    merged_lineages, cluster, droped_lineages = merge_lineages(
        alias_list, lineage_counts, skip=vocs, cutoff=cutoff
    )
    print("Dropped lineages", droped_lineages)
    lineage_tensor_red = aggregate_tensor(lineage_tensor, cluster)
    return merged_lineages, lineage_tensor_red, cluster


def create_ancestor_matrix(lin_names, families=None):
    ancestor_matrix = np.zeros((len(lin_names), len(lin_names)))
    if families is None:
        for i, r in enumerate(range(ancestor_matrix.shape[0])):
            this_lineage = lin_names[r]
            for c in range(ancestor_matrix.shape[1]):
                other_lineage = lin_names[c]
                lin_list = this_lineage.split(".")
                lin_regex = (
                    r"^"
                    + lin_list[0]
                    + (r"\." if len(lin_list) > 1 else r"")
                    + r"\.".join(lin_list[1:])
                    + r"(?=\.|$)"
                )
                m = re.match(lin_regex, other_lineage)

                if m:
                    ancestor_matrix[r, c] = 1
    else:
        for i, r in enumerate(range(ancestor_matrix.shape[0])):
            this_lineage = lin_names[r]
            for j, c in enumerate(range(ancestor_matrix.shape[1])):
                if i == j:
                    ancestor_matrix[r, c] = 1
                if this_lineage in families:
                    other_lineage = lin_names[c]
                    lin_list = this_lineage.split(".")
                    lin_regex = (
                        r"^"
                        + lin_list[0]
                        + (r"\." if len(lin_list) > 1 else r"")
                        + r"\.".join(lin_list[1:])
                        + r"(?=\.|$)"
                    )
                    m = re.match(lin_regex, other_lineage)

                    if m:
                        ancestor_matrix[r, c] = 1

    return ancestor_matrix.T
