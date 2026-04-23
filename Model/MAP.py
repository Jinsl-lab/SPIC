
import warnings
import scipy.sparse
import scipy.sparse.csgraph

from functionss_CUDA import *

def MAP(Xs,
             joint={},
             joint_idxs={},

             metrics=None,
             metric_kwds=None,
             joint_metrics={},

             n_neighbors=None,
             cardinality=None,
             angular=False,
             set_op_mix_ratio=1.0,
             local_connectivity=1.0,

             n_components=2,
             spread=1.0,
             min_dist=None,
             init='spectral',
             n_epochs=150,
             a=None,
             b=None,
             strengths=None,

             random_state=0,

             verbose=True,

             graph_only=False,
             ):
    '''
    Run MAP on a collection of dimensionality reduction matrices. Returns a ``(parameters,
    neighbor_graph, embedding)`` tuple, with the embedding optionally skipped if ``graph_only=True``.

    Input
    -----
    Xs : list of ``np.array``
        The dimensionality reductions of the datasets to integrate, observations as rows.

        >>> Xs = [DR_A, DR_B, DR_C]
    joint : dict of ``np.array``
        The joint dimensionality reductions generated for all pair combinations of the input
        datasets. The keys are to be two-integer tuples, specifying the indices of the two
        datasets in ``Xs``

        >>> joint = {(0,1):DR_AB, (0,2):DR_AC, (1,2):DR_BC}
    graph_only : ``bool``, optional (default: ``False``)
        If ``True``, skip producing the embedding and only return the neighbour graph.

    All other arguments as described in ``MultiMAP.Integration()``.
    '''

    # turn off warnings if we're not verbose
    if joint is None:
        joint = {}
    if not verbose:
        warnings.simplefilter('ignore')

    for i in range(len(Xs)):
        if not scipy.sparse.issparse(Xs[i]):
            Xs[i] = np.array(Xs[i])
    len_Xs = [len(i) for i in Xs]

    if not joint:
        joint = {tuple(range(len(Xs))): Xs}

    joint = elaborate_relation_dict(joint, list_elems=True)
    joint_idxs = elaborate_relation_dict(joint_idxs, list_elems=True)
    joint_metrics = elaborate_relation_dict(joint_metrics, list_elems=False)
    for k in joint.keys():
        joint[k] = [i.toarray() if scipy.sparse.issparse(i) else np.array(i) for i in joint[k]]
        if k not in joint_idxs.keys():
            if k[::-1] in joint_idxs.keys():
                joint_idxs[k] = joint_idxs[k[::-1]]
            else:
                joint_idxs[k] = [np.arange(len_Xs[k[0]]), np.arange(len_Xs[k[1]])]
        if k not in joint_metrics.keys():
            if k[::-1] in joint_metrics.keys():
                joint_metrics[k] = joint_metrics[k[::-1]]
            else:
                joint_metrics[k] = 'l2'

    if metrics is None:
        metrics = ['correlation' for i in range(len(Xs))]
    if metric_kwds is None:
        metric_kwds = [{} for i in range(len(Xs))]

    if n_neighbors is None:
        n_neighbors = 15 * len(Xs)
    if cardinality is None:
        cardinality = np.log2(n_neighbors)
    if min_dist is None:
        min_dist = 0.5 * 15 / n_neighbors

    if scipy.sparse.issparse(init):
        init = init.toarray()
    else:
        init = np.array(init)
    if n_epochs is None:
        if np.sum(len_Xs) <= 10000:
            n_epochs = 500
        else:
            n_epochs = 200
    if a is None or b is None:
        a, b = find_ab_params(spread, min_dist)


    if strengths is None:
        strengths = np.ones(len(Xs)) * 0.5
    weights = find_weights(strengths, len_Xs, joint_idxs)


    if verbose:
        print("Constructing fuzzy simplicial sets ...") #构造模糊简单集
    graphs, joint_graphs, full_graph, weights = fuzzy_simplicial_set(
        Xs,
        joint,
        joint_idxs,
        weights,
        n_neighbors,
        cardinality,
        metrics,
        metric_kwds,
        joint_metrics,
        angular,
        set_op_mix_ratio,
        local_connectivity,
        n_epochs,
        random_state,
        verbose=False
    )

    # set up parameter output
    params = {'n_neighbors': n_neighbors,
              'metric': metrics[0],
              'multimap': {'cardinality': cardinality,
                           'set_op_mix_ratio': set_op_mix_ratio,
                           'local_connectivity': local_connectivity,
                           'n_components': n_components,
                           'spread': spread,
                           'min_dist': min_dist,
                           'init': init,
                           'n_epochs': n_epochs,
                           'a': a,
                           'b': b,
                           'strengths': strengths,
                           'random_state': random_state}}

    # return parameter and graph tuple
    # TODO: add the distances graph to this once it exists
    if graph_only:
        return (params, full_graph)

    if verbose:
        print("Initializing embedding ...")
    embeddings = init_layout(
        init,
        Xs,
        graphs,
        n_components,
        metrics,
        metric_kwds,
        random_state
    )

    if verbose:
        print("Optimizing embedding ...")
    embeddings = optimize_layout(embeddings, graphs, joint_graphs, weights, n_epochs, a, b, random_state,
                                            gamma=1.0, initial_alpha=1.0, negative_sample_rate=5.0, parallel=False,
                                            verbose=verbose)
    sc_embedding = embeddings[0]
    st_embedding = embeddings[1]
    # undo warning reset
    if not verbose:
        warnings.resetwarnings()

    # return an embedding/graph/parameters tuple
    # TODO: add the distances graph to this once it exists
    return (params, full_graph, np.concatenate(embeddings),sc_embedding, st_embedding)




