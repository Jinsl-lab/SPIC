
import scanpy as sc
import anndata


def Integration(adatas, use_reps,n_epochs ,
                n_comps = 50,
                n_neighbors=None,
                local_connectivity=1.0,
                n_components:int = 2,
                spread=1.0,
                min_dist=None,
                scale=True,
                embedding=True, seed=2024, **kwargs):
    '''
    Run MAP to integrate a number of AnnData objects from various multi-omics experiments
    into a single joint dimensionally reduced space. Returns a joint object with the resulting
    embedding stored in ``.obsm[\'X_multimap\']`` (if instructed) and appropriate graphs in
    ``.obsp``. The final object will be a concatenation of the individual ones provided on
    input, so in the interest of ease of exploration it is recommended to have non-scaled data
    in ``.X``.

    Input
    -----
    adatas : list of ``AnnData``
        The objects to integrate. The ``.var`` spaces will be intersected across subsets of
        the objects to compute shared PCAs, so make sure that you have ample features in
        common between the objects. ``.X`` data will be used for computation.
    use_reps : list of ``str``
        The ``.obsm`` fields for each of the corresponding ``adatas`` to use as the
        dimensionality reduction to represent the full feature space of the object. Needs
        to be precomputed and present in the object at the time of calling the function.
    scale : ``bool``, optional (default: ``True``)
        Whether to scale the data to N(0,1) on a per-dataset basis prior to computing the
        cross-dataset PCAs. Improves integration.
    embedding : ``bool``, optional (default: ``True``)
        Whether to compute the MultiMAP embedding. If ``False``, will just return the graph,
        which can be used to compute a regular UMAP. This can produce a manifold quicker,
        but at the cost of accuracy.
    n_neighbors : ``int`` or ``None``, optional (default: ``None``)
        The number of neighbours for each node (data point) in the MultiGraph. If ``None``,
        defaults to 15 times the number of input datasets.
    n_components : ``int`` (default: 2)
        The number of dimensions of the MultiMAP embedding.
    seed : ``int`` (default: 0)
        RNG seed.
    strengths: ``list`` of ``float`` or ``None`` (default: ``None``)
        The relative contribution of each dataset to the layout of the embedding. The
        higher the strength the higher the weighting of its cross entropy in the layout loss.
        If provided, needs to be a list with one 0-1 value per dataset; if ``None``, defaults
        to 0.5 for each dataset.
    cardinality : ``float`` or ``None``, optional (default: ``None``)
        The target sum of the connectivities of each neighbourhood in the MultiGraph. If
        ``None``, defaults to ``log2(n_neighbors)``.

    The following parameter definitions are sourced from UMAP 0.5.1:

    n_epochs : int (optional, default None)
        The number of training epochs to be used in optimizing the
        low dimensional embedding. Larger values result in more accurate
        embeddings. If None is specified a value will be selected based on
        the size of the input dataset (200 for large datasets, 500 for small).
    init : string (optional, default 'spectral')
        How to initialize the low dimensional embedding. Options are:
            * 'spectral': use a spectral embedding of the fuzzy 1-skeleton
            * 'random': assign initial embedding positions at random.
            * A numpy array of initial embedding positions.
    min_dist : float (optional, default 0.1)
        The effective minimum distance between embedded points. Smaller values
        will result in a more clustered/clumped embedding where nearby points
        on the manifold are drawn closer together, while larger values will
        result on a more even dispersal of points. The value should be set
        relative to the ``spread`` value, which determines the scale at which
        embedded points will be spread out.
    spread : float (optional, default 1.0)
        The effective scale of embedded points. In combination with ``min_dist``
        this determines how clustered/clumped the embedded points are.
    set_op_mix_ratio : float (optional, default 1.0)
        Interpolate between (fuzzy) union and intersection as the set operation
        used to combine local fuzzy simplicial sets to obtain a global fuzzy
        simplicial sets. Both fuzzy set operations use the product t-norm.
        The value of this parameter should be between 0.0 and 1.0; a value of
        1.0 will use a pure fuzzy union, while 0.0 will use a pure fuzzy
        intersection.
    local_connectivity : int (optional, default 1)
        The local connectivity required -- i.e. the number of nearest
        neighbors that should be assumed to be connected at a local level.
        The higher this value the more connected the manifold becomes
        locally. In practice this should be not more than the local intrinsic
        dimension of the manifold.
    a : float (optional, default None)
        More specific parameters controlling the embedding. If None these
        values are set automatically as determined by ``min_dist`` and
        ``spread``.
    b : float (optional, default None)
        More specific parameters controlling the embedding. If None these
        values are set automatically as determined by ``min_dist`` and
        ``spread``.
    '''

    # the main thing will be pulling out the various subsets of the adatas, sticking them
    # together, running joint PCAs, and then splitting up the joint PCAs into datasets of
    # origin. to do so, let's introduce a helper .obs column in copied versions of adatas
    flagged = []
    for i, adata in enumerate(adatas):
        flagged.append(adata.copy())
        # while we're at it, may as well potentially scale our data copy
        if scale:
            sc. pp.scale(flagged[-1])
        flagged[-1].obs['multimap_index'] = i

    # call the wrapper. returns (params, connectivities, embedding), with embedding optional
    import wrapper
    mmp =wrapper.Wrapper(n_neighbors,local_connectivity,n_components,spread,min_dist,n_comps=n_comps,
                         n_epochs=n_epochs,flagged=flagged, use_reps=use_reps, embedding=embedding,
                         seed=seed ,**kwargs)#,
    # make one happy collapsed object and shove the stuff in correct places
    # outer join to capture as much gene information as possible for annotation
    adata = anndata.concat(adatas, join='outer') #保留所有特征
    if embedding:
        adata.obsm['X_multimap'] = mmp[2]
        adata.uns['sc_embedding'] = mmp[3]
        adata.uns['st_embedding'] = mmp[4]

    # the graph is weighted, the higher the better, 1 best. sounds similar to connectivities
    # TODO: slot distances into .obsp['distances']
    adata.obsp['connectivities'] = mmp[1]
    # set up .uns['neighbors'], setting method to umap as these are connectivities
    adata.uns['neighbors'] = {}
    adata.uns['neighbors']['params'] = mmp[0]
    adata.uns['neighbors']['params']['method'] = 'umap'
    adata.uns['neighbors']['distances_key'] = 'distances'
    adata.uns['neighbors']['connectivities_key'] = 'connectivities'
    return adata



##################################################################################################
