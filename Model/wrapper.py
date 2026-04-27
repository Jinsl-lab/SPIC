import numpy as np
import scanpy as sc

def Wrapper(n_neighbors,local_connectivity,n_components,spread,min_dist,
            n_epochs,flagged, use_reps, n_comps,embedding, seed, **kwargs):
    '''
    A function that computes the paired PCAs between the datasets to integrate, calls MultiMAP
    proper, and returns a  (parameters, connectivities, embedding) tuple. Embedding optional
    depending on ``embedding``.

    Input
    -----
    flagged : list of ``AnnData``
        Preprocessed objects to integrate. Need to have the single-dataset DRs computed at
        this stage. Need to have ``.obs[\'multimap_index\']`` defined, incrementing integers
        matching the object's index in the list. Both ``Integrate()`` and ``Batch()`` make
        these.

    All other arguments as described in ``MultiMAP.Integration()``.
    '''
    # MAP wants the shared PCAs delivered as a dictionary, with the subset indices
    # tupled up as a key. let's make that then
    joint = {}
    # process all dataset pairs
    for ind1 in np.arange(len(flagged) - 1):
        for ind2 in np.arange(ind1 + 1, len(flagged)):
            subset = (ind1, ind2)
            # collapse into a single object and run a PCA
            adata = flagged[ind1].concatenate(flagged[ind2], join='inner') 
            sc.tl.pca(adata,n_comps = n_comps)

            X_pca = adata.obsm['X_pca'].copy()

            multimap_index = adata.obs['multimap_index'].values
            del adata
            joint[subset] = []
            for i in subset:
                joint[subset].append(X_pca[multimap_index == i, :])


    Xs = []
    for adata, use_rep in zip(flagged, use_reps):
        Xs.append(adata.obsm[use_rep])

    # set seed
    np.random.seed(seed)

    # and with that, we're now truly free to call the MultiMAP function
    # need to negate embedding and provide that as graph_only for the function to understand
    from MAP import MAP
    mmp = MAP(Xs=Xs, joint=joint, n_neighbors=n_neighbors, local_connectivity=local_connectivity,
                       n_components=n_components, spread=spread, min_dist=min_dist, n_epochs=n_epochs,
                       graph_only=(not embedding), **kwargs)

    return mmp
