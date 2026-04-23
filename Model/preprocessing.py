import scanpy as sc

def spatial_coordinates_scaled(adata, key='spatial', new_key='spatial_normalized'):

    if key in adata.obsm:
        spatial_coords = adata.obsm[key]
        centered_coords = spatial_coords - spatial_coords.mean(axis=0)
        cordx = spatial_coords[:, 0]
        cordy = spatial_coords[:, 1]
        xmin = cordx.min() - 1
        ymin = cordy.min() - 1
        xmax = cordx.max() + 1
        ymax = cordy.max() + 1
        scaled_points = (spatial_coords - [xmin, ymin]) / ([xmax - xmin, ymax - ymin])

        adata.obsm[new_key] = scaled_points


def preprocess(sc_adata,
               st_adata,
               min_genes:int = 200,
               min_cells:int = 3,
               st_type: str = 'spot',
               n_features: int =4000,
               normalize: bool = True,
               select_hvg: str = 'intersection',):
    """
    pre-process the scRNA-seq and ST data (find HVGs and normalized the data)
    :param select_hvg: 'intersection' or 'union'
    :param sc_adata: AnnData object of scRNA-seq data
    :param st_adata: AnnData object of St data
    :param st_type: the type of ST data, `spot` or `image`
    :param n_features: the number of HVGs to select
    :param normalize: whether to normalize the data or not
    :return: AnnData object of processed scRNA-seq data and ST data
    """

    sc_adata.var_names_make_unique()
    st_adata.var_names_make_unique()

    sc.pp.filter_cells(sc_adata, min_genes=min_genes)
    sc.pp.filter_genes(sc_adata, min_cells=min_cells)
    sc.pp.filter_cells(st_adata, min_genes=min_genes)
    sc.pp.filter_genes(st_adata, min_cells=min_cells)

    assert sc_adata.shape[1] >= n_features, 'There are too few genes in scRNA-seq data, please check again!'
    sc.pp.highly_variable_genes(sc_adata, flavor="seurat_v3", n_top_genes=n_features)

    assert st_type in ['spot', 'image'], 'Please select the correct type of ST data, `spot` or `image`!'
    if st_type == 'spot':
        assert st_adata.shape[1] >= n_features, 'There are too few genes in ST data, please check again!'
        sc.pp.highly_variable_genes(st_adata, flavor="seurat_v3", n_top_genes=n_features)
    elif st_type == 'image':
        if st_adata.shape[1] >= n_features:
            sc.pp.highly_variable_genes(st_adata, flavor="seurat_v3", n_top_genes=n_features)
        else:
            sc.pp.highly_variable_genes(st_adata, flavor="seurat_v3", n_top_genes=st_adata.shape[1])

    if normalize:
        sc.pp.normalize_total(sc_adata, target_sum=1e4)
        sc.pp.log1p(sc_adata) # sc_adata

        sc.pp.normalize_total(st_adata, target_sum=1e4)
        sc.pp.log1p(st_adata) # sc_adata

    sc_adata.raw = sc_adata # Save the raw data
    st_adata.raw = st_adata

    sc_hvg = sc_adata.var['highly_variable'][sc_adata.var['highly_variable'] == True].index
    st_hvg = st_adata.var['highly_variable'][st_adata.var['highly_variable'] == True].index
    if select_hvg == 'intersection':
        inter_gene = set(sc_hvg).intersection(set(st_hvg))
    elif select_hvg == 'union':
        sc_gene = set(sc_adata.var_names)
        st_gene = set(st_adata.var_names)
        common_gene = set(sc_gene).intersection(set(st_gene))

        inter_gene = set(sc_hvg).union(set(st_hvg))
        inter_gene = set(inter_gene).intersection(set(common_gene))

    sc_adata = sc_adata[:, list(inter_gene)]
    st_adata = st_adata[:, list(inter_gene)]

    spatial_coordinates_scaled(sc_adata)
    spatial_coordinates_scaled(st_adata)



    sc_adata.obsm['X'] = sc_adata.X.toarray()
    st_adata.obsm['X'] = st_adata.X.toarray()

    sc_adata.obs['source'] = 'scRNA-seq'
    st_adata.obs['source'] = 'ST'



    print("sc_adata:", sc_adata)
    print("st_adata:", st_adata)
    print("Data have been pre-processed!")

    return sc_adata, st_adata


#######################################################################################################

