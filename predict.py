import torch
from torch.utils.data import DataLoader
from utils import *
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics import normalized_mutual_info_score as nmi_score
from sklearn.metrics import mean_squared_error, mean_absolute_error

MODEL_PATH = ''

def lighthggep_predict(model, test_loader, device=torch.device('cpu')):
    """
    Predict function cho Light-HGGEP voi Spatial SGC
    Dataset tra ve: patch_3ch, positions, exp, centers, section_name, local_indices
    """
    model.eval()
    model = model.to(device)
    preds = None
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting"):
            # Dataset tra ve 6 items cho test
            patch_3ch, positions, exp, centers, section_name, local_indices = batch
            patch_3ch, positions = patch_3ch.to(device), positions.to(device)
            local_indices = local_indices.to(device)
            
            pred = model(patch_3ch, positions, section_name, local_indices)
            
            if preds is None:
                preds = pred
                ct = centers
                gt = exp
            else:
                preds = torch.cat((preds, pred), dim=0)
                ct = torch.cat((ct, centers), dim=0)
                gt = torch.cat((gt, exp), dim=0)
    
    preds = preds.cpu().squeeze().numpy()
    ct = ct.cpu().squeeze().numpy()
    gt = gt.cpu().squeeze().numpy()
    
    adata = ann.AnnData(preds)
    adata.obsm['spatial'] = ct
    
    adata_gt = ann.AnnData(gt)
    adata_gt.obsm['spatial'] = ct
    
    return adata, adata_gt

def get_R(data1,data2,dim=1,func=pearsonr):
    adata1=data1.X
    adata2=data2.X
    r1,p1=[],[]
    for g in range(data1.shape[dim]):
        if dim==1:
            r,pv=func(adata1[:,g],adata2[:,g])
        elif dim==0:
            r,pv=func(adata1[g,:],adata2[g,:])
        r1.append(r)
        p1.append(pv)
    r1=np.array(r1)
    p1=np.array(p1)
    return r1,p1

def cluster(adata,label):
    idx=label!='undetermined'
    tmp=adata[idx]
    l=label[idx]
    sc.pp.pca(tmp)
    sc.tl.tsne(tmp)
    kmeans = KMeans(n_clusters=len(set(l)), init="k-means++", random_state=0).fit(tmp.obsm['X_pca'])
    p=kmeans.labels_.astype(str)
    lbl=np.full(len(adata),str(len(set(l))))
    lbl[idx]=p
    adata.obs['kmeans']=lbl
    return p,round(ari_score(p,l),3)

def get_MSE(data1, data2, dim=1):
    adata1 = data1.X
    adata2 = data2.X
    mse_list = []
    for g in range(data1.shape[dim]):
        if dim == 1:
            mse = mean_squared_error(adata1[:, g], adata2[:, g])
        elif dim == 0:
            mse = mean_squared_error(adata1[g, :], adata2[g, :])
        mse_list.append(mse)
    return np.array(mse_list)

def get_MAE(data1, data2, dim=1):
    adata1 = data1.X
    adata2 = data2.X
    mae_list = []
    for g in range(data1.shape[dim]):
        if dim == 1:
            mae = mean_absolute_error(adata1[:, g], adata2[:, g])
        elif dim == 0:
            mae = mean_absolute_error(adata1[g, :], adata2[g, :])
        mae_list.append(mae)
    return np.array(mae_list)


def get_Spearman(data1, data2, dim=1):
    """
    Tính gene-wise Spearman Correlation Coefficient.
    Cùng convention với get_R: dim=1 → lặp theo gene (cột),
    trả về (rho_array, pvalue_array) shape (n_genes,).
    Spearman bổ sung cho PCC vì không giả định phân phối tuyến tính --
    bắt được cả monotonic relationship giữa pred và gt.
    """
    adata1 = data1.X
    adata2 = data2.X
    rho_list, p_list = [], []
    for g in range(data1.shape[dim]):
        if dim == 1:
            rho, pv = spearmanr(adata1[:, g], adata2[:, g])
        elif dim == 0:
            rho, pv = spearmanr(adata1[g, :], adata2[g, :])
        rho_list.append(rho)
        p_list.append(pv)
    return np.array(rho_list), np.array(p_list)


def get_MoransI(adata, gene_idx, spatial_key='spatial'):
    """
    Tính Moran's I cho 1 gene trên toàn bộ spot, dùng tọa độ spatial làm
    weight matrix (inverse distance, cắt tại K=6 láng giềng gần nhất).

    Moran's I đo mức độ auto-correlation không gian của biểu hiện gene:
      I ≈ +1 → biểu hiện phân bố thành cụm không gian (spatially clustered)
      I ≈  0 → ngẫu nhiên
      I ≈ -1 → phân tán đều

    Trả về scalar I ∈ [-1, 1].

    Tham số:
        adata      : AnnData với adata.X shape (N, G) và adata.obsm[spatial_key] shape (N, 2)
        gene_idx   : chỉ số gene (int) hoặc tên gene (str) trong adata.var_names
        spatial_key: key trong obsm chứa tọa độ (x, y)
    """
    from sklearn.metrics.pairwise import pairwise_distances

    coords = adata.obsm[spatial_key].astype(float)   # (N, 2)
    if isinstance(gene_idx, str):
        gene_idx = list(adata.var_names).index(gene_idx)
    x = adata.X[:, gene_idx].astype(float)           # (N,)

    N = len(x)
    x_mean = x.mean()
    x_dev = x - x_mean

    # Build K-NN weight matrix (K=6, inverse distance)
    D = pairwise_distances(coords, metric='euclidean')
    K = min(6, N - 1)
    W = np.zeros((N, N), dtype=float)
    for i in range(N):
        order = np.argsort(D[i])
        neighbors = order[order != i][:K]
        for j in neighbors:
            W[i, j] = 1.0 / (D[i, j] + 1e-8)

    W_sum = W.sum()
    if W_sum == 0:
        return float('nan')

    numerator   = N * np.sum(W * np.outer(x_dev, x_dev))
    denominator = W_sum * np.sum(x_dev ** 2)
    if denominator == 0:
        return float('nan')

    return numerator / denominator


def get_MoransI_all(data_pred, data_gt, top_k=50, spatial_key='spatial'):
    """
    Tính Moran's I cho cả pred lẫn gt trên top_k gene có variance cao nhất
    (tính trên gt để chọn gene thú vị về mặt sinh học).

    Trả về dict:
        {
          'pred': np.array shape (top_k,),  -- Moran's I của từng gene trên pred
          'gt':   np.array shape (top_k,),  -- Moran's I của từng gene trên gt
          'gene_indices': np.array shape (top_k,)
        }
    """
    gt_X = data_gt.X
    var_per_gene = np.var(gt_X, axis=0)                    # variance theo từng gene
    top_indices  = np.argsort(var_per_gene)[::-1][:top_k]  # top_k gene variance cao nhất

    mi_pred, mi_gt = [], []
    for idx in top_indices:
        mi_pred.append(get_MoransI(data_pred, idx, spatial_key))
        mi_gt.append(get_MoransI(data_gt,   idx, spatial_key))

    return {
        'pred':         np.array(mi_pred),
        'gt':           np.array(mi_gt),
        'gene_indices': top_indices,
    }


def cluster_with_nmi(adata, label):
    """
    Mở rộng cluster(): tính thêm NMI bên cạnh ARI.
    Trả về (cluster_labels, ARI, NMI).

    NMI bổ sung cho ARI vì:
      - ARI hiệu chỉnh theo chance, nhạy với số cluster và size imbalance.
      - NMI đo mức độ chia sẻ thông tin giữa 2 phân hoạch, ít bị ảnh hưởng
        bởi số cluster hơn.
    """
    label = np.asarray(label)
    # HER2ST uses integer IDs and marks undetermined spots with -1.  Retain
    # support for the original string labels used by older datasets.
    unknown = -1 if np.issubdtype(label.dtype, np.number) else 'undetermined'
    idx = label != unknown
    tmp = adata[idx].copy()
    l   = label[idx]
    if len(l) < 2 or len(np.unique(l)) < 2:
        return np.array([], dtype=str), float('nan'), float('nan')
    sc.pp.pca(tmp)
    sc.tl.tsne(tmp)
    kmeans = KMeans(n_clusters=len(set(l)), init="k-means++", random_state=0).fit(tmp.obsm['X_pca'])
    p = kmeans.labels_.astype(str)

    lbl = np.full(len(adata), str(len(set(l))))
    lbl[idx] = p
    adata.obs['kmeans'] = lbl

    ari = round(ari_score(p, l), 4)
    nmi = round(nmi_score(p, l, average_method='arithmetic'), 4)
    return p, ari, nmi


def stnet_predict(model, test_loader, device=torch.device('cpu')):
    """
    Predict function cho STModel dùng HER2ST dataset.
    HER2ST test trả về: (patch, loc, exp, center)
      - patch  : (B, 3, 224, 224)
      - loc    : (B, 2)  -- tọa độ grid (x, y)
      - exp    : (B, n_genes)
      - center : (B, 2)  -- tọa độ pixel
    STModel.forward(patch, center) → pred (B, n_genes)
    """
    model.eval()
    model = model.to(device)
    preds, gts, centers = [], [], []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="STNet Predicting"):
            patch, loc, exp, center = batch
            patch  = patch.to(device)
            center = center.to(device)
            pred   = model(patch, center)
            preds.append(pred.cpu())
            gts.append(exp)
            centers.append(center.cpu())

    preds   = torch.cat(preds,   dim=0).numpy()
    gts     = torch.cat(gts,     dim=0).numpy()
    centers = torch.cat(centers, dim=0).numpy()

    adata_pred = ann.AnnData(preds)
    adata_pred.obsm['spatial'] = centers

    adata_gt = ann.AnnData(gts)
    adata_gt.obsm['spatial'] = centers

    return adata_pred, adata_gt


def histogene_predict(model, test_loader, device=torch.device('cpu')):
    """
    Predict function cho HisToGene dùng HER2ST dataset.

    HisToGene.forward(patches, centers) kỳ vọng input slide-level:
      patches : (1, N_spots, patch_dim)   -- flatten từng patch
      centers : (1, N_spots, 2)           -- tọa độ grid đã discretize

    HER2ST test trả về patch-level: (patch, loc, exp, center)
      patch  : (B, 3, H, W)
      loc    : (B, 2)   -- tọa độ grid float
      exp    : (B, n_genes)
      center : (B, 2)   -- tọa độ pixel

    Chiến lược: gom toàn bộ spots của test section vào 1 batch slide,
    flatten patch và discretize loc sang index để dùng Embedding.
    Vì test chỉ có 1 section (LOOCV), load hết rồi forward 1 lần.
    """
    model.eval()
    model = model.to(device)

    all_patches, all_locs, all_exps, all_centers = [], [], [], []
    for batch in test_loader:
        patch, loc, exp, center = batch
        all_patches.append(patch)
        all_locs.append(loc)
        all_exps.append(exp)
        all_centers.append(center)

    # Gom thành 1 tensor
    patches = torch.cat(all_patches, dim=0)   # (N, 3, H, W)
    locs    = torch.cat(all_locs,    dim=0)   # (N, 2)
    exps    = torch.cat(all_exps,    dim=0)   # (N, n_genes)
    centers = torch.cat(all_centers, dim=0)   # (N, 2)

    # The model was trained with centred 112 px crops (patch_dim=3*112*112).
    # Keep inference identical even though HER2ST stores 224 px patches.
    if patches.shape[-2:] != (112, 112):
        h, w = patches.shape[-2:]
        top, left = (h - 112) // 2, (w - 112) // 2
        patches = patches[:, :, top:top + 112, left:left + 112]

    # Flatten patch: (N, 3*112*112) → thêm batch dim → (1, N, patch_dim)
    N = patches.shape[0]
    # Centre-cropping can create a non-contiguous tensor; reshape preserves
    # values while safely flattening it for the patch embedding.
    patch_flat = patches.reshape(N, -1).unsqueeze(0).to(device)  # (1, N, patch_dim)

    # Discretize tọa độ grid sang long index cho Embedding
    # HisToGene dùng n_pos=64 → clamp về [0, 63]
    locs_long = locs.long().clamp(0, 63).unsqueeze(0).to(device)  # (1, N, 2)

    with torch.no_grad():
        pred = model(patch_flat, locs_long)   # (1, N, n_genes)
    pred = pred.squeeze(0).cpu().numpy()      # (N, n_genes)

    centers_np = centers.numpy()
    exps_np    = exps.numpy()

    adata_pred = ann.AnnData(pred)
    adata_pred.obsm['spatial'] = centers_np

    adata_gt = ann.AnnData(exps_np)
    adata_gt.obsm['spatial'] = centers_np

    return adata_pred, adata_gt
