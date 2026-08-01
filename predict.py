import torch
from torch.utils.data import DataLoader
from utils import *
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')
from scipy.stats import pearsonr
from sklearn.metrics import adjusted_rand_score as ari_score
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