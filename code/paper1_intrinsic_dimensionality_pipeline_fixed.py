#!/usr/bin/env python3
"""
Paper 1 analysis pipeline
========================
Working scientific question:
Can the intrinsic dimensionality of multilayer nanophotonic response manifolds
predict the data and model complexity required for accurate forward modeling?

Input: Datasets(2).zip (or an extracted Datasets directory)
Expected multilayer pairs: 8,10,12,14,16,18,20 layers, 50,000 samples each,
200 response points per sample.

Outputs:
  qc_summary.csv
  pca_dimension_summary.csv
  ae_reconstruction_results.csv
  ae_dimension_summary.csv
  learning_curve_results.csv
  capacity_results.csv
  correlation_summary.csv
  figures/*.png and *.pdf

Modes:
  --mode pilot : one seed, reduced grids, quick validation
  --mode full  : 3 seeds, publication analysis grids

Example:
  python paper1_intrinsic_dimensionality_pipeline.py --zip Datasets(2).zip --mode pilot
  python paper1_intrinsic_dimensionality_pipeline.py --zip Datasets(2).zip --mode full
"""

from __future__ import annotations
import argparse, json, math, os, random, re, shutil, zipfile
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as e:
    raise SystemExit("PyTorch is required. Install with: pip install torch") from e

LAYERS = [8, 10, 12, 14, 16, 18, 20]
EXPECTED_SAMPLES = 50_000
EXPECTED_RESPONSE_DIM = 200


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_csv_matrix(path: Path) -> np.ndarray:
    # CSVs are variables/frequencies x samples; transpose to samples x features.
    arr = pd.read_csv(path, header=None).to_numpy(dtype=np.float32)
    return arr.T


def locate_dataset_root(zip_path: Path | None, data_dir: Path | None, work_dir: Path) -> Path:
    if data_dir is not None:
        return data_dir
    if zip_path is None:
        raise ValueError("Provide --zip or --data-dir")
    extract_dir = work_dir / "extracted"
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)
    candidates = list(extract_dir.rglob("SiO2_TiO2_8_layer_50K_200_points_RS.csv"))
    if not candidates:
        raise FileNotFoundError("Could not locate multilayer CSVs after extracting ZIP")
    return candidates[0].parent


def paths_for_layer(root: Path, layer: int):
    ds = root / f"SiO2_TiO2_{layer}_layer_50K_200_points_DS.csv"
    rs = root / f"SiO2_TiO2_{layer}_layer_50K_200_points_RS.csv"
    if not ds.exists() or not rs.exists():
        raise FileNotFoundError(f"Missing files for {layer} layers")
    return ds, rs


def qc_all(root: Path, out: Path) -> pd.DataFrame:
    rows = []
    for L in LAYERS:
        ds_path, rs_path = paths_for_layer(root, L)
        X = read_csv_matrix(ds_path)
        Y = read_csv_matrix(rs_path)
        rows.append({
            "layers": L,
            "n_design_samples": X.shape[0],
            "n_design_features": X.shape[1],
            "n_response_samples": Y.shape[0],
            "n_response_features": Y.shape[1],
            "design_min": float(np.nanmin(X)),
            "design_max": float(np.nanmax(X)),
            "response_min": float(np.nanmin(Y)),
            "response_max": float(np.nanmax(Y)),
            "design_nan": int(np.isnan(X).sum()),
            "response_nan": int(np.isnan(Y).sum()),
            "design_inf": int(np.isinf(X).sum()),
            "response_inf": int(np.isinf(Y).sum()),
            "duplicate_designs": int(pd.DataFrame(X).duplicated().sum()),
            "duplicate_responses": int(pd.DataFrame(Y).duplicated().sum()),
            "shape_ok": bool(X.shape == (EXPECTED_SAMPLES, L) and
                             Y.shape == (EXPECTED_SAMPLES, EXPECTED_RESPONSE_DIM)),
        })
    df = pd.DataFrame(rows)
    df.to_csv(out / "qc_summary.csv", index=False)
    if not df["shape_ok"].all():
        raise RuntimeError("QC failed. Inspect qc_summary.csv before training models.")
    return df


def fixed_split(n: int, seed: int = 20260816):
    idx = np.arange(n)
    train_idx, temp_idx = train_test_split(idx, test_size=0.20, random_state=seed, shuffle=True)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, random_state=seed, shuffle=True)
    return train_idx, val_idx, test_idx


def pca_dimensions(root: Path, out: Path, max_fit_samples: int = 50000) -> pd.DataFrame:
    rows = []
    for L in LAYERS:
        _, rs_path = paths_for_layer(root, L)
        Y = read_csv_matrix(rs_path)
        train_idx, _, _ = fixed_split(len(Y))
        fit_idx = train_idx[:min(max_fit_samples, len(train_idx))]
        Yfit = Y[fit_idx].astype(np.float64, copy=False)
        # PCA baseline is deliberately linear and independent of AE architecture.
        pca = PCA(n_components=min(Yfit.shape), svd_solver="randomized", random_state=17)
        pca.fit(Yfit)
        evr = pca.explained_variance_ratio_
        cum = np.cumsum(evr)
        eig = pca.explained_variance_.astype(np.float64)
        participation_ratio = float((eig.sum() ** 2) / np.square(eig).sum())
        row = {
            "layers": L,
            "participation_ratio": participation_ratio,
            "pca_dim_95": int(np.searchsorted(cum, 0.95) + 1),
            "pca_dim_99": int(np.searchsorted(cum, 0.99) + 1),
            "pca_dim_995": int(np.searchsorted(cum, 0.995) + 1),
            "pca_dim_999": int(np.searchsorted(cum, 0.999) + 1),
            "pc1_variance_fraction": float(evr[0]),
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out / "pca_dimension_summary.csv", index=False)
    return df


class AutoEncoder(nn.Module):
    def __init__(self, input_dim=200, latent_dim=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, input_dim),
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))


class ForwardNet(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden: list[int]):
        super().__init__()
        dims = [input_dim] + hidden + [output_dim]
        modules = []
        for a, b in zip(dims[:-2], dims[1:-1]):
            modules += [nn.Linear(a, b), nn.ReLU()]
        modules += [nn.Linear(dims[-2], dims[-1])]
        self.net = nn.Sequential(*modules)
    def forward(self, x):
        return self.net(x)


def normalize_design(Xtr, Xv, Xte):
    lo = Xtr.min(axis=0, keepdims=True)
    hi = Xtr.max(axis=0, keepdims=True)
    span = np.where((hi - lo) == 0, 1.0, hi - lo)
    return (Xtr-lo)/span, (Xv-lo)/span, (Xte-lo)/span


def response_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    err = y_true - y_pred
    mse = float(np.mean(err ** 2))
    mae = float(np.mean(np.abs(err)))
    var = float(np.var(y_true))
    nmse = mse / var if var > 0 else np.nan
    sse = float(np.sum(err ** 2))
    sst = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - sse/sst if sst > 0 else np.nan
    return mse, mae, nmse, r2


def train_torch_model(model, Xtr, Ytr, Xv, Yv, *, epochs, batch_size, lr, patience, device):
    model.to(device)
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(Xtr.astype(np.float32)), torch.from_numpy(Ytr.astype(np.float32))),
        batch_size=batch_size, shuffle=True, pin_memory=torch.cuda.is_available())
    Xv_t = torch.from_numpy(Xv.astype(np.float32)).to(device)
    Yv_t = torch.from_numpy(Yv.astype(np.float32)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    best = math.inf; best_state = None; stale = 0
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vloss = loss_fn(model(Xv_t), Yv_t).item()
        if vloss < best - 1e-8:
            best = vloss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, epoch + 1, best


def predict_numpy(model, X, device, batch=4096):
    model.eval(); out=[]
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb=torch.from_numpy(X[i:i+batch].astype(np.float32)).to(device)
            out.append(model(xb).cpu().numpy())
    return np.vstack(out)


def ae_analysis(root: Path, out: Path, mode: str, device: str):
    seeds = [11] if mode == "pilot" else [11, 29, 47]
    latent_dims = [1,2,3,4,5,6,8,10,12,16,20] if mode == "pilot" else [1,2,3,4,5,6,7,8,9,10,12,14,16,20,24,30,40]
    epochs = 80 if mode == "pilot" else 200
    rows=[]
    for L in LAYERS:
        _, rs_path = paths_for_layer(root, L)
        Y=read_csv_matrix(rs_path)
        tr, va, te = fixed_split(len(Y))
        Ytr,Yv,Yte=Y[tr],Y[va],Y[te]
        for z in latent_dims:
            for seed in seeds:
                seed_everything(seed)
                model=AutoEncoder(200,z)
                model, ne, bv=train_torch_model(model,Ytr,Ytr,Yv,Yv,epochs=epochs,batch_size=512,lr=1e-3,patience=15,device=device)
                pred=predict_numpy(model,Yte,device)
                mse,mae,nmse,r2=response_metrics(Yte,pred)
                rows.append({"layers":L,"latent_dim":z,"seed":seed,"epochs":ne,
                             "val_mse":bv,"test_mse":mse,"test_mae":mae,"test_nmse":nmse,"test_r2":r2})
                pd.DataFrame(rows).to_csv(out/"ae_reconstruction_results.csv",index=False)
    raw=pd.DataFrame(rows)
    summary=[]
    for L,g in raw.groupby("layers"):
        agg=g.groupby("latent_dim").agg(test_r2_mean=("test_r2","mean"),test_r2_std=("test_r2","std"),
                                           test_nmse_mean=("test_nmse","mean"),test_nmse_std=("test_nmse","std")).reset_index()
        row={"layers":L}
        for thr,name in [(0.99,"99"),(0.995,"995"),(0.999,"999")]:
            ok=agg.loc[agg.test_r2_mean>=thr,"latent_dim"]
            row[f"ae_dim_r2_{name}"]=int(ok.min()) if len(ok) else np.nan
        # Knee-like dimension: maximum perpendicular distance from line joining endpoints in log NMSE curve.
        x=agg.latent_dim.to_numpy(float); y=np.log10(np.clip(agg.test_nmse_mean.to_numpy(float),1e-12,None))
        x0=(x-x.min())/(x.max()-x.min()); y0=(y-y.min())/(y.max()-y.min()+1e-12)
        p1=np.array([x0[0],y0[0]]); p2=np.array([x0[-1],y0[-1]])
        v=p2-p1; denom=np.linalg.norm(v)+1e-12
        # Perpendicular distance from each 2D point to the line through p1 and p2.
        # Avoid np.cross on 2D vectors because NumPy 2.x requires 3D vectors.
        pts=np.column_stack([x0,y0])
        rel=pts-p1
        dist=np.abs(v[0]*rel[:,1]-v[1]*rel[:,0])/denom
        row["ae_knee_dim"]=int(x[np.argmax(dist)])
        summary.append(row)
    sdf=pd.DataFrame(summary); sdf.to_csv(out/"ae_dimension_summary.csv",index=False)
    return raw,sdf


def forward_learning_curves(root: Path, out: Path, mode: str, device: str):
    seeds=[11] if mode=="pilot" else [11,29,47]
    sizes=[1000,5000,10000,30000] if mode=="pilot" else [500,1000,2500,5000,10000,20000,30000,40000]
    hidden=[128,128,128]
    epochs=100 if mode=="pilot" else 250
    rows=[]
    for L in LAYERS:
        ds_path,rs_path=paths_for_layer(root,L)
        X,Y=read_csv_matrix(ds_path),read_csv_matrix(rs_path)
        tr,va,te=fixed_split(len(X))
        Xtr,Xv,Xte=X[tr],X[va],X[te]; Ytr,Yv,Yte=Y[tr],Y[va],Y[te]
        Xtr,Xv,Xte=normalize_design(Xtr,Xv,Xte)
        for n in sizes:
            n=min(n,len(Xtr))
            for seed in seeds:
                seed_everything(seed)
                rng=np.random.default_rng(seed)
                sub=rng.choice(len(Xtr),size=n,replace=False)
                model=ForwardNet(L,200,hidden)
                model,ne,bv=train_torch_model(model,Xtr[sub],Ytr[sub],Xv,Yv,epochs=epochs,batch_size=512,lr=1e-3,patience=20,device=device)
                pred=predict_numpy(model,Xte,device)
                mse,mae,nmse,r2=response_metrics(Yte,pred)
                rows.append({"layers":L,"train_samples":n,"seed":seed,"model":"128x3","epochs":ne,
                             "test_mse":mse,"test_mae":mae,"test_nmse":nmse,"test_r2":r2})
                pd.DataFrame(rows).to_csv(out/"learning_curve_results.csv",index=False)
    return pd.DataFrame(rows)


def capacity_analysis(root: Path, out: Path, mode: str, device: str):
    seeds=[11] if mode=="pilot" else [11,29,47]
    architectures={
        "small_64x2":[64,64],
        "medium_128x3":[128,128,128],
        "large_256x3":[256,256,256],
    }
    if mode=="full": architectures["xlarge_512x4"]=[512,512,512,512]
    epochs=120 if mode=="pilot" else 300
    rows=[]
    for L in LAYERS:
        ds_path,rs_path=paths_for_layer(root,L)
        X,Y=read_csv_matrix(ds_path),read_csv_matrix(rs_path)
        tr,va,te=fixed_split(len(X))
        Xtr,Xv,Xte=X[tr],X[va],X[te]; Ytr,Yv,Yte=Y[tr],Y[va],Y[te]
        Xtr,Xv,Xte=normalize_design(Xtr,Xv,Xte)
        for name,hidden in architectures.items():
            for seed in seeds:
                seed_everything(seed)
                model=ForwardNet(L,200,hidden)
                nparams=sum(p.numel() for p in model.parameters())
                model,ne,bv=train_torch_model(model,Xtr,Ytr,Xv,Yv,epochs=epochs,batch_size=512,lr=1e-3,patience=25,device=device)
                pred=predict_numpy(model,Xte,device)
                mse,mae,nmse,r2=response_metrics(Yte,pred)
                rows.append({"layers":L,"architecture":name,"parameters":nparams,"seed":seed,"epochs":ne,
                             "test_mse":mse,"test_mae":mae,"test_nmse":nmse,"test_r2":r2})
                pd.DataFrame(rows).to_csv(out/"capacity_results.csv",index=False)
    return pd.DataFrame(rows)


def derive_correlations(out: Path):
    pca=pd.read_csv(out/"pca_dimension_summary.csv")
    ae=pd.read_csv(out/"ae_dimension_summary.csv") if (out/"ae_dimension_summary.csv").exists() else None
    lc=pd.read_csv(out/"learning_curve_results.csv") if (out/"learning_curve_results.csv").exists() else None
    cap=pd.read_csv(out/"capacity_results.csv") if (out/"capacity_results.csv").exists() else None
    base=pca.copy()
    if ae is not None: base=base.merge(ae,on="layers",how="left")
    # Operational learnability endpoints, intentionally reported at several thresholds.
    if lc is not None:
        a=lc.groupby(["layers","train_samples"])["test_r2"].mean().reset_index()
        for thr,label in [(0.99,"99"),(0.995,"995"),(0.999,"999")]:
            vals=[]
            for L,g in a.groupby("layers"):
                ok=g[g.test_r2>=thr]
                vals.append((L, ok.train_samples.min() if len(ok) else np.nan))
            base=base.merge(pd.DataFrame(vals,columns=["layers",f"samples_for_r2_{label}"]),on="layers",how="left")
    if cap is not None:
        a=cap.groupby(["layers","architecture","parameters"])["test_r2"].mean().reset_index()
        for thr,label in [(0.99,"99"),(0.995,"995"),(0.999,"999")]:
            vals=[]
            for L,g in a.groupby("layers"):
                ok=g[g.test_r2>=thr]
                vals.append((L, ok.parameters.min() if len(ok) else np.nan))
            base=base.merge(pd.DataFrame(vals,columns=["layers",f"params_for_r2_{label}"]),on="layers",how="left")
    base.to_csv(out/"paper1_master_summary.csv",index=False)

    predictors=[c for c in ["participation_ratio","pca_dim_99","pca_dim_995","pca_dim_999","ae_knee_dim","ae_dim_r2_99","ae_dim_r2_995","ae_dim_r2_999"] if c in base]
    outcomes=[c for c in base if c.startswith("samples_for_") or c.startswith("params_for_")]
    rows=[]
    for x in predictors:
        for y in outcomes:
            d=base[[x,y]].dropna()
            if len(d)>=4 and d[x].nunique()>1 and d[y].nunique()>1:
                pr,pp=pearsonr(d[x],d[y]); sr,sp=spearmanr(d[x],d[y])
                rows.append({"predictor":x,"outcome":y,"n":len(d),"pearson_r":pr,"pearson_p":pp,
                             "spearman_rho":sr,"spearman_p":sp})
    pd.DataFrame(rows).to_csv(out/"correlation_summary.csv",index=False)
    return base


def figures(out: Path):
    figdir=out/"figures"; figdir.mkdir(exist_ok=True)
    pca=pd.read_csv(out/"pca_dimension_summary.csv")
    plt.figure(figsize=(7,5));
    plt.plot(pca.layers,pca.pca_dim_99,"o-",label="PCA 99%")
    plt.plot(pca.layers,pca.pca_dim_999,"s-",label="PCA 99.9%")
    plt.plot(pca.layers,pca.participation_ratio,"^-",label="Participation ratio")
    plt.xlabel("Number of layers"); plt.ylabel("Dimension estimate"); plt.legend(); plt.tight_layout()
    plt.savefig(figdir/"Fig1_linear_dimension_vs_layers.png",dpi=300); plt.savefig(figdir/"Fig1_linear_dimension_vs_layers.pdf"); plt.close()

    if (out/"ae_reconstruction_results.csv").exists():
        ae=pd.read_csv(out/"ae_reconstruction_results.csv")
        a=ae.groupby(["layers","latent_dim"])["test_nmse"].mean().reset_index()
        plt.figure(figsize=(8,6))
        for L,g in a.groupby("layers"): plt.semilogy(g.latent_dim,g.test_nmse,"o-",label=f"{L} layers")
        plt.xlabel("Autoencoder latent dimension"); plt.ylabel("Test NMSE"); plt.legend(ncol=2); plt.tight_layout()
        plt.savefig(figdir/"Fig2_ae_reconstruction_curves.png",dpi=300); plt.savefig(figdir/"Fig2_ae_reconstruction_curves.pdf"); plt.close()

    if (out/"learning_curve_results.csv").exists():
        lc=pd.read_csv(out/"learning_curve_results.csv")
        a=lc.groupby(["layers","train_samples"])["test_nmse"].mean().reset_index()
        plt.figure(figsize=(8,6))
        for L,g in a.groupby("layers"): plt.loglog(g.train_samples,g.test_nmse,"o-",label=f"{L} layers")
        plt.xlabel("Training samples"); plt.ylabel("Forward-model test NMSE"); plt.legend(ncol=2); plt.tight_layout()
        plt.savefig(figdir/"Fig3_learning_curves.png",dpi=300); plt.savefig(figdir/"Fig3_learning_curves.pdf"); plt.close()

    if (out/"capacity_results.csv").exists():
        cp=pd.read_csv(out/"capacity_results.csv")
        a=cp.groupby(["layers","parameters"])["test_nmse"].mean().reset_index()
        plt.figure(figsize=(8,6))
        for L,g in a.groupby("layers"): plt.loglog(g.parameters,g.test_nmse,"o-",label=f"{L} layers")
        plt.xlabel("Trainable parameters"); plt.ylabel("Forward-model test NMSE"); plt.legend(ncol=2); plt.tight_layout()
        plt.savefig(figdir/"Fig4_capacity_curves.png",dpi=300); plt.savefig(figdir/"Fig4_capacity_curves.pdf"); plt.close()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--zip",type=Path,default=None)
    ap.add_argument("--data-dir",type=Path,default=None)
    ap.add_argument("--out",type=Path,default=Path("paper1_results"))
    ap.add_argument("--mode",choices=["pilot","full"],default="pilot")
    ap.add_argument("--skip-ae",action="store_true")
    ap.add_argument("--skip-learning-curves",action="store_true")
    ap.add_argument("--skip-capacity",action="store_true")
    args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    root=locate_dataset_root(args.zip,args.data_dir,args.out)
    device="cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Dataset root: {root}")
    qc=qc_all(root,args.out); print("QC passed")
    pca=pca_dimensions(root,args.out); print(pca.to_string(index=False))
    if not args.skip_ae: ae_analysis(root,args.out,args.mode,device)
    if not args.skip_learning_curves: forward_learning_curves(root,args.out,args.mode,device)
    if not args.skip_capacity: capacity_analysis(root,args.out,args.mode,device)
    master=derive_correlations(args.out)
    figures(args.out)
    print("\nMaster summary:\n",master.to_string(index=False))
    print(f"\nDone. Results written to: {args.out.resolve()}")

if __name__=="__main__":
    main()
