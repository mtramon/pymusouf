#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import h5py
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import sys
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

# package module(s)
from config import STRUCT_DIR
from survey import CURRENT_SURVEY
from utils.tools import print_file_datetime, check_array_order

class UNet3D(nn.Module):
    def __init__(self,in_channels=1, out_channels=1,  base_features=16, max_levels=None, input_size=None):
        super().__init__()
       
        """
        base_features : nombre de features au premier niveau (sera doublé à chaque niveau)
        max_levels : nombre maximum de niveaux (si None, déterminé à partir de input_size)
        input_size : tuple (d, h, w) pour calculer automatiquement le nombre de niveaux
        """
        super().__init__()
        if max_levels is None and input_size is not None:
            # Calculer le nombre de niveaux possible sans que la taille devienne nulle
            max_levels = min(int(np.log2(min(input_size))) + 1, 4)  # par exemple
        features = [base_features * (2**i) for i in range(max_levels)]
        
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.pool = nn.MaxPool3d(2)
        # Encodeur
        for f in features:
            self.encoder.append(self._conv_block(in_channels, f))
            in_channels = f
        # Bottleneck
        self.bottleneck = self._conv_block(features[-1], features[-1]*2)
        # Décodeur
        for f in reversed(features):
            self.decoder.append(nn.ConvTranspose3d(f*2, f, kernel_size=2, stride=2))
            self.decoder.append(self._conv_block(f*2, f))
        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)

    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1), # padding_mode='reflect'
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1), # padding_mode='reflect'
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        skip_connections = []
        for enc in self.encoder:
            x = enc(x)
            skip_connections.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        for idx in range(0, len(self.decoder), 2):
            x = self.decoder[idx](x)  # upconv
            skip = skip_connections[idx//2]
            # Ajustement des dimensions si nécessaire
            if x.shape[2:] != skip.shape[2:]:
                # Calcul des différences (D, H, W)
                diff = [skip.shape[i+2] - x.shape[i+2] for i in range(3)]
                # Construction du padding (ordre inverse : W, H, D)
                pad = []
                for d in diff[::-1]:
                    pad.extend([max(0, d//2), max(0, d - d//2)])
                x = nn.functional.pad(x, pad)
            x = torch.cat((skip, x), dim=1)
            x = self.decoder[idx+1](x)  # double conv

        return self.final_conv(x)



# Fonction utilitaire pour sauvegarder un volume 3D en VTK (grille structurée)
def save_volume_to_vtk(input, target, output, filename, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0)):
    """
    Sauvegarde un volume numpy 3D (shape (nx, ny, nz)) au format VTK structuré.
    L'ordre des indices est (x, y, z) avec x le premier.
    """
    import vtk
    from vtk.util import numpy_support
    nx, ny, nz = input.shape
    # Grille de points
    grid = vtk.vtkStructuredGrid()
    grid.SetDimensions(nx + 1, ny + 1, nz + 1)
    points = vtk.vtkPoints()
    points.Allocate((nx + 1) * (ny + 1) * (nz + 1))
    for k in range(nz + 1):
        for j in range(ny + 1):
            for i in range(nx + 1):
                x = origin[0] + i * spacing[0]
                y = origin[1] + j * spacing[1]
                z = origin[2] + k * spacing[2]
                points.InsertNextPoint(x, y, z)
    grid.SetPoints(points)
    # Données cellule (ordre Fortran pour VTK)
    # print(check_array_order(volume))
    input_flat = input.ravel()
    target_flat = target.ravel()
    output_flat = output.ravel()

    # vol_flat = volume.reshape(-1, order='F')
    # print(check_array_order(vol_flat))
    input_vtk = numpy_support.numpy_to_vtk(input_flat, deep=True, array_type=vtk.VTK_FLOAT)
    input_vtk.SetName("input")
    grid.GetCellData().AddArray(input_vtk)
    target_vtk = numpy_support.numpy_to_vtk(target_flat, deep=True, array_type=vtk.VTK_FLOAT)
    target_vtk.SetName("target")
    grid.GetCellData().AddArray(target_vtk)
    output_vtk = numpy_support.numpy_to_vtk(output_flat, deep=True, array_type=vtk.VTK_FLOAT)
    output_vtk.SetName("output")
    grid.GetCellData().AddArray(output_vtk)
    # Écriture
    writer = vtk.vtkXMLStructuredGridWriter()
    writer.SetFileName(str(filename))
    writer.SetInputData(grid)
    writer.Write()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # parser.add_argument('--data', type=str, default=None, help='Chemin vers le fichier de données npz')
    parser.add_argument('--voxel_size', type=int, default=32, help='Taille des voxels en mètres')
    parser.add_argument('--seed', type=int, default=9506, help='Seed random number generator')
    parser.add_argument('--batch_size', type=int, default=4, help='Taille de batch')
    parser.add_argument('--epochs', type=int, default=100, help='Nombre maximal d’époques')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--patience', type=int, default=10, help='Patience pour early stopping')
    parser.add_argument('--gpu', action='store_true', help='Utiliser GPU si disponible')
    args = parser.parse_args()
    
    vs=args.voxel_size
    survey_name = CURRENT_SURVEY.name
    dir_survey = STRUCT_DIR / survey_name
    dirs = {
        "survey": dir_survey,
        "dem": dir_survey / "dem",
        "voxel": dir_survey / "voxel",
        "model": dir_survey / "model",
        "tel": dir_survey / "telescope",
    }
    dirs["post"] = dirs["model"] / "postreg"
    dirs["train"] = dirs["post"] / "training"
    dirs["train"].mkdir(parents=True, exist_ok=True)
    input_data = dirs["train"] / "training_data.npz"
    dirs["val"] = dirs["post"] / "validation"
    dirs["val"].mkdir(parents=True, exist_ok=True)
    data = np.load(input_data)
    X = data["X"]
    nmodels = X.shape[0]
    # X=X.reshape((nmodels,-1),order="F") #posterior
    y = data["y"]
    # y=y.reshape((nmodels,-1),order="F") #posterior
    mask = data["mask"].reshape(-1,order="F")
    active = data["active"]
    # print(check_array_order(X),check_array_order(y[0]), check_array_order(mask), check_array_order(active) )
    # print(X[0][mask].shape, X[0][mask][:10])
    # print(X[0,active].shape)
    nvx, nvy, nvz = data["shape"]
    nvox = nvx * nvy * nvz
    assert mask.shape == (nvox,), "Le masque doit avoir la taille nx*ny*nz"
    assert X.shape[1] == nvox, "X doit avoir la deuxième dimension = nx*ny*nz"
    assert y.shape[1] == nvox, "y doit avoir la deuxième dimension = nx*ny*nz"

    X = X.reshape(-1, 1, nvx, nvy, nvz).astype(np.float32) # ordre (batch, channels, profondeur, hauteur, largeur)
    y = y.reshape(-1, 1, nvx, nvy, nvz).astype(np.float32)
    mask_3d = mask.reshape(1, 1, nvx, nvy, nvz).astype(bool)#.astype(np.float32)
    # mask_3d = np.repeat(mask_3d, nmodels, axis=0)
    # print(mask_3d.shape)
    # print(np.all(X[mask_3d]>0), np.all(y[mask_3d]>0))
    # exit()
    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')
    print(f"Utilisation du device : {device}")

    # Normalisation : on centre-réduit sur les voxels actifs de l'ensemble d'entraînement
    # Pour éviter le data leakage, on calcule les stats sur le train uniquement après split.
    
    N = X.shape[0]
    indices = np.arange(N)
    
    np.random.seed(args.seed)

    np.random.shuffle(indices)
    n_train = int(0.7 * N)
    n_val = int(0.15 * N)
    n_test = N - n_train - n_val
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train+n_val]
    test_idx = indices[n_train+n_val:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    # print(X_train.shape, X_val.shape, X_test.shape)
    # Calcul des moyennes et écarts-types sur les voxels actifs du train
    mask_flat = mask.reshape(-1, order="F")
    trainput_flat = X_train.reshape(n_train, -1)[:, active]  # (n_train, n_active)
    mean_train = trainput_flat.mean()
    std_train = trainput_flat.std()
    np.savez(dirs["train"]/"norm_stats.npz", mean=mean_train, std=std_train)

    # Normalisation
    def normalize(x, mean, std):
        return (x - mean) / (std + 1e-8)

    X_train = normalize(X_train, mean_train, std_train)
    X_val   = normalize(X_val,   mean_train, std_train)
    X_test  = normalize(X_test,  mean_train, std_train)

    # Les cibles y sont normalisées de la même manière (optionnel, mais cohérent)
    y_train = normalize(y_train, mean_train, std_train)
    y_val   = normalize(y_val,   mean_train, std_train)
    y_test  = normalize(y_test,  mean_train, std_train)

    # Conversion en tenseurs PyTorch
    X_train = torch.from_numpy(X_train)
    y_train = torch.from_numpy(y_train)
    X_val   = torch.from_numpy(X_val)
    y_val   = torch.from_numpy(y_val)
    X_test  = torch.from_numpy(X_test)
    y_test  = torch.from_numpy(y_test)
    mask_tensor = torch.from_numpy(mask_3d)  # (1,1,nx,ny,nz)

    # Datasets et DataLoaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset   = TensorDataset(X_val, y_val)
    test_dataset  = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=args.batch_size, shuffle=False)

    model = UNet3D(in_channels=1, out_channels=1, max_levels=3).to(device)

    # -------------------- Fonction de perte avec masque --------------------
    def masked_mse_loss(pred, target, mask):
        """
        pred, target : (B,1,H,W,D)
        mask : (1,1,H,W,D) ou (B,1,H,W,D) - broadcastable
        """
        diff = (pred - target) ** 2
        diff = diff * mask  # met à zéro les voxels hors volcan
        loss = diff.sum() / (mask.sum() + 1e-8)
        return loss

    def gradient_vertical_loss(pred, target, mask):
        # Différence finie selon z (ordre 1)
        diff_z_pred = pred[:, :, :, :, 1:] - pred[:, :, :, :, :-1]
        diff_z_target = target[:, :, :, :, 1:] - target[:, :, :, :, :-1]
        loss = torch.mean(((diff_z_pred - diff_z_target) * mask[:, :, :, :, 1:])**2)
        return loss

    # -------------------- Optimiseur et scheduler --------------------
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    # -------------------- Boucle d'entraînement --------------------
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    alpha=0.5
    for epoch in tqdm(range(1, args.epochs + 1)):
        # Entraînement
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in  tqdm(train_loader, position=0, leave=True):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(X_batch)
            mse_loss = masked_mse_loss(output, y_batch, mask_tensor.to(device))
            # loss =  mse_loss 
            grad_loss = gradient_vertical_loss(output, y_batch, mask_tensor.to(device))
            loss = (1-alpha)* mse_loss +alpha* grad_loss

            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                output = model(X_batch)
                mse_loss = masked_mse_loss(output, y_batch, mask_tensor.to(device))
                # loss = mse_loss
                grad_loss = gradient_vertical_loss(output, y_batch, mask_tensor.to(device))
                loss = (1-alpha)* mse_loss + alpha* grad_loss
                val_loss += loss.item() * X_batch.size(0)
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), dirs["train"] / 'best_model.pth')
            print("  -> Meilleur modèle sauvegardé")
            # --- Sauvegarde d’un échantillon de validation en VTK ---
            if epoch < args.epochs: 
                model.eval()
                with torch.no_grad():
                    # Premier échantillon du validation set
                    X_sample, y_sample = val_dataset[0]
                    X_sample = X_sample.unsqueeze(0).to(device)  # (1,1,nx,ny,nz)
                    output = model(X_sample)

                    # Dénormalisation
                    output = output * std_train + mean_train
                    y_sample = y_sample * std_train + mean_train
                    X_sample = X_sample * std_train + mean_train

                    # Passage en numpy et suppression des dimensions batch/canal
                    X_np = X_sample.squeeze().cpu().numpy().astype(np.float32)   # (nx,ny,nz)
                    y_np = y_sample.squeeze().cpu().numpy().astype(np.float32)
                    out_np = output.squeeze().cpu().numpy().astype(np.float32)

                    # Application du masque (hors volcan → 0)
                    mask_np = mask_tensor.squeeze().cpu().numpy().astype(bool)
                    X_np[~mask_np] = 0
                    y_np[~mask_np] = 0
                    out_np[~mask_np] = 0

                    # Création du dossier de sortie et sauvegarde
                    epoch_str = f"{epoch:03d}"
                    save_volume_to_vtk(X_np, y_np, out_np, dirs["val"] / f"val0_epoch{epoch_str}.vts")
                    print(f"  Échantillon de validation sauvegardé dans {dirs['val']}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping après {epoch} époques")
                break

    # -------------------- Courbes d'apprentissage --------------------
    plt.figure()
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE masquée)')
    plt.xlim(0, len(train_losses))
    plt.ylim(0, max(max(train_losses), max(val_losses)))
    plt.legend()
    np.savez(dirs["train"]/"losses.npz", training=train_losses, validation=val_losses)
    plt.savefig(dirs["train"] / 'learning_curves.png')
    plt.show()

    # -------------------- Évaluation sur le test --------------------
    model.load_state_dict(torch.load(dirs["train"] / 'best_model.pth'))
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            output = model(X_batch)
            loss = masked_mse_loss(output, y_batch, mask_tensor.to(device))
            test_loss += loss.item() * X_batch.size(0)
    test_loss /= len(test_loader.dataset)
    print(f"Test Loss (MSE masquée) : {test_loss:.6f}")

    # Sauvegarde des prédictions pour quelques échantillons (optionnel)
    # ...

    print("Entraînement terminé.")