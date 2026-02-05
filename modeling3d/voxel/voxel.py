# -*- coding: utf-8 -*-
#!/usr/bin/env python3

from matplotlib.axes import Axes
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import os
from pathlib import Path
import pickle
from scipy.interpolate import griddata,LinearNDInterpolator
from scipy.integrate import dblquad, nquad
import time
from tqdm import tqdm
from typing import Union
from vtk.util import numpy_support

#personal package(s)
from cli import get_common_args
from raypath import RayPath
from survey import CURRENT_SURVEY
from telescope import Telescope, DICT_TEL, str2telescope
from utils.common import Common

params = {'legend.fontsize': 'xx-large',
         'axes.labelsize': 'xx-large',
         'axes.titlesize':'xx-large',
         'xtick.labelsize':'xx-large',
         'ytick.labelsize':'xx-large',
         'axes.labelpad':10,
         'mathtext.fontset': 'stix',
         'font.family': 'STIXGeneral'
         
         }

plt.rcParams.update(params)



class DirectProblem:
    
    
    def __init__(self, telescope:Telescope, vox_matrix:np.ndarray, res_vox:int=64) -> None:
        self.tel = telescope
        self.vox_matrix = vox_matrix
        self.tx, self.ty, self.tz = self.tel.coordinates
        self.taz, self.tze = self.tel.azimuth*np.pi/180, self.tel.zenith*np.pi/180
        self.res_vox = res_vox
        

    def __call__(self, file:Union[str, Path], raypath:dict):
        self.tel.compute_angle_matrix()
        self.voxray = {}
        if isinstance(file, str): file = Path(file)
        file.parents[0].mkdir(parents=True, exist_ok=True)
        pkl_file = str(file)+'.pkl'
        if not os.path.exists(pkl_file): 
            for key, conf in self.tel.configurations.items():
                t0 = time.time()
                panels = conf.panels
                front_panel, rear_panel = panels[0], panels[-1]
                self.tl = abs(front_panel.position.z  - rear_panel.position.z) * 1e-3 #mm->m
                self.tazM, self.tzeM = self.tel.azimuth_matrix[key], self.tel.zenith_matrix[key]
                print(f"Compute voxel ray matrix for {self.tel.name} ({key})...")
                thick_mat = raypath[key]['thickness']
                self.voxray[key] = self.build(thickness=thick_mat)
                print(f"DirectProblem.build({self.tel.name}[{key}]) end --- {time.time() - t0:.3f} s")
            with open(pkl_file, 'wb') as f : 
               pickle.dump(self.voxray, f, pickle.HIGHEST_PROTOCOL)
        else : 
            print(f"Load {file}.pkl ") #file.relative_to(Path.cwd())
            with open(pkl_file, 'rb') as f : 
                self.voxray = pickle.load(f)
               
    def computexyzIntersections(self, xb:np.ndarray, yb:np.ndarray, zb:np.ndarray):
        '''
        Returns the ...

            Parameters:
                
            Returns:
                    res (?): ...
        '''
        xs, ys, zs = self.tx, self.ty, self.tz # UTM (m)
        L = self.tl
        azt, zet = self.taz, self.tze  #rad
        thetas = np.arctan2((xb-xs),(yb-ys)) # this is the azimut angle computed at each node
        elevs = np.arctan((zb-zs)/np.sqrt((xb-xs)**2+(yb-ys)**2)) # this is the elevation angle (rad) computed at each node
        distancesFromTelescope = np.sqrt((xb-xs)**2+(yb-ys)**2+(zb-zs)**2) # distance from telescope (in meters)

        ux = np.cos(elevs)*np.sin(thetas) # unitary vector of this node
        uy = np.cos(elevs)*np.cos(thetas) # unitary vector of this node
        uz = np.sin(elevs) # unitary vector of this node

        uxn = np.sin(zet)*np.sin(azt)*L # vector normal to the telescope matrices with norm L
        uyn = np.sin(zet)*np.cos(azt)*L # vector normal to the telescope matrices with norm L
        uzn = np.cos(zet)*L # vector normal to the telescope matrices with norm L

        tIntersection = (uxn**2 + uyn**2 + uzn**2)/(ux*uxn + uy*uyn + uz*uzn)
        txIntersection = ux*tIntersection 
        tyIntersection = uy*tIntersection
        tzIntersection = uz*tIntersection
        txyzIntersection = np.vstack((txIntersection, tyIntersection,  tzIntersection))
        RotAZ = np.array([[np.cos(azt), -np.sin(azt), 0], [np.sin(azt),  np.cos(azt), 0], [0, 0, 1]])
        RotZA = np.array([[1, 0, 0], [0, np.cos(zet-np.pi/2), -np.sin(zet-np.pi/2)], [0, np.sin(zet-np.pi/2), np.cos(zet-np.pi/2)]])
        
        txyzIntersection = RotZA@RotAZ@txyzIntersection
        txIntersection = txyzIntersection[0,:]
        tyIntersection = txyzIntersection[1,:]
        tzIntersection = txyzIntersection[2,:]
        
        return txIntersection, tyIntersection, tzIntersection, tIntersection, distancesFromTelescope


    def build(self, thickness:np.ndarray) -> np.ndarray:
        '''
        Adapted from Marina Rosas-Carbajal's MatLab function
        Compute voxel apparent thickness along each telescope ray path.
            Parameters:
                thickness (np.ndarray): telescope apparent thickness matrix (nray_x, nray_y)
            Returns:
                voxrayMatrix (np.ndarray): matrix shape (nrays, nvox) 
        '''
        res=self.res_vox
        xb, yb, zb = self.vox_matrix[:, 25], self.vox_matrix[:, 26], self.vox_matrix[:, 27] #xyz voxel barycenters
        txIntersection, tyIntersection, tzIntersection, tIntersection, distancesFromTelescope = self.computexyzIntersections(xb, yb, zb)
        print("Compute voxels apparent thickness along each telescope ray path...\n")
        apparentThicknessMatrix = thickness
        nvox = self.vox_matrix.shape[0]
        ndx, ndy = self.tazM.shape
        voxrayMatrix = np.zeros(shape=(ndx*ndy,nvox))*np.nan
        MNorm = np.zeros(shape=(ndx, ndy))
        nbarsx, nbarsy = self.tel.panels[0].matrix.nbars_x, self.tel.panels[0].matrix.nbars_y
        wx = self.tel.panels[0].matrix.scintillator.width #in mm width scintillators
        wy = wx
        for ii in tqdm(range(ndx), desc="DirectProblem.build()"): # loop on DX    
            DX = (ii+1)-nbarsx    
            # print(f'\t\t\t line {ii+1} / {ndx} \n')
            for jj in range(ndy):  # loop on DY        
                DY = (jj+1)-nbarsy
                apparentThicknessRay = apparentThicknessMatrix[ii,jj]
                za = self.tzeM[ii,jj]*180/np.pi
                if (apparentThicknessRay > 0) & (~np.isnan(apparentThicknessRay)) & (np.isfinite(apparentThicknessRay)) & (apparentThicknessRay < 1000) & (za <= 90):
                    #print(self.tzeM[ii,:]*180/np.pi)
                    # print('\t\t\t\t this axis is valid \n')
                    tomographyKernel = np.zeros(shape=(nvox,))
                    # edit this axis acceptance pattern with the geometrical acceptance (Sullivan 1970, Thomas 1971)
                    dx = lambda x: (x-DX)*wx*1e-1
                    dy = lambda y: (y-DY)*wy*1e-1
                    L_cm = self.tl*1e2
                    l = lambda x,y: np.sqrt(L_cm**2 + dx(x)**2 + dy(y)**2)
                    phi = lambda x,y: np.arctan(np.sqrt(dx(x)**2 + dy(y)**2)/L_cm)
                    acceptancePattern = lambda x,y: np.cos(phi(x,y))/(l(x,y)**2)
                    contributionPattern = lambda x,y: 1-(np.abs(x)+np.abs(y))+np.abs(x*y)
                    acceptancePattern2 = lambda x,y: acceptancePattern(x,y)*contributionPattern(x,y)
                    normFactor = dblquad(acceptancePattern,-1,1,-1,1)[0]
                    #normFactor = nquad(acceptancePattern2, [(-1,1), (-1,1)])[0]
                    acceptancePattern3 = lambda x,y: acceptancePattern2(x,y)/normFactor
                    MNorm[ii,jj] = normFactor
                    '''
                    delta_x, delta_y = dxdy
                    Nxy = self.Nxy
                    L = self.length*1e-1 # mm -> cm
                    w = self.width*1e-1 # mm -> cm
                    ####Number of pixel couples for given (delta_x, delta_y) direction
                    couple_dxdy = Nxy**2 - Nxy * ( np.abs(delta_x) + np.abs(delta_y) )  + np.abs( delta_x * delta_y ) #ok
                    alpha = np.arctan(w * np.sqrt((delta_x)**2+(delta_y)**2) /(2*L)) #ok ####Erreur in Kevin's thesis???? np.arctan(w * np.sqrt((delta_x)**2+(delta_y)**2) / 2*L)
                    t_dxdy = w**4 * np.cos(alpha) / (4*L**2 + w**2 * ((delta_x)**2+(delta_y)**2) ) #ok 
                    acc_dxdy = couple_dxdy * t_dxdy
                    '''
                    sv_inThisRay = np.where(((abs(txIntersection/(wy*1e-3)+DY) <= 4) & (abs(tzIntersection/(wx*1e-3)+DX) <= 4) & (tIntersection > 0)) | (distancesFromTelescope < 100) ) [0]
                    iteration = 5 # hypersample the cube (2**iteration)
                    NsubCubes = (2**iteration)**3
                    for kk in range(len(sv_inThisRay)):
                        x_sub_bary = np.linspace(-res/2,res/2,2**(iteration+1) + 1) 
                        x_sub_bary = x_sub_bary[1:-1:2]
                        y_sub_bary = np.linspace(-res/2,res/2,2**(iteration+1) + 1) 
                        y_sub_bary = y_sub_bary[1:-1:2]
                        z_sub_bary = np.linspace(-res/2,res/2,2**(iteration+1) + 1) 
                        z_sub_bary = z_sub_bary[1:-1:2]
                        X, Y, Z = np.meshgrid(x_sub_bary,y_sub_bary,z_sub_bary)
                        sub_bary_coor = np.vstack((X.flatten(), Y.flatten(), Z.flatten())).T
                        x_sub_bary_coor = sub_bary_coor[:, 0] + xb[sv_inThisRay[kk]]
                        y_sub_bary_coor = sub_bary_coor[:, 1] + yb[sv_inThisRay[kk]]
                        z_sub_bary_coor = sub_bary_coor[:, 2] + zb[sv_inThisRay[kk]]
                        sub_txIntersection, sub_tyIntersection, sub_tzIntersection, sub_tIntersection, sub_distancesFromTelescope = self.computexyzIntersections(x_sub_bary_coor,y_sub_bary_coor,z_sub_bary_coor)
                        sv_valid_inThisCube = (abs(sub_txIntersection/0.05+DY) <= 1) & (abs(sub_tzIntersection/0.05+DX) <= 1) & (sub_tIntersection > 0) & (sub_distancesFromTelescope > 2)
                        if np.any(sv_valid_inThisCube==True):
                            cubeKernel = acceptancePattern3(sub_txIntersection[sv_valid_inThisCube]/0.05+DY, sub_tzIntersection[sv_valid_inThisCube]/0.05+DX)
                            cubeKernel = cubeKernel/sub_distancesFromTelescope[sv_valid_inThisCube]**2
                            thisCubeContribution = np.sum(cubeKernel)/NsubCubes
                            thisCubeContribution = (thisCubeContribution*res**3)
                        else :
                            thisCubeContribution = 0
                        tomographyKernel[sv_inThisRay[kk]] = thisCubeContribution
                    voxrayMatrix[ii*ndx + jj,:] = tomographyKernel/(normFactor*25)
        return voxrayMatrix    



class Voxel:
    """
    A class to voxelize a volume from terrain model.

    Attributes
    ----------
    surface_grid : np.ndarray with shape (m, n, 3) [meter]

    surface_center : np.ndarray with shape (2,) [meter]

    res_vox : int [meter]
    
    Methods
    -------
    generateTopography()
    
    plotTopography()

    generateMesh()

    getVoxels()

    extractVoxels()

    getVoxelDistances()

    getVolumeTotal()

    plot3Dmesh()
    ...

    """

    def __init__(self, surface_grid:np.ndarray, surface_center:np.ndarray, res_vox:int=64):
        
        self.SX, self.SY, self.SZ = surface_grid.T
        self.sc = surface_center
        self.res_vox = res_vox
        self.xmin, self.xmax = np.min(self.SX[:,0]), np.max(self.SX[:,0])
        self.ymin, self.ymax = np.min(self.SY[0,:]), np.max(self.SY[0,:])
        self.zmin, self.zmax = np.min(self.SZ), np.max(self.SZ)

        self.SX, self.SY, self.SZ = self.generateTopography(res_vox) 
        self.vox_matrix = None
        self.vox_xyz = None
        self.vox_distances = None
        self.vox_volumes = None


    def generateTopography(self, res:int=64, ) -> np.ndarray:
        '''
            Parameters:
                    res (int): topography resolution in meter
            Returns:
                    X, Y, Z (np.ndarray): coordinate grid matrices
        ''' 

        X, Y = np.meshgrid(np.arange(self.xmin, self.xmax, res), np.arange(self.ymin,self.ymax, res))                                          
        # X, Y = X + self.sc[0], Y + self.sc[1]
        points, values = np.array([self.SX.flatten(), self.SY.flatten()]).T, self.SZ.flatten()
        Z = griddata(points, values, (X,Y), method='linear' )
        return X, Y, Z


    def plotTopography(self, ax:Axes, mask:np.ndarray=None, res:int=64,  **kwargs) -> None:
        '''
        Returns the ...

            Parameters:
                    par1 (?): ...
                    par2 (?): ...

            Returns:
                    res (?): ...
        '''
       
        if mask is not None: self.SZ[mask] = np.nan
        ax.plot_surface(self.SX,self.SY,self.SZ,**kwargs)


    def generateMesh(self) -> None:
        '''
        Adapted from Marina Rosas-Carbajal's MatLab function
        '''
        
        res = self.res_vox
        altitudeMin = self.zmin
        altitudeMax = self.zmax
        Nx,Ny = self.SX.shape
        Nz = int((altitudeMax-altitudeMin)//res) 
        altitudeVector = np.arange(altitudeMin, altitudeMax+res, res)
        nvoxels=int((Nx-1)*(Ny-1)*(Nz-1))
        vox_matrix = np.zeros(shape=(nvoxels,  1 + 6*4 + 3 + 3))
        status = 0
        points, values = (self.SX.ravel(),self.SY.ravel()), self.SZ.ravel()
        topo_interp = LinearNDInterpolator(points, values)  
        volumes = np.zeros((nvoxels,2))
        for ii in range(Nx-1):
            
            for jj in range(Ny-1):
               
                rep = 0
                z_surface = np.array([self.SZ[ii,jj], self.SZ[ii,jj+1], self.SZ[ii+1,jj+1], self.SZ[ii+1,jj]])
                z_surface_min = np.min(z_surface)
                z_surface_max = np.max(z_surface)
                
                for kk in range(Nz-1):
                    if rep == 1 :# once we satisfy this condition we don't need to continue inside this loop
                        break
                    x_down = np.array([self.SX[ii,jj], self.SX[ii,jj+1], self.SX[ii+1,jj+1], self.SX[ii+1,jj]])
                    x_up = np.array([self.SX[ii,jj], self.SX[ii,jj+1], self.SX[ii+1,jj+1], self.SX[ii+1,jj]])
                    y_down = np.array([self.SY[ii,jj], self.SY[ii,jj+1], self.SY[ii+1,jj+1], self.SY[ii+1,jj]])
                    y_up = np.array([self.SY[ii,jj], self.SY[ii,jj+1], self.SY[ii+1,jj+1], self.SY[ii+1,jj]])
                    z_d, z_u = altitudeVector[kk] , altitudeVector[kk+1]
                    z_down = np.ones(4)*z_d
                    z_up = np.ones(4)*z_u
                    v = 0 
                    # 1 : interface inf, 2 : interface sup, 3 : in between (weird), 4 : below interface, 5 : above interface
                    if (z_surface_min <= z_u) and (z_d < z_surface_min):
                        z_up = z_surface  # topography's coordinates
                        status = 1 # below surface
                        rep = 1  
                    elif (z_u < z_surface_max) and (z_d > z_surface_min) : # cube in the middle, not taken into account
                        status = 2 # in the middle
                    elif (z_u > z_surface_max) and (z_d < z_surface_max) : # cube in the middle, not taken into account
                        z_down = z_surface
                        status = 3 # in the middle
                    elif (z_u < z_surface_min) : 
                        status = 4 # below surface
                    elif (z_d > z_surface_max) : # cube above surface, not taken into account
                        status = 5 # above surface
                    
                    xyz_vox = np.vstack((np.vstack((x_down,y_down,z_down)).T, np.vstack((x_up,y_up,z_up)).T ))
                    self.barycenters = np.mean(xyz_vox, axis=0)
                    voxel =  np.concatenate(([status], x_down, x_up, y_down, y_up, z_down, z_up, self.barycenters,[ii, jj, kk]), axis=0)
                    ix_vox = ii*(Ny-1)*(Nz-1) + jj*(Nz-1) + kk
                    vox_matrix[ix_vox,:] = voxel
                    volume_estimate = np.zeros(2)
                    if status == 1:
                        volume_estimate = self.getVolumeVoxel(voxel, topo_interp) #(val,err)
                    elif status == 4 : 
                        volume_estimate[0] = self.res_vox**3
                    volumes[ix_vox] = volume_estimate
        
        status = vox_matrix[:,0]
        sv = (status == 1) | (status == 4)
        self.vox_matrix = vox_matrix[sv,:]
        self.vox_volumes = volumes[sv,:]

    
    def getVoxels(self) -> None:
        '''
        '''
        x, y, z = self.vox_matrix[:,1:5], self.vox_matrix[:,9:13], self.vox_matrix[:,17:25]
        #### https://stackoverflow.com/questions/42611342/representing-voxels-with-matplotlib
        self.vox_xyz = np.array( [
            [  [x[:,0], y[:,3], z[:,0]], [x[:,0], y[:,0], z[:,0]],[x[:,1], y[:,0], z[:,0]], [x[:,1], y[:,3], z[:,0]] ],
            [  [x[:,0], y[:,0], z[:,0]], [x[:,0], y[:,0], z[:,4]],[x[:,1], y[:,0], z[:,5]], [x[:,1], y[:,0], z[:,0]] ],
            [  [x[:,1], y[:,0], z[:,5]], [x[:,1], y[:,0], z[:,0]],[x[:,1], y[:,3], z[:,0]], [x[:,1], y[:,3], z[:,6]] ],
            [  [x[:,0], y[:,0], z[:,4]], [x[:,0], y[:,0], z[:,0]],[x[:,0], y[:,3], z[:,0]], [x[:,0], y[:,3], z[:,7]] ],
            [  [x[:,0], y[:,3], z[:,0]], [x[:,0], y[:,3], z[:,7]],[x[:,1], y[:,3], z[:,6]], [x[:,1], y[:,3], z[:,0]] ],
            [  [x[:,0], y[:,3], z[:,7]], [x[:,0], y[:,0], z[:,4]],[x[:,1], y[:,0], z[:,5]], [x[:,1], y[:,3], z[:,6]] ]
        ] ) 
        self.vox_xyz = np.swapaxes(self.vox_xyz.T,1,-1)


    def extractVoxels(self, vox_matrix: np.ndarray) -> None:
        '''
        Parameters: 
            vox_matrix (np.ndarray) : voxel summit coordinate matrix shape (n, 31)
        Returns: 
            None
        '''
        x, y, z = vox_matrix[:,1:5], vox_matrix[:,9:13], vox_matrix[:,17:25]
        #### https://stackoverflow.com/questions/42611342/representing-voxels-with-matplotlib
        voxels = np.array( [
            [  [x[:,0], y[:,3], z[:,0]], [x[:,0], y[:,0], z[:,0]],[x[:,1], y[:,0], z[:,0]], [x[:,1], y[:,3], z[:,0]] ],
            [  [x[:,0], y[:,0], z[:,0]], [x[:,0], y[:,0], z[:,4]],[x[:,1], y[:,0], z[:,5]], [x[:,1], y[:,0], z[:,0]] ],
            [  [x[:,1], y[:,0], z[:,5]], [x[:,1], y[:,0], z[:,0]],[x[:,1], y[:,3], z[:,0]], [x[:,1], y[:,3], z[:,6]] ],
            [  [x[:,0], y[:,0], z[:,4]], [x[:,0], y[:,0], z[:,0]],[x[:,0], y[:,3], z[:,0]], [x[:,0], y[:,3], z[:,7]] ],
            [  [x[:,0], y[:,3], z[:,0]], [x[:,0], y[:,3], z[:,7]],[x[:,1], y[:,3], z[:,6]], [x[:,1], y[:,3], z[:,0]] ],
            [  [x[:,0], y[:,3], z[:,7]], [x[:,0], y[:,0], z[:,4]],[x[:,1], y[:,0], z[:,5]], [x[:,1], y[:,3], z[:,6]] ]
        ] ) 
        voxels = np.swapaxes(voxels.T,1,-1)
        return voxels

    def getVolumeVoxel(self, voxel, topo):
        x_min, x_max = np.min(voxel[1:4]), np.max(voxel[1:4]) 
        y_min, y_max = np.min(voxel[9:12]), np.max(voxel[9:12])
        z_min = voxel[17]
        f_zc = lambda xc,yc :  (topo((xc, yc)) - z_min)
        vol, err = dblquad(f_zc, y_min, y_max, x_min, x_max,)
        return np.asarray([vol, err])


    def getVoxelDistances(self)-> None:
        '''
        Matrix featuring distance between each voxel couple (i,j): (d)_i,j
        '''
        nvox = self.barycenters.shape[0]
        self.vox_distances = np.zeros(shape=(nvox,nvox))
        for j in range(nvox):
            for i in range(nvox):
                self.vox_distances[i,j] = np.linalg.norm(self.barycenters[i]-self.barycenters[j]) 

    def getVolumeTotal(self, sv_vox:np.ndarray) -> float:
        '''
        '''
        M = self.vox_matrix
        sv_surface = np.where(M[sv_vox,0]==1)[0]
        sv_not_surface = np.where(M[sv_vox,0]==4)[0]
        vol_tot = self.res_vox**3 * len(sv_not_surface) 
        SX,SY,SZ = self.generateTopography(self.res_vox) 
        points, values = (SX.flatten(),SY.flatten()), SZ.flatten()
        finterp = LinearNDInterpolator(points, values)        
        for s in sv_surface:
            ivox = sv_vox[s]
            x_min, x_max = np.min(M[ivox,1:4]), np.max(M[ivox,1:4]) 
            y_min, y_max = np.min(M[ivox,9:12]), np.max(M[ivox,9:12])
            z_min = M[ivox,17]
            f_zc = lambda xc,yc :  (finterp((xc, yc)) - z_min)
            dv =  dblquad(f_zc,  y_min, y_max, x_min, x_max,)
            vol_tot += dv[0]
        return vol_tot

    def plot3Dmesh(self, ax:Axes3D, vox_xyz:np.ndarray=None, **kwargs) -> None:
        '''
        Parameters : 
            ax (Axes3D) : 
            vox_xyz (np.ndarray) : voxel summit coordinate matrix shape (n, 6, 4, 3)
        Returns: 
            None
        '''
        if vox_xyz is None: 
            vox_xyz = self.vox_xyz
            if self.vox_xyz is None: self.getVoxels()
        pc = Poly3DCollection(np.concatenate(vox_xyz), **kwargs)
        ax.add_collection3d(pc)

    def convertToVTU(self, file:Path|str):
        import vtk
        points = vtk.vtkPoints()
        ugrid = vtk.vtkUnstructuredGrid()

        point_id_map = {}
        current_point_id = 0
        nvoxels, nfaces, ncorners, ncoords = self.vox_xyz.shape
        for v in range(nvoxels):
            # Récupération des 24 sommets
            coords = self.vox_xyz[v].reshape(-1, 3)
            # Sommets uniques
            unique_pts = np.unique(coords, axis=0)
            if unique_pts.shape[0] != 8:
                raise ValueError("Voxel non hexaédrique")
            # Tri spatial pour imposer l’ordre VTK
            center = unique_pts.mean(axis=0)
            rel = unique_pts - center
            # z puis y puis x → ordre stable
            order = np.lexsort((rel[:,0], rel[:,1], rel[:,2]))
            ordered_pts = unique_pts[order]
            hex_cell = vtk.vtkHexahedron()
            for i, p in enumerate(ordered_pts):
                key = tuple(p)
                if key not in point_id_map:
                    points.InsertNextPoint(p)
                    point_id_map[key] = current_point_id
                    current_point_id += 1
                hex_cell.GetPointIds().SetId(i, point_id_map[key])
            ugrid.InsertNextCell(hex_cell.GetCellType(),
                                hex_cell.GetPointIds())

        ugrid.SetPoints(points)
        if self.vox_volumes is not None: 
            vol_arr = numpy_support.numpy_to_vtk(
                np.array(self.vox_volumes),
                deep=True,
                array_type=vtk.VTK_DOUBLE
            )
            vol_arr.SetName("voxel_volume")
            ugrid.GetCellData().AddArray(vol_arr)
            ugrid.GetCellData().SetActiveScalars("voxel_volume")
        writer = vtk.vtkXMLUnstructuredGridWriter()
        writer.SetFileName(file)
        print(f'Save {file}')
        writer.SetInputData(ugrid)
        writer.Write()


if __name__ == "__main__":
    
    t0 = time.time()
    print("Start: ", time.strftime("%H:%M:%S", time.localtime()))#start time

    args = get_common_args()
    cmn = Common(args)

    survey = cmn.survey
    tel = cmn.telescope

    res_vox = 64 #m
    c = np.array([642960, 1774280])
    voxel = Voxel(    surface_grid=survey.surface_grid,
                      surface_center=c,#survey.surface_center, 
                      res_vox=res_vox)

    conf='3p1'
    front, rear = tel.configurations[conf].panels[0], tel.configurations[conf].panels[-1]
    
    dem_path = survey.dem.parent
    survey_path = dem_path.parent
    tel_files_path = survey_path / 'telescope'  / tel.name

    dout_vox_struct = survey_path / "voxel"
    dout_vox_struct.mkdir(parents=True, exist_ok=True)
    fout_vox_struct = dout_vox_struct / f"vox_matrix_res{res_vox}m.npy"
    # if fout_vox_struct.exists(): 
    #     vox_matrix = np.load(fout_vox_struct)
    #     voxel.vox_matrix = vox_matrix
    # else : 
    voxel.generateMesh()
    print(f"generateMesh() --- {time.time() - t0:.3f} s")
    voxel.getVoxels()
    np.save(fout_vox_struct, voxel.vox_xyz)
    print(f"Save {fout_vox_struct}")
    output_file = dout_vox_struct / f"vox_matrix_res{res_vox}m.vtu"
    voxel.convertToVTU(output_file)
    # output_file = dout_vox_struct / f"volumes_res{res_vox}m.npy"
    # np.save(output_file, voxel.vox_volumes)
    # print(f"Save {output_file}")






    # raypath = RayPath(telescope=tel,
    #                     surface_grid=survey.surface_grid,
    #                     )
    # dout_ray = tel_files_path / 'raypath'/ f'az{tel.azimuth:.1f}_elev{tel.elevation:.1f}' 
    # raypath(file=dout_ray / 'raypath', max_range=1500)
    # thickness = raypath.raypath[conf]['thickness']
    # fout_cmd = dout_ray / "voxel" / f"voxray_{conf}_res{res_vox}m.npy"
    # fout_cmd.parents[0].mkdir(parents=True, exist_ok=True)
    # dirpb = DirectProblem(telescope=tel, 
    #                       vox_matrix=voxel.vox_matrix, 
    #                       res_vox=res_vox)

    # fout_voxray = dout_ray / "voxel" / f"voxray_res{res_vox}m"
    # fout_voxray.parent.mkdir(parents=True, exist_ok=True)
    # dirpb(file=fout_voxray, raypath=raypath.raypath)
    # voxrayMatrix = dirpb.voxray[conf]
    # print(f"voxrayMatrix.shape = {voxrayMatrix.shape}")
    


    ###PLOTS

    import palettable
    import matplotlib.colors as cm
    cmap_rho = palettable.scientific.sequential.Batlow_20.mpl_colormap 
    rho_min, rho_max, n = .8, 2.7, 100
    range_val = np.linspace(rho_min, rho_max, n)
    norm_r = cm.Normalize(vmin=rho_min, vmax=rho_max)
    color_scale_rho =  cmap_rho(norm_r(range_val))
    color_vox = np.array([[0.98039216, 0.8,  0.98039216, 1.        ]])
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111, projection='3d')
    kwargs_mesh = dict(facecolor = color_vox, edgecolor = "grey", alpha = 0.3)
    voxel.plot3Dmesh(ax=ax, vox_xyz=voxel.vox_xyz, **kwargs_mesh)
    kwargs_topo = dict (color = 'lightgrey', edgecolor = 'grey', alpha = 0.2)
    voxel.plotTopography(ax, **kwargs_topo)
    dx=1000
    xrange = [survey.surface_center[0]-dx, survey.surface_center[0]+dx]
    yrange = [survey.surface_center[1]-dx, survey.surface_center[1]+dx]
    zrange = [1.0494e+03, 1.4658e+03 + 50]
    ax.set_xlim(xrange)
    ax.set_ylim(yrange)
    ax.set_zlim(zrange)
    ax.set_aspect('auto') #'equal'
    ax.grid()
    ax.view_init(30, -60)
    ax.dist = 8    # define perspective (default=10)
    xstart, xend = ax.get_xlim()
    ax.xaxis.set_ticks(np.arange(xstart, xend, 5e2))
    ystart, yend = ax.get_ylim()
    ax.yaxis.set_ticks(np.arange(ystart, yend, 5e2))
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ltel_n = ["SB", "SNJ", "BR", "OM"]
    str_tel =  "_".join(ltel_n)
    ltel_coord = np.array([ str2telescope(name).coordinates for name in ltel_n])
    ltel_color = np.array([ str2telescope(name).color for name in ltel_n])
    ax.scatter(ltel_coord[:,0], ltel_coord[:,1], ltel_coord[:,-1], c=ltel_color, s=30,marker='s',)
    # mask = np.isnan(thickness).flatten()
    # tel.plot_ray_paths(ax=ax, front_panel=front, rear_panel=rear, mask=mask, rmax=1500,  color='grey', linewidth=0.3 )#
    plt.show()

    ###Compute volume
    # voxrayMatrix[np.isnan(voxrayMatrix)] = 0
    # mvox = np.any(voxrayMatrix>0, axis=0)
    # sv_vox = np.where(mvox==True)[0]
    # vol = voxel.getVolumeTotal(sv_vox=sv_vox)
    # print(f"Volume covered by {tel.name} ({conf}) : {vol:.5e} m^3")

    print(f"End --- {time.time() - t0:.3f} s")