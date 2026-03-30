#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from dataclasses import dataclass, field
from enum import Enum, auto
import matplotlib.axes 
import numba
import numpy as np 
import json
from typing import List

#package module(s)
from config import STRUCT_DIR
from utils import tools


@dataclass(frozen=True)
class Scintillator:
    type: str 
    #dimensions in mm
    length: float 
    width: float
    thickness: float
    def __str__(self): return f"{self.type}"


@dataclass
class ChannelMap:
    file: str
    
    def map(self):
        """
        Read the .dat mapping file and fill a dictionary 'dict_ch_to_bar' :
        { keys = channel No (in [0;63]) : values = X or Y coordinate (e.g 'X01') }
        """
        ####convert ch to chmap
        with open(self.file, 'r') as fmap:
           self.dict_ch_to_bar = json.loads(fmap.read())
        self.dict_ch_to_bar = {int(k):v for k,v in self.dict_ch_to_bar.items()}
        self.dict_bar_to_ch = { v: k for k,v in self.dict_ch_to_bar.items()  }
        self.channels = list(self.dict_ch_to_bar.keys())
        self.bars = list(self.dict_ch_to_bar.values())
                
    def __post_init__(self):
        self.map()

    def __str__(self):
        return f'ChannelMap: {self.dict_ch_to_bar}'
    

@dataclass(frozen=True)
class Matrix:
    version : int
    scintillator : Scintillator
    nbars_x : int
    nbars_y : int  
    wls_type : str
    fiber_out : str
    def __str__(self):
        return f"v{self.version} with ({self.nbars_x}, {self.nbars_y}) {self.scintillator} scintillators"


class PositionEnum(Enum):
    Front = auto()
    Middle1 = auto()
    Middle2 = auto()
    Rear = auto()


class Position:
    def __init__(self, loc:PositionEnum, index:int, z:float):
        self.loc = loc.name 
        self.index = index
        self.z = z  #in mm


@dataclass(frozen=True)
class Panel:
    matrix : Matrix 
    id : int
    channelmap : ChannelMap 
    position: Position #Tuple[PositionEnum, float] 
    def __str__(self,):
        return self.position.loc


@dataclass(frozen=True)
class PMT:
    id : int
    panel : List[Panel] 
    channelmap : ChannelMap
    type : str = field(default='MAPMT')


@dataclass
class PanelConfig:
   
    name : str
    panels : List[Panel]
    pmts : List[PMT] = field(default_factory=list)


    def __post_init__(self):
        front, rear =  self.panels[0], self.panels[-1]
        # s_xy = front.matrix.scintillator.length
        s_xy = front.matrix.scintillator.length #- front.matrix.scintillator.width #- front.matrix.scintillator.width/2
        z_front, z_rear =  front.position.z, rear.position.z
        length_z = abs(z_front - z_rear)
        
        # if len(self.panels) == 3:
            # z_middle = self.panels[1].position.z
            # if constraint on middle panel:
            # dz = abs(z_front-z_middle)
            # alpha = (dz/ length_z)
            # _ext = self.get_angular_extremum(s_xy, length_z, alpha=alpha)
        
        # Muography angular coordinates XY :  u = dx/dz = tan_theta_x ; v = dy/dz = tan_theta_y 
        uv_ext = s_xy / (length_z) 
        range_uv = np.array([[-uv_ext, uv_ext], [-uv_ext, uv_ext]])
        object.__setattr__(self, 'length_z',  length_z )
        object.__setattr__(self, 'range_uv',  range_uv )
        nx, ny = front.matrix.nbars_x, front.matrix.nbars_y
        nu, nv = 2*nx-1, 2*ny-1 # number of angular bins in u and v directions (binning centered on 0)
        object.__setattr__(self, 'npixels', nu*nv )
        object.__setattr__(self, 'shape_uv',(nu, nv))
        u_min, u_max = range_uv[0][0], range_uv[0][1]
        v_min, v_max = range_uv[1][0], range_uv[1][1]
        u_edges = np.linspace(u_min, u_max, nu+1)
        v_edges = np.linspace(v_min, v_max, nv+1)
        object.__setattr__(self, 'u_edges', u_edges)
        object.__setattr__(self, 'v_edges', v_edges)

    def get_angular_extremum(self, size_xy, delta_z, alpha=1/2):
        return size_xy/delta_z * min(alpha, 1-alpha)
    
    def __str__(self):
        matrices = [p.matrix for p in self.panels]
        versions = [m.version for m in matrices]
        v = {f'v{v}': versions.count(v) for v in  set(versions)}
        sout =f"Config {self.name}: {len(self.panels)} panels " + "-".join([p.position.loc for p in self.panels]) + "\n"
        return sout

@dataclass
class Telescope:
    name : str
    coordinates : np.ndarray = field(default_factory=lambda: np.ndarray(shape=(3,))) #coordinates (easting, northing, altitude)
    azimuth : float = field(default_factory=float) #deg
    zenith : float = field(default_factory=float) #deg
    elevation : float = field(default_factory=float) #deg 
    color : str = field(default_factory=lambda: "")
    site : str = field(default_factory=lambda: "")
    survey : str = field(default_factory=lambda: "")
    min_plan :  int = field(default_factory=lambda: int)
    max_plan :  int = field(default_factory=lambda: int)

    def __post_init__(self, ): 
        self.configurations = {}
        self.panels = List[Panel]
        self.pmts = List[PMT]
        self.rays = None
      
    def __setitem__(self, name:str, configuration:PanelConfig): 
        self.configurations[name] = configuration

    def __getitem__(self, name:str): 
        config = self.configurations[name]
        return config

    def __str__(self):
        sout = f"Telescope: {self.name}\n "
        if self.site : sout += f"- Site: {self.site}\n "
        if np.all(self.coordinates != None) :  sout += "- UTM (easting, northing, altitude): ("+ ', '.join([f'{i:.0f}' for i  in self.coordinates]) + ") m\n "
        if self.azimuth and self.elevation: sout += f"- Orientation (azimuth,elevation): ({self.azimuth}, {self.elevation}) deg\n "
        sout += "- Configurations:\n" 
        for _, conf in self.configurations.items(): sout += "\t" + conf.__str__() 
        return sout
    

    def get_ray_matrix(self, front_panel:Panel, rear_panel:Panel):
        """
        Ray paths referenced as (DX,DY) couples
        """
        nbars_xf, nbars_yf  = front_panel.matrix.nbars_x,front_panel.matrix.nbars_y
        nbars_xr, nbars_yr  = rear_panel.matrix.nbars_x,rear_panel.matrix.nbars_y
        barNoXf, barNoYf = np.arange(1, nbars_xf+1),np.arange(1, nbars_yf+1)
        barNoXr, barNoYr = np.arange(1, nbars_xr+1),np.arange(1, nbars_yr+1)
        DX_min, DX_max = np.min(barNoXf) - np.max(barNoXr) ,  np.max(barNoXf) - np.min(barNoXr) 
        DY_min, DY_max = np.min(barNoYf) - np.max(barNoYr) ,  np.max(barNoYf) - np.min(barNoYr) 
        mat_rays = np.mgrid[DX_min:DX_max+1:1, DY_min:DY_max+1:1].reshape(2,-1).T.reshape(2*nbars_xf-1,2*nbars_yf-1,2) 
        return mat_rays


    def get_ray_paths(self, front_panel:Panel, rear_panel:Panel, rmax:float=600,): 
        """
        Compute telescope ray paths (or line of sights)
        """
        front = front_panel
        rear = rear_panel
        L = (rear.position.z - front.position.z)  * 1e-3
        w = front.matrix.scintillator.length * 1e-3
        nx, ny = front.matrix.nbars_x, front.matrix.nbars_y
        step = w/nx
        #mat_rays = self._get_ray_matrix(front_panel=front, rear_panel=rear)

        barNoXf, barNoYf = np.arange(1, nx+1),np.arange(1, ny+1)
        barNoXr, barNoYr = np.arange(1, nx+1),np.arange(1, ny+1)
        DX_min, DX_max = np.min(barNoXf) - np.max(barNoXr) ,  np.max(barNoXf) - np.min(barNoXr) 
        DY_min, DY_max = np.min(barNoYf) - np.max(barNoYr) ,  np.max(barNoYf) - np.min(barNoYr) 
        arrDX, arrDY = np.arange(DX_min, DX_max+1), np.arange(DY_min, DY_max+1)

        nrays = (2*nx-1) * (2*ny-1)
        mat_pixel = np.zeros(shape=(nrays,3))

        delta_az, delta_incl = 0,0
        azimuth =  self.azimuth + delta_az
        elevation = self.elevation + delta_incl
        alfa, beta = (360 - azimuth)*np.pi/180, (90 + elevation)*np.pi/180 
        gamma = 0
        Rinv_alfa = np.array([[np.cos(alfa),-np.sin(alfa),0 ], [np.sin(alfa),np.cos(alfa),0], [0,0,1]]) 
        Rinv_beta = np.array([[1,0,0], [0,np.cos(beta),-np.sin(beta)], [0,np.sin(beta),np.cos(beta)]]) 
        Rinv_gamma = np.array([[np.cos(gamma),-np.sin(gamma),0], [np.sin(gamma),np.cos(gamma),0], [0,0,1]])
        Rinv = np.matmul(Rinv_alfa, np.matmul(Rinv_beta, Rinv_gamma))
        self.rays = np.zeros(shape=(nrays,2,3))
        k=0
        for dy in arrDY:
            ycoord = dy*step
            for dx in arrDX:
                xcoord = dx*step
                mat_pixel[k,:] = np.matmul(Rinv,np.array([xcoord, ycoord, -L]).T)  #rotation
                self.rays[k,:] = np.array([self.coordinates[:], self.coordinates[:] + mat_pixel[k,:] * rmax])
                k += 1
        self.rays = np.flipud(self.rays)

    def plot_ray_paths(self, ax:matplotlib.axes.Axes, front_panel:Panel, rear_panel:Panel, mask:np.ndarray=None, rmax:float=600, **kwargs):
        """
        Plot telescope ray paths (or line of sights) on given 'axis' (matplotlib.axes.Axes)
        INPUTS: 
        - color_map (np.ndarray) : RGBA array (n, 4)  
        - mask (np.ndarray) : bool array (n,)  
        """
        if self.rays is None : self.get_ray_paths(front_panel=front_panel, rear_panel=rear_panel, rmax=rmax)
        if mask is not None: self.rays[mask,:] = [np.ones(3)*np.nan, np.ones(3)*np.nan]
        for k in range(self.rays.shape[0]):      
            ax.plot(self.rays[k,:, 0], self.rays[k,:, 1], self.rays[k,:, 2], **kwargs)
        
    def plot_ray_values(self, ax:matplotlib.axes.Axes, color_values:np.ndarray, front_panel:Panel, rear_panel:Panel, mask:np.ndarray=None, rmax:float=600, **kwargs):
        """
        Plot array of values associated to n ray paths at the end of paths (from the telescope position)
        INPUTS: 
        - color_map (np.ndarray) : RGBA array (n, 4)  
        - mask (np.ndarray) : bool array (n,)  
        """
        if self.rays is None : self.get_ray_paths(front_panel=front_panel, rear_panel=rear_panel, rmax=rmax)
        if mask is not None: self.rays[mask,:] = [np.ones(3)*np.nan, np.ones(3)*np.nan]
        for k in range(self.rays.shape[0]):       
            ax.scatter(self.rays[k,-1, 0], self.rays[k,-1, 1], self.rays[k,-1, 2], c=color_values[k], **kwargs)    

    def compute_angular_coordinates(self):
        """
        Calcule, pour chaque configuration (3p / 4p),
        les matrices (phi, theta) associées au binning angulaire local
        (dx/dz, dy/dz), en tenant compte de l'orientation du télescope.
        """
        self.directions_matrix = {}
        self.azimuth_matrix = {}
        self.zenith_matrix = {}
        self.zenith_os_matrix = {}

        az = np.pi/2 - np.deg2rad(self.azimuth)
        ze = np.pi/2 - np.deg2rad(self.elevation)
        # Rotation autour de Z (azimuth)
        Rz = np.array([
            [ np.cos(az), -np.sin(az), 0.0],
            [ np.sin(az),  np.cos(az), 0.0],
            [ 0.0       ,  0.0       , 1.0]
        ])
        # Rotation autour de Y (élévation)
        Ry = np.array([
            [ np.cos(ze), 0.0, np.sin(ze)],
            [ 0.0       , 1.0, 0.0       ],
            [-np.sin(ze), 0.0, np.cos(ze)]
        ])
        R = Rz @ Ry  # rotation locale → globale
        for _, conf in self.configurations.items():
            front = conf.panels[0]
            nx = front.matrix.nbars_x
            ny = front.matrix.nbars_y
            w  = front.matrix.scintillator.width
            dz = conf.length_z
            nu, nv = conf.shape_uv
            dir_mat = np.zeros((nu, nv, 3))
            phi_mat   = np.zeros((nu, nv))
            theta_mat = np.zeros((nu, nv))
            theta_os_mat = np.zeros((nu, nv))
            for j, dy in enumerate(range(-ny + 1, ny)):
                for i, dx in enumerate(range(-nx + 1, nx)):
                    # Direction local pixel (x, y, z)
                    v_local = np.array([
                        - dx * w,
                        - dy * w,
                        dz,
                    ], dtype=float)
                    v_local /= np.linalg.norm(v_local)
                    # Direction globale
                    v_global = R @ v_local
                    dir_mat[i, j] = v_global
                    # Angles sphériques
                    phi   = np.arctan2(v_global[1], v_global[0])   # [-π, π]
                    theta = np.arccos(v_global[2])                 # [0, π]
                    theta_os = np.arccos(v_local[2])
                    phi_mat[i, j]   = phi
                    theta_mat[i, j] = theta 
                    theta_os_mat[i,j] = theta_os
            self.directions_matrix[conf.name] = dir_mat
            self.azimuth_matrix[conf.name] = tools.wrapToPi(phi_mat)
            self.zenith_matrix[conf.name]  = theta_mat
            self.zenith_os_matrix[conf.name]  = theta_os_mat
        self.rotation_matrix = R


    def get_pixel_xy(self):
        """
        Position XY pixels
        """
        func = lambda xf,xr: xf-xr
        front_panel, rear_panel = self.panels[0], self.panels[-1]
        nbars_xf, nbars_yf  = front_panel.matrix.nbars_x,front_panel.matrix.nbars_y
        nbars_xr, nbars_yr  = rear_panel.matrix.nbars_x,rear_panel.matrix.nbars_y
        barNoXf, barNoYf = np.arange(1, nbars_xf+1),np.arange(1, nbars_yf+1)
        barNoXr, barNoYr = np.arange(1, nbars_xr+1),np.arange(1, nbars_yr+1)
        DX_min, DX_max = np.min(barNoXf) - np.max(barNoXr) ,  np.max(barNoXf) - np.min(barNoXr) 
        DY_min, DY_max = np.min(barNoYf) - np.max(barNoYr) ,  np.max(barNoYf) - np.min(barNoYr) 
        res_dx = np.tile(np.mgrid[DX_min:DX_max+1:1],  (2*nbars_xf-1, 1)).T
        res_dy = np.tile(np.mgrid[DY_min:DY_max+1:1],  (2*nbars_yf-1, 1))
        mat_dx, mat_dy = np.zeros(shape=(res_dx.shape[0],res_dx.shape[1],2)), np.zeros(shape=(res_dy.shape[0],res_dy.shape[1],2))
        for i in range(1,nbars_xf+1): 
            for j in np.flip(range(1,nbars_xf+1)): 
                mat_dx[res_dx==func(i,j),:] = [i,j]
                mat_dy[res_dy==func(i,j),:] = [i,j]
        mat = np.concatenate((mat_dx, mat_dy), axis=2)
        return mat
    

    def plot3D(self, fig, ax, position:np.ndarray=np.zeros(3)):
        '''
        Input:
        - 'ax' (plt.Axes3D) : e.g 'ax = fig.add_subplot(111, projection='3d')'
        - 'position' (np.ndarray)
        '''
        zticks=[]
        for p in self.panels:
            w  = float(p.matrix.scintillator.width)
            nbars_x = int(p.matrix.nbars_x)
            nbars_y = int(p.matrix.nbars_y)
            sx = w*nbars_x 
            sy = w*nbars_y
            x, y = np.linspace(position[0], position[0]+sx , nbars_x+1 ), np.linspace(position[0], position[0]+sy , nbars_x+1 )
            X, Y = np.meshgrid(x, y)
            zpos = position[2]
            Z = np.ones(shape=X.shape)*(zpos + p.position.z)
            ax.text(0, 0, zpos + p.position.z, p.position.loc, 'y', alpha=0.5, color='grey')#, rotation_mode='default')
            ax.plot_surface(X,Y,Z, alpha=0.2, color='greenyellow', edgecolor='turquoise' )
            zticks.append(Z[0,0])
        ###shield panel
        X = np.linspace(position[0], position[0]+sx ,2 )
        Y = np.linspace(position[0], position[0]+sy ,2 )
        X, Y = np.meshgrid(X, Y)
        zshield = self.panels[-1].position.z/2
        Z = np.ones(shape=X.shape)*(zpos + zshield)
        ax.plot_surface(X,Y,Z, alpha=0.1, color='none', edgecolor='tomato' )
        ax.text(0, 0, zshield, "Shielding", 'y', alpha=0.5, color='grey')
        panel_side=float(self.panels[0].matrix.nbars_x)*float(self.panels[0].matrix.scintillator.width)
        ax.set_xlabel("X [mm]")
        ax.set_ylabel("Y [mm]")
        ax.set_xticks(np.linspace(0, panel_side, 3))
        ax.set_yticks(np.linspace(0, panel_side, 3))
        ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.invert_zaxis()
        ax.set_zticks(zticks)
        ax.set_zticklabels([])
        ax = fig.add_axes(tools.MyAxes3D(ax, 'l'))
        ax.set_zticks(zticks)
        ax.set_zticklabels([])
        ax.annotate("Z", xy=(0.5, .5), xycoords='axes fraction', xytext=(0.04, .78),)
        return ax

    # @numba.njit
    def adjust_height(
        self,
        x_edges, y_edges, z_edges,
        mask_voxel,
        offset=1.0
    ):
        """
        Vérifie si le détecteur est sous la topographie voxelisée.
        Si oui, le repositionne à (surface + offset).
        Retour :
            new_z0, was_adjusted (bool)
        """
        nx = len(x_edges) - 1
        ny = len(y_edges) - 1
        nz = len(z_edges) - 1
        # Trouver indices i, j
        i = -1
        j = -1
        x0, y0, z0 = self.coordinates
        for ix in range(nx):
            if x_edges[ix] <= x0 < x_edges[ix + 1]:
                i = ix
                break
        for iy in range(ny):
            if y_edges[iy] <= y0 < y_edges[iy + 1]:
                j = iy
                break
        if i == -1 or j == -1:
            # détecteur hors grille horizontale
            return z0, False
        # Chercher voxel actif le plus haut
        top_z = -1e30
        found = False
        for k in range(nz):
            vid = i + nx * (j + ny * k)
            if mask_voxel[vid] == 1:
                top_z = z_edges[k + 1]  # sommet du voxel
                found = True
        if not found:
            # aucune topographie sous ce point
            return z0, False
        surface_z = top_z
        znew = surface_z + offset
        if z0 < znew:
            self.coordinates[2] = znew
            return znew, True
        return z0, False

scint_Fermi = Scintillator(type="Fermilab", length=800, width=50, thickness=7 )
scint_JINR = Scintillator(type="JINR", length=800, width=25, thickness=7 )
matrixv1_1 = Matrix(version="1.1",  scintillator=scint_Fermi, nbars_x=16, nbars_y=16, wls_type="BCF91A",fiber_out="TR644 POM")
matrixv2_0 = Matrix(version="2.0",  scintillator=scint_JINR, nbars_x=32, nbars_y=32, wls_type="Y11",fiber_out="TR644 POM")

survey_path = STRUCT_DIR
souf_tel_path = survey_path / "soufriere" / "telescope"
cop_tel_path  = survey_path / "copahue" / "telescope"

##BR: BaronRouge GW Rocher Fendu 2015-2019 3 matrices = 1 * v1.1 + 2 * v2.0 (mapping might be wrong)
tel_name = 'BR'
tel_BR = Telescope(name=tel_name)

tel_BR.coordinates = np.array([643345.81, 1774030.46,1267])
tel_BR.azimuth = 297#295.
tel_BR.zenith = 80#16.
tel_BR.elevation = round(90.0-tel_BR.zenith, 1) #16
tel_BR.site = "Rocher Fendu - Soufrière"
tel_BR.color = "red"

chmap32 = ChannelMap(file=str( souf_tel_path / tel_name / "channel_bar_map" / "mapping32x32.json"))
chmap16 = ChannelMap(file=str( souf_tel_path / tel_name / "channel_bar_map" / "mapping16x16.json"))
id_front = 23
front_panel = Panel(matrix = matrixv2_0, id=id_front, position=Position(PositionEnum.Front,0,0), channelmap=chmap32)
front_pmt = PMT(id=id_front, panel=front_panel, channelmap=chmap32) 
id_middle = 24
middle1_panel = Panel(matrix = matrixv1_1, id=id_middle, position=Position(PositionEnum.Middle1,1,600), channelmap=chmap16)
middle1_pmt = PMT(id=id_middle, panel=middle1_panel, channelmap=chmap16) 
id_rear = 25
rear_panel = Panel(matrix = matrixv2_0, id=id_rear, position=Position(PositionEnum.Rear,2,1200), channelmap=chmap32)
rear_pmt = PMT(id=id_rear, panel=rear_panel, channelmap=chmap32) 
conf_name = '3p1'
Config_3p1_32x32 = PanelConfig(name = conf_name, 
                               panels=[front_panel, middle1_panel, rear_panel],
                               pmts=[front_pmt, middle1_pmt, rear_pmt])
tel_BR[conf_name] = Config_3p1_32x32
tel_BR.panels = Config_3p1_32x32.panels
tel_BR.pmts = Config_3p1_32x32.pmts

#####COPAHUE 
tel_name = 'COP'
tel_COP = Telescope(name=tel_name)

tel_COP.coordinates = np.array([310722.14, 5808130.41, 2543.]) 
tel_COP.azimuth = 262.0#295
tel_COP.zenith = 86.0#16
tel_COP.elevation = round(90.0-tel_COP.zenith,1)#16
tel_COP.site = "Copahue"
tel_COP.color = "grey"

chmap32 = ChannelMap(file=str( cop_tel_path / tel_name / "channel_bar_map" / "mapping32x32.json"))
chmap16 = ChannelMap(file=str( cop_tel_path / tel_name / "channel_bar_map" / "mapping16x16.json"))
front_panel = Panel(matrix = matrixv2_0, id=0, position=Position(PositionEnum.Front,0,0), channelmap=chmap32)
front_pmt = PMT(id=0, panel=front_panel, channelmap=chmap32) 
middle1_panel = Panel(matrix = matrixv1_1, id=1, position=Position(PositionEnum.Middle1,1,600), channelmap=chmap16)
middle1_pmt = PMT(id=1, panel=middle1_panel, channelmap=chmap16) 
rear_panel = Panel(matrix = matrixv2_0, id=2, position=Position(PositionEnum.Rear,2,1200), channelmap=chmap32)
rear_pmt = PMT(id=2, panel=rear_panel, channelmap=chmap32) 
conf_name = '3p1'
Config_3p1_32x32 = PanelConfig(name = conf_name, 
                               panels=[front_panel, middle1_panel, rear_panel],
                               pmts=[front_pmt, middle1_pmt, rear_pmt])
tel_COP[conf_name] = Config_3p1_32x32
tel_COP.panels = Config_3p1_32x32.panels
tel_COP.pmts = Config_3p1_32x32.pmts

#####OM: OrangeMecanique GW Fente du Nord 2017-2019 3 matrices = 1 * v1.1 + 2 * v2.0
tel_name = 'OM'
tel_OM = Telescope(name=tel_name)

tel_OM.coordinates = np.array([642954.802937528, 1774560.94061667, 1344.6 + 1.5])
tel_OM.azimuth = 192
tel_OM.zenith = 76.6
tel_OM.elevation = round(90.-tel_OM.zenith,1)
tel_OM.site = "Fente du Nord - Soufrière"
tel_OM.color = "orange"

chmap32 = ChannelMap(file=str( souf_tel_path / tel_name / "channel_bar_map" / "mapping32x32.json"))
chmap16 = ChannelMap(file=str( souf_tel_path / tel_name / "channel_bar_map" / "mapping16x16.json"))
front_panel = Panel(matrix = matrixv2_0, id=0, position=Position(PositionEnum.Front,0,0), channelmap=chmap32)
front_pmt = PMT(id=0, panel=front_panel, channelmap=chmap32) 
middle1_panel = Panel(matrix = matrixv1_1, id=1, position=Position(PositionEnum.Middle1,1,600), channelmap=chmap16)
middle1_pmt = PMT(id=1, panel=middle1_panel, channelmap=chmap16) 
rear_panel = Panel(matrix = matrixv2_0, id=2, position=Position(PositionEnum.Rear,2,1200), channelmap=chmap32)
rear_pmt = PMT(id=2, panel=rear_panel, channelmap=chmap32) 
conf_name = '3p1'
Config_3p1_32x32 = PanelConfig(name = conf_name, 
                               panels=[front_panel, middle1_panel, rear_panel],
                               pmts=[front_pmt, middle1_pmt, rear_pmt])
tel_OM[conf_name] = Config_3p1_32x32
tel_OM.panels = Config_3p1_32x32.panels
tel_OM.pmts = Config_3p1_32x32.pmts

#####SB: SacreBleu GW Savane-à-mulets (ouest NJ) 2017-2019 3 matrices = 1 * v1.1 + 2 * v2.0
tel_name = 'SB'
tel_SB = Telescope(name=tel_name)

tel_SB.coordinates = np.array([642611.084416928, 1773797.5200942, 1185.])
tel_SB.azimuth = 40.0#44.9
tel_SB.zenith = 79.0#75
tel_SB.elevation = round(90 - tel_SB.zenith ,1)
tel_SB.site = "Savane-à-mulets - Soufrière"
tel_SB.color = "blue"

chmap32 = ChannelMap(file=str( souf_tel_path / tel_name / "channel_bar_map" / "mapping32x32.json"))
chmap16 = ChannelMap(file=str( souf_tel_path / tel_name / "channel_bar_map" / "mapping16x16.json"))
front_panel = Panel(matrix = matrixv2_0, id=26, position=Position(PositionEnum.Front,0,0), channelmap=chmap32)
front_pmt = PMT(id=26, panel=front_panel, channelmap=chmap32) 
middle1_panel = Panel(matrix = matrixv1_1, id=27, position=Position(PositionEnum.Middle1,1,600), channelmap=chmap16)
middle1_pmt = PMT(id=27, panel=middle1_panel, channelmap=chmap16) 
rear_panel = Panel(matrix = matrixv2_0, id=28, position=Position(PositionEnum.Rear,2,1200), channelmap=chmap32)
rear_pmt = PMT(id=28, panel=rear_panel, channelmap=chmap32) 
conf_name = '3p1'
Config_3p1_32x32 = PanelConfig(name = conf_name, 
                               panels=[front_panel, middle1_panel, rear_panel],
                               pmts=[front_pmt, middle1_pmt, rear_pmt])
tel_SB[conf_name] = Config_3p1_32x32
tel_SB.panels = Config_3p1_32x32.panels
tel_SB.pmts = Config_3p1_32x32.pmts

####SNJ: SuperNainJaune GW Parking 2019
tel_name = 'SNJ'
tel_SNJ = Telescope(name=tel_name)
tel_SNJ.coordinates = np.array([642782.001377887, 1773682.54931093, 1143.])
tel_SNJ.azimuth = 18.0#20.5#20
tel_SNJ.zenith = 74.9#16
tel_SNJ.elevation = round(90-tel_SNJ.zenith, 1)
tel_SNJ.site = "Parking - Soufrière"
tel_SNJ.color = "yellow"

channelmap = ChannelMap(file=str( souf_tel_path / tel_name / "channel_bar_map" / "mapping.json"))
front_panel = Panel(matrix = matrixv1_1, id=9, position=Position(PositionEnum.Front,0,0), channelmap=channelmap)
front_pmt = PMT(id=9, panel=front_panel, channelmap=channelmap) 
middle1_panel = Panel(matrix = matrixv1_1, id=10, position=Position(PositionEnum.Middle1,1,600), channelmap=channelmap)
middle1_pmt = PMT(id=10, panel=middle1_panel, channelmap=channelmap) 
middle2_panel = Panel(matrix = matrixv1_1, id=11, position=Position(PositionEnum.Middle2,2,1200), channelmap=channelmap)
middle2_pmt = PMT(id=11, panel=middle2_panel, channelmap=channelmap) 
rear_panel = Panel(matrix = matrixv1_1, id=12, position=Position(PositionEnum.Rear,3,1800), channelmap=channelmap)
rear_pmt = PMT(id=12, panel=rear_panel, channelmap=channelmap) 
conf_name = '3p1'
Config_3p1_16x16 = PanelConfig(name = conf_name, 
                               panels=[front_panel, middle1_panel, middle2_panel],
                               pmts=[front_pmt, middle1_pmt, middle2_pmt])
tel_SNJ[conf_name] = Config_3p1_16x16
conf_name = '3p2'
Config_3p2_16x16 = PanelConfig(name = conf_name,
                               panels=[middle1_panel, middle2_panel, rear_panel],
                               pmts=[middle1_pmt, middle2_pmt, rear_pmt])
tel_SNJ[conf_name] = Config_3p2_16x16
conf_name = '4p'
Config_4p_16x16 = PanelConfig(name = conf_name, 
                              panels=[front_panel, middle1_panel, middle2_panel, rear_panel],
                              pmts=[front_pmt, middle1_pmt, middle2_pmt, rear_pmt])
tel_SNJ[conf_name] = Config_4p_16x16
tel_SNJ.panels = Config_4p_16x16.panels
tel_SNJ.pmts = Config_4p_16x16.pmts

##SBR: SuperBaronRouge GW Rocher Fendu 2021-2022 4 matrices = 4 * v1.1
tel_name = 'SBR'
tel_SBR = Telescope(name=tel_name)

tel_SBR.coordinates = np.array([643345.81, 1774030.46,1267])
tel_SBR.azimuth = 297 #?
tel_SBR.zenith = 80 #?
tel_SBR.elevation = round(90.0-tel_BR.zenith, 1) #16
tel_SBR.site = "Rocher Fendu - Soufrière"
tel_SBR.color = "red"

channelmap = ChannelMap(file=str( souf_tel_path / tel_name / "channel_bar_map" / "mapping.json"))
front_panel = Panel(matrix = matrixv1_1, id=0, position=Position(PositionEnum.Front,0,0), channelmap=channelmap)
middle1_panel = Panel(matrix = matrixv1_1, id=2, position=Position(PositionEnum.Middle1,1,600), channelmap=channelmap)
front_middle1_pmt = PMT(id=6, panel=[front_panel, middle1_panel], channelmap=channelmap)

middle2_panel = Panel(matrix = matrixv1_1, id=1, position=Position(PositionEnum.Middle2,2,1200), channelmap=channelmap)
rear_panel = Panel(matrix = matrixv1_1, id=3, position=Position(PositionEnum.Rear,3,1800), channelmap=channelmap)
middle2_rear_pmt = PMT(id=7, panel=[middle2_panel, rear_panel], channelmap=channelmap) 
conf_name = '3p1'
Config_3p1_16x16 = PanelConfig(name = conf_name, 
                               panels=[front_panel, middle1_panel, middle2_panel],
                               pmts=[front_middle1_pmt, middle2_rear_pmt])
tel_SBR[conf_name] = Config_3p1_16x16
conf_name = '3p2'
Config_3p2_16x16 = PanelConfig(name = conf_name,
                               panels=[middle1_panel, middle2_panel, rear_panel],
                               pmts=[front_middle1_pmt, middle2_rear_pmt])
tel_SBR[conf_name] = Config_3p2_16x16
conf_name = '4p'
Config_4p_16x16 = PanelConfig(name = conf_name, 
                              panels=[front_panel, middle1_panel, middle2_panel, rear_panel],
                              pmts=[front_middle1_pmt, middle2_rear_pmt])
tel_SBR[conf_name] = Config_4p_16x16
tel_SBR.panels = Config_4p_16x16.panels
tel_SBR.pmts = Config_4p_16x16.pmts

##SXF: GW Faille du 30 Août
tel_name = 'SXF'
tel_SXF = Telescope(name=tel_name)

tel_SXF.coordinates = np.array([643109,1773943,1275.9]) #?
tel_SXF.azimuth = 352.0 #?
tel_SXF.zenith = 90-27 #?
tel_SXF.elevation = round(90.0-tel_BR.zenith, 1) #16
tel_SXF.site = "Faille du 30 Août - Soufrière"
tel_SXF.color = "purple"

channelmap = ChannelMap(file=str( souf_tel_path / tel_name / "channel_bar_map" / "mapping.json"))
front_panel = Panel(matrix = matrixv1_1, id=0, position=Position(PositionEnum.Front,0,0), channelmap=channelmap)
middle1_panel = Panel(matrix = matrixv1_1, id=2, position=Position(PositionEnum.Middle1,1,600), channelmap=channelmap)
front_middle1_pmt = PMT(id=6, panel=[front_panel, middle1_panel], channelmap=channelmap)

middle2_panel = Panel(matrix = matrixv1_1, id=1, position=Position(PositionEnum.Middle2,2,1200), channelmap=channelmap)
rear_panel = Panel(matrix = matrixv1_1, id=3, position=Position(PositionEnum.Rear,3,1800), channelmap=channelmap)
middle2_rear_pmt = PMT(id=7, panel=[middle2_panel, rear_panel], channelmap=channelmap) 
conf_name = '3p1'
Config_3p1_16x16 = PanelConfig(name = conf_name, 
                               panels=[front_panel, middle1_panel, middle2_panel],
                               pmts=[front_middle1_pmt, middle2_rear_pmt])
tel_SXF[conf_name] = Config_3p1_16x16
conf_name = '3p2'
Config_3p2_16x16 = PanelConfig(name = conf_name,
                               panels=[middle1_panel, middle2_panel, rear_panel],
                               pmts=[front_middle1_pmt, middle2_rear_pmt])
tel_SXF[conf_name] = Config_3p2_16x16
conf_name = '4p'
Config_4p_16x16 = PanelConfig(name = conf_name, 
                              panels=[front_panel, middle1_panel, middle2_panel, rear_panel],
                              pmts=[front_middle1_pmt, middle2_rear_pmt])
tel_SXF[conf_name] = Config_4p_16x16
tel_SXF.panels = Config_4p_16x16.panels
tel_SXF.pmts = Config_4p_16x16.pmts


DICT_TEL = { 'SNJ': tel_SNJ, 'BR': tel_BR, 'OM': tel_OM, 'SB': tel_SB, 'COP' : tel_COP, 'SBR': tel_SBR, 'SXF': tel_SXF }


def str2telescope(v):
   
    if isinstance(v, Telescope):
       return v
    if v in list(DICT_TEL.keys()):
        return DICT_TEL[v]
    elif v in [f"tel_{k}" for k in list(DICT_TEL.keys()) ]:
        return DICT_TEL[v[4:]]
    elif v in [ k.lower() for k in list(DICT_TEL.keys())]:
        return DICT_TEL[v.upper()]
    elif v in [f"tel_{k.lower()}" for k in list(DICT_TEL.keys()) ]:
        return DICT_TEL[v[4:].upper()]
    else:
        raise argparse.ArgumentTypeError('Input telescope does not exist.')


if __name__ == '__main__':
    print(tel_SNJ.pmts[0].channelmap)



