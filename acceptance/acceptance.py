import numpy as np

from telescope import Telescope

class GeometricalAcceptance:
    def __init__(self, telescope:Telescope):
        """
        
        """
        self.tel=telescope
        self.maps = {}

    def angular_acceptance_map(
        self,
        u_edges,
        v_edges,
        delta_z,
        Lx=80.0,
        Ly=80.0,
    ):
        """
        Acceptance géométrique normalisée A(u,v)
        selon Thomas & Willis (rectangles).

        u_edges, v_edges : bins (dx/dz, dy/dz)
        delta_z : séparation entre plans extrêmes (cm)
        """
        # centres des bins
        u = 0.5 * (u_edges[:-1] + u_edges[1:])
        v = 0.5 * (v_edges[:-1] + v_edges[1:])
        U, V = np.meshgrid(u, v, indexing="ij")
        Ax = np.maximum(0.0, 1.0 - np.abs(U) * delta_z / Lx)
        Ay = np.maximum(0.0, 1.0 - np.abs(V) * delta_z / Ly)
        A = Ax * Ay
        return A
    
    def correct_histogram(self, h, u_edges, v_edges, delta_z, Lx=80.0, Ly=80.0, eps=1e-12):
        A = self.angular_acceptance_map(u_edges, v_edges, delta_z, Lx, Ly)
        Hcorr = h / (A + eps)
        return Hcorr