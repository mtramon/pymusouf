### Context
The repository contains Python packages to reconstruct and analyze muography data recorded during two surveys:
- La Soufrière de Guadeloupe, in the Lesser Antilles.
- Copahue volcano, on the Argentina–Chile border.

The detectors used in this study are scintillator-based hodoscopes developed at IP2I Lyon.

### Repository layout
- package configuration in [config/config.yaml](config/config.yaml)
- telescope catalogue in [telescope/telescopes.yaml](telescope/telescopes.yaml)
- internal JSON channel-to-bar mappings in [telescope](telescope)
- lightweight demonstration data files in [sample](sample)
- optional heavy local data under [data_link](data_link) and [struct_link](struct_link)

### Reconstruction
See [processing/README.md](processing/README.md).

### 3D modeling and inversion
See [inversion](inversion).