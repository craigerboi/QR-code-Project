"""
This is an outline of how the data was generated
"""
import numpy as np

from tqdm import tqdm
import pickle
from aflow import *
import pandas as pd

"""
Using the aflow module written by Conrad Rosenbrock, we can query the AFLOW database subject to certain constraints which is very nice.
"""
heus_result = search(batch_size=10000
                ).filter( K.natoms==4).filter(K.nspecies==3).filter((K.spacegroup_relax==216) |
                    (K.spacegroup_relax==225) | (K.spacegroup_relax==119) | (K.spacegroup_relax==139)).select(
                                                                                            K.enthalpy_formation_atom,
                                                                                            K.spacegroup_relax,
                                                                                            K.spin_atom,
                                                                                            K.compound, 
                                                                                            K.volume_cell)

heus_result.finalize()

lpy_of_formation=[]
mag_mom=[]
compound=[]
entries=[]
space_group=[]
volume=[]

for entry in heus_result:
    enthalpy_of_formation.append(entry.enthalpy_formation_atom)
    mag_mom.append(abs(entry.spin_atom))
    compound.append(entry.compound)
    entries.append(entry)
    space_group.append(entry.spacegroup_relax)
    volume.append(entry.volume_cell)

df = pd.DataFrame()
df['Compound'] = compound
df['spacegroup'] = space_group
df['Formation_ene'] = enthalpy_of_formation
df['Volume'] = volume
df['Magnetic moment'] = mag_mom

di = {225: 0, 216: 1, 139:2, 119:3}
df['spacegroup'].replace(di, inplace=True)

"""
Now we have a dataset containing information on Heusler alloys and each one's simulated formation energy, space group, volume and magentic moment. Further work then must be done to represent each compound as a vector. All you need here is python dictionaries mapping strings of elements to values of chemical properties. Then you must make a function that can take a compound string as input and return a vector and you can do machine learning on the data!
"""
