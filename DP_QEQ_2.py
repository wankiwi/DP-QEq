
from dmff.utils import pair_buffer_scales, regularize_pairs, jit_condition
import jax.numpy as jnp
from dmff.admp.recip import generate_pme_recip, Ck_1
from dmff.admp.pme import energy_pme
from jax import jit, grad, jacfwd, jacrev, value_and_grad
from jax.scipy.special import erfc
import jaxopt
from define import name2index, name2chi, name2eta, R_Covalence
import os
import dpdata
from tqdm import tqdm
import freud
import numpy as npewald
import shutil
from typing import Tuple, Optional
import numpy as np

def setup_ewald_parameters(
    rc: float,
    ethresh: float, 
    box: Optional[np.ndarray] = None,
    spacing: Optional[float] = None,
    method: str = 'openmm'
) -> Tuple[float, int, int, int]:

    if method == "openmm":
        kappa = np.sqrt(-np.log(2 * ethresh)) / rc
        K1 = np.ceil(2 * kappa * box[0, 0] / 3 / ethresh**0.2)
        K2 = np.ceil(2 * kappa * box[1, 1] / 3 / ethresh**0.2)
        K3 = np.ceil(2 * kappa * box[2, 2] / 3 / ethresh**0.2)
        return kappa, int(K1), int(K2), int(K3)

def get_ewald_parameter(dir_name, type_map, rc, ethresh):
    #ls = dpdata.LabeledSystem(dir_name, "deepmd/raw", type_map)
    ls = dpdata.System("POSCAR", "vasp/poscar", type_map)
    for box in ls["cells"]:
        kappa, K1, K2, K3 = setup_ewald_parameters(rc, ethresh, box)
        print("kappa, K1, K2, K3: ", kappa, K1, K2, K3)
    return kappa, K1, K2, K3

class NeighborListFreud:
    def __init__(self, box, rcut, cov_map, padding=True, max_shape=0):
        if freud is None:
            raise ImportError("Freud not installed.")
        self.fbox = freud.box.Box.from_matrix(box)
        self.rcut = rcut
        self.capacity_multiplier = None
        self.padding = padding
        self.cov_map = cov_map
        self.max_shape = max_shape
    
    def _do_cov_map(self, pairs):
        nbond = self.cov_map[pairs[:, 0], pairs[:, 1]]
        pairs = jnp.concatenate([pairs, nbond[:, None]], axis=1)
        return pairs

    def allocate(self, coords, box=None):
        self._positions = coords  # cache it
        fbox = freud.box.Box.from_matrix(box) if box is not None else self.fbox
        aq = freud.locality.AABBQuery(fbox, coords)
        res = aq.query(coords, dict(r_max=self.rcut, exclude_ii=True))
        nlist = res.toNeighborList()
        nlist = np.vstack((nlist[:, 0], nlist[:, 1])).T
        nlist = nlist.astype(np.int32)
        msk = (nlist[:, 0] - nlist[:, 1]) < 0
        nlist = nlist[msk]
        if self.capacity_multiplier is None:
            if self.max_shape == 0:
                self.capacity_multiplier = int(nlist.shape[0] * 1.5)
            else:
                self.capacity_multiplier = self.max_shape
        
        if not self.padding:
            self._pairs = self._do_cov_map(nlist)
            return self._pairs

        if self.max_shape == 0:
            self.capacity_multiplier = max(self.capacity_multiplier, nlist.shape[0])
        else:
            self.capacity_multiplier = self.max_shape

        padding_width = self.capacity_multiplier - nlist.shape[0]
        if padding_width == 0:
            self._pairs = self._do_cov_map(nlist)
            return self._pairs
        elif padding_width > 0:
            padding = np.ones((self.capacity_multiplier - nlist.shape[0], 2), dtype=np.int32) * coords.shape[0]
            nlist = np.vstack((nlist, padding))
            self._pairs = self._do_cov_map(nlist)
            return self._pairs
        else:
            raise ValueError("padding width < 0")

    def update(self, positions, box=None):
        self.allocate(positions, box)

    @property
    def pairs(self):
        return self._pairs

    @property
    def scaled_pairs(self):
        return self._pairs

    @property
    def positions(self):
        return self._positions

@jit
def ds_pairs(positions, box, pairs):
    pos1 = positions[pairs[:,0].astype(int)]
    pos2 = positions[pairs[:,1].astype(int)]
    box_inv = jnp.linalg.inv(box)
    dpos = pos1 - pos2
    dpos = dpos.dot(box_inv)
    dpos -= jnp.floor(dpos+0.5)
    dr = dpos.dot(box)
    ds = jnp.linalg.norm(dr,axis=1)
    return ds

@jit
def ds_pairs_2(positions, box, pairs):
    pos1 = positions[pairs[:,0].astype(int)]
    pos2 = positions[pairs[:,1].astype(int)]
    box_inv = jnp.linalg.inv(box)
    dpos = pos1 - pos2
    dpos = dpos.dot(box_inv)
    dpos -= jnp.floor(dpos+0.5)
    dr = dpos.dot(box)
    #ds = jnp.linalg.norm(dr,axis=1)
    return dr

def typemap_list_to_symbols(atom_numbs: list, atom_names: list):
    atomic_symbols = []
    idx = 0
    for numb in atom_numbs:
        atomic_symbols.extend((atom_names[idx], )*numb)
        idx += 1
    return atomic_symbols

def generate_get_energy(kappa, K1, K2, K3):
    pme_recip_fn = generate_pme_recip(
        Ck_fn=Ck_1,
        kappa=kappa / 10,
        gamma=False,
        pme_order=6,
        K1=K1,
        K2=K2,
        K3=K3,
        lmax=0,
    )
    def get_energy_kernel(positions, box, pairs, charges, mscales):
        atomCharges = charges
        atomChargesT = jnp.reshape(atomCharges, (-1, 1))
        return energy_pme(
            positions * 10,
            box * 10,
            pairs,
            atomChargesT,
            None,
            None,
            None,
            mscales,
            None,
            None,
            None,
            pme_recip_fn,
            kappa / 10,
            K1,
            K2,
            K3,
            0,
            False,
        )
    def get_energy(positions, box, pairs, charges, mscales):
        return get_energy_kernel(positions, box, pairs, charges, mscales)
    return get_energy

@jit 
def get_Energy_Qeq_2(charges, positions, box, pairs, eta, chi, hardness):
    @jit 
    def get_Energy_PME():
        pme = generate_get_energy(4.3804348, 21, 75, 22)
        e = pme(positions/10, box/10, pairs, charges, mscales=jnp.array([1., 1., 1., 1., 1., 1.]))
        return e
    @jit 
    def get_Energy_Correction():
        ds = ds_pairs(positions, box, pairs)
        buffer_scales = pair_buffer_scales(pairs)
        e_corr_pair = charges[pairs[:,0]] * charges[pairs[:,1]] * erfc(ds / (jnp.sqrt(2) * jnp.sqrt(eta[pairs[:,0]]**2 + eta[pairs[:,1]]**2))) * 1389.35455846 / ds  * buffer_scales
        e_corr_self = charges * charges * 1389.35455846 /(2*jnp.sqrt(jnp.pi) * eta)
        return  -jnp.sum(e_corr_pair) + jnp.sum(e_corr_self)
    @jit
    def get_Energy_Onsite():
        E_tf =  (chi * charges + 0.5 * hardness * charges *charges) * 96.4869
        return jnp.sum(E_tf)
    return (get_Energy_PME() + get_Energy_Correction() + get_Energy_Onsite()) / 96.4869

@jit
def get_hessian(charges, positions, box, pairs, eta, chi, hardness):
    h = jacfwd(jacrev(get_Energy_Qeq_2, argnums=(0)))(charges, positions, box, pairs, eta, chi, hardness)
    return h

def solve_q(charges, positions, box, pairs, eta, chi, hardness):
    A = get_hessian(charges, positions, box, pairs, eta, chi, hardness)

    M = np.zeros((len(charges)+1, len(charges)+1))
    M[:-1,:-1] = A
    M[-1,-1] = 0
    M[-1,:-1] = 1
    M[:-1,-1] = 1

    b = np.zeros(len(charges)+1)
    b[:-1] = -chi
    b[-1] = 0

    q = jnp.linalg.solve(M, b)
    return q

@jit
def get_force(charges, positions, box, pairs, eta, chi, hardness):
    energy,force = value_and_grad(get_Energy_Qeq_2,argnums=(1))(charges, positions, box, pairs, eta, chi, hardness)
    return energy, -force

def get_qeq_energy_and_force(charges, positions, box, pairs, eta, chi, hardness):
    q = solve_q(charges, positions, box, pairs, eta, chi, hardness)
    energy, force = get_force(q[0:-1], positions, box, pairs, eta, chi, hardness)
    return energy, force, q

def get_neighbor_list(box, rc, positions, natoms, padding=True, max_shape=0):
    nbl = NeighborListFreud(box, rc, jnp.zeros([natoms, natoms], dtype=jnp.int32), padding=padding, max_shape=max_shape)
    nbl.allocate(positions)
    pairs = nbl.pairs
    pairs = pairs.at[:, :2].set(regularize_pairs(pairs[:, :2]))
    return pairs

def get_qeq_charge_for_md_traj(file_name="wrapped_trajectory.traj"):
    import ase.io as IO
    from my_utils import cell_to_box

    atoms = IO.read(file_name, index=":")
    nframes = len(atoms)
    #natoms = len(atoms[0])
    iterator = tqdm(range(0, nframes, 1))
    qeq_charges = []
    for iframe in iterator:
        coordinate = atoms[iframe].get_positions()
        cell_tmp = atoms[iframe].get_cell_lengths_and_angles()
        box = cell_to_box(cell_tmp[0], cell_tmp[1], cell_tmp[2], cell_tmp[3], cell_tmp[4], cell_tmp[5])
        symbols = atoms[iframe].get_chemical_symbols()
        eta = jnp.array([R_Covalence[name2index[tmp]] for tmp in symbols])
        chi = jnp.array([name2chi[tmp] for tmp in symbols])
        hardness = jnp.array([name2eta[tmp] for tmp in symbols])
        charges = jnp.array(np.random.random(len(symbols))) # initial guess charges
        pairs = get_neighbor_list(box, 6, coordinate, len(coordinate), padding=True, max_shape=200000)
        energy, force, q = get_qeq_energy_and_force(charges, coordinate, box, pairs, eta, chi, hardness)
        qeq_charges.append(q[0:-1])
    np.savetxt("qeq_charges", np.reshape(qeq_charges, [nframes, -1]), fmt="%.10f")


def get_qeq_energy_and_force_for_file(file_name="POSCAR", file_type="poscar", type_map=["Li", "C", "H", "O", "P", "F"], rc=6):
    if file_type == "poscar":
        ss = dpdata.System(file_name, fmt="vasp/poscar", type_map=type_map)

    natoms = ss.get_natoms()
    box = ss["cells"][0]
    positions = ss["coords"][0]
    pairs = get_neighbor_list(box, rc, positions, natoms, padding=True, max_shape=0)
    symbols = typemap_list_to_symbols(ss["atom_numbs"], ss["atom_names"])
    chi = jnp.array([name2chi[tmp] for tmp in symbols])
    hardness = jnp.array([name2eta[tmp] for tmp in symbols])
    eta = jnp.array([R_Covalence[name2index[tmp]] for tmp in symbols])
    charges = jnp.array(np.random.random(len(symbols)))

    energy, force, q = get_qeq_energy_and_force(charges, positions, box, pairs, eta, chi, hardness)
    #print(energy, force,)
    np.savetxt("qeq_energy.jax", np.reshape(energy, [1, 1]), fmt="%.6f")
    np.savetxt("qeq_force.jax", np.reshape(force, [1, -1]), fmt="%.6f")
    np.savetxt("qeq_charges.jax", np.reshape(q[0:-1], [1, -1]), fmt="%.6f")

def get_nblist_max_shape(dir_name, type_map, rc):
    ls = dpdata.LabeledSystem(dir_name, "deepmd/raw", type_map)

    nframes = ls.get_nframes() 
    #nframes = 10
    natoms = ls.get_natoms()
    pairs_shape = []

    # 0. get chi, hardness, eta, and so on
    symbols = typemap_list_to_symbols(ls["atom_numbs"], ls["atom_names"])
    chi = jnp.array([name2chi[tmp] for tmp in symbols])
    hardness = jnp.array([name2eta[tmp] for tmp in symbols])
    eta = jnp.array([R_Covalence[name2index[tmp]] for tmp in symbols])
    charges = jnp.array(np.random.random(len(symbols))) # initial guess charges

    # 1. get the max shape of pairs for jitable
    for iframe in tqdm(range(0, nframes, 1)):
        box = ls["cells"][iframe]
        positions = ls["coords"][iframe]
        pairs = get_neighbor_list(box, rc, positions, natoms, padding=True, max_shape=0)
        pairs_shape.append(pairs.shape[0])
        
    #print("max shape of pairs: ", np.max(pairs_shape))
    max_pairs_shape = np.max(pairs_shape)
    
    return max_pairs_shape 


def get_qeq_energy_and_force_for_deepmd_dataset(dir_name, type_map, rc, max_pairs_shape):
    ls = dpdata.LabeledSystem(dir_name, "deepmd/raw", type_map)

    nframes = ls.get_nframes() 
    #nframes = 10
    natoms = ls.get_natoms()
    pairs_shape = []

    # 0. get chi, hardness, eta, and so on
    symbols = typemap_list_to_symbols(ls["atom_numbs"], ls["atom_names"])
    chi = jnp.array([name2chi[tmp] for tmp in symbols])
    hardness = jnp.array([name2eta[tmp] for tmp in symbols])
    eta = jnp.array([R_Covalence[name2index[tmp]] for tmp in symbols])
    charges = jnp.array(np.random.random(len(symbols))) # initial guess charges
   
    # 2. get the nblist and then calculate qeq energy, force and charge
    qeq_energy, qeq_force, qeq_charges = [], [], []
    for iframe in tqdm(range(0, nframes, 1)):
        box = ls["cells"][iframe]
        positions = ls["coords"][iframe]
        pairs = get_neighbor_list(box, rc, positions, natoms, padding=True, max_shape=max_pairs_shape)
        energy, force, q = get_qeq_energy_and_force(charges, positions, box, pairs, eta, chi, hardness)
        qeq_energy.append(energy)
        qeq_force.append(force)
        qeq_charges.append(q[0:-1])
    
    # 3. save qeq energy, force and charge
    np.savetxt("qeq_energy", np.reshape(qeq_energy, [nframes, 1]), fmt="%.10f")
    np.savetxt("qeq_force", np.reshape(qeq_force, [nframes, -1]), fmt="%.10f")
    np.savetxt("qeq_charges", np.reshape(qeq_charges, [nframes, -1]), fmt="%.10f")


    # 4. DFT minus QEQ
    dft_force = np.loadtxt("force.raw")
    qeq_force = np.loadtxt("qeq_force")
    force_diff = dft_force - qeq_force

    dft_energy = np.loadtxt("energy.raw")
    qeq_energy = np.loadtxt("qeq_energy")
    energy_diff = dft_energy - qeq_energy

    shutil.move("energy.raw", "original_energy.raw")
    shutil.move("force.raw", "original_force.raw")
    #shutil.move("set.000", "original.set.000")
    os.system("rm -rf set.*")

    np.savetxt("force.raw", np.reshape(force_diff, [nframes, -1]))
    np.savetxt("energy.raw", np.reshape(energy_diff, [nframes, 1]))

    ls = dpdata.LabeledSystem("./", "deepmd/raw", type_map)
    ls.to_deepmd_npy("./")

    
def calculate_model_devi_ase_traj(
        file_name: str = "md.traj",
        file_type: str = "ase/traj",
        type_dict: dict = {"Li": 0, "C": 1, "H": 2, "O": 3, "P": 4, "F": 5},
        pb_file: list = ["graph.000.compress.pb", "graph.001.compress.pb", "graph.002.compress.pb", "graph.003.compress.pb"],
        frequency: int = 100,
):
    from deepmd.infer import DeepPot as DP
    from deepmd.infer import calc_model_devi
    from deepmd.infer.model_devi import write_model_devi_out
    from ase.io import read
    from my_utils import cell_to_box

    graphs = [DP(tmp) for tmp in pb_file]

    if file_type == "ase/traj":
        atoms = read(file_name, index=":")
        nframes = len(atoms)
        devi = []
        iterator = tqdm(range(0, nframes, frequency))
        
        for iframe in iterator:
            result = []
            coordinate = atoms[iframe].get_positions()
            cell_tmp = atoms[iframe].get_cell_lengths_and_angles()
            box = cell_to_box(cell_tmp[0], cell_tmp[1], cell_tmp[2], cell_tmp[3], cell_tmp[4], cell_tmp[5])
            symbols = atoms[iframe].get_chemical_symbols()
            atype = [type_dict[tmp] for tmp in symbols]
            model_devi = calc_model_devi(np.array(coordinate).reshape([1, -1]), np.array(box).reshape([1, -1]), np.array(atype), graphs)
            result.append(iframe)
            result.extend(model_devi[0][1:])
            devi.append(result)
        write_model_devi_out(np.array(devi), "model_devi.out")

def traj_convert():
    from ase.io import read, write
    trajectory = read('md.traj', index=':')  
    for atoms in trajectory:
        atoms.wrap()  
    write('wrapped_trajectory.traj', trajectory)

if __name__ == "__main__":
    import time
    ss = dpdata.System("POSCAR", fmt="vasp/poscar", type_map=["Li", "C", "H", "O", "P", "F"])
    box = ss["cells"][0]
    positions = ss["coords"][0]
    symbols = typemap_list_to_symbols(ss["atom_numbs"], ss["atom_names"])
    time1 = time.process_time()
    chi = determine_chi(box, positions, symbols, mode=2, voltage=12.0, debug_mode=0)
    time2 = time.process_time()
    print(chi, time2-time1)


        