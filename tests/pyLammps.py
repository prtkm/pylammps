
import os
import shutil
import shlex
import time
from subprocess import Popen, PIPE
from threading import Thread
from re import compile as re_compile, IGNORECASE
from tempfile import mkdtemp, NamedTemporaryFile, mktemp as uns_mktemp
import numpy as np
import decimal as dec
from ase import Atoms
from ase.parallel import paropen
from ase.units import GPa
from ase.io import read,write
import numpy as np
from filecmp import cmp
#from pylammps import *


class prism:
# This class is directly copied from lammpsrun.py
    def __init__(self, cell, pbc=(True,True,True), digits=10):
        """Create a lammps-style triclinic prism object from a cell

        The main purpose of the prism-object is to create suitable
        string representations of prism limits and atom positions
        within the prism.
        When creating the object, the digits parameter (default set to 10)
        specify the precission to use.
        lammps is picky about stuff being within semi-open intervals,
        e.g. for atom positions (when using create_atom in the in-file),
        x must be within [xlo, xhi).
        """
        a, b, c = cell
        an, bn, cn = [np.linalg.norm(v) for v in cell]

        alpha = np.arccos(np.dot(b, c)/(bn*cn))
        beta  = np.arccos(np.dot(a, c)/(an*cn))
        gamma = np.arccos(np.dot(a, b)/(an*bn))

        xhi = an
        xyp = np.cos(gamma)*bn
        yhi = np.sin(gamma)*bn
        xzp = np.cos(beta)*cn
        yzp = (bn*cn*np.cos(alpha) - xyp*xzp)/yhi
        zhi = np.sqrt(cn**2 - xzp**2 - yzp**2)

        # Set precision
        self.car_prec = dec.Decimal('10.0') ** \
            int(np.floor(np.log10(max((xhi,yhi,zhi))))-digits)
        self.dir_prec = dec.Decimal('10.0') ** (-digits)
        self.acc = float(self.car_prec)
        self.eps = np.finfo(xhi).eps

        # For rotating positions from ase to lammps
        Apre = np.array(((xhi, 0,   0),
                         (xyp, yhi, 0),
                         (xzp, yzp, zhi)))
        self.R = np.dot(np.linalg.inv(cell), Apre)

        # Actual lammps cell may be different from what is used to create R
        def fold(vec, pvec, i):
            p = pvec[i]
            x = vec[i] + 0.5*p
            n = (np.mod(x, p) - x)/p
            return [float(self.f2qdec(a)) for a in (vec + n*pvec)]

        Apre[1,:] = fold(Apre[1,:], Apre[0,:], 0)
        Apre[2,:] = fold(Apre[2,:], Apre[1,:], 1)
        Apre[2,:] = fold(Apre[2,:], Apre[0,:], 0)

        self.A = Apre
        self.Ainv = np.linalg.inv(self.A)

        if self.is_skewed() and \
                (not (pbc[0] and pbc[1] and pbc[2])):
            raise RuntimeError('Skewed lammps cells MUST have '
                               'PBC == True in all directions!')

    def f2qdec(self, f):
        return dec.Decimal(repr(f)).quantize(self.car_prec, dec.ROUND_DOWN)

    def f2qs(self, f):
        return str(self.f2qdec(f))

    def f2s(self, f):
        return str(dec.Decimal(repr(f)).quantize(self.car_prec, dec.ROUND_HALF_EVEN))

    def dir2car(self, v):
        "Direct to cartesian coordinates"
        return np.dot(v, self.A)

    def car2dir(self, v):
        "Cartesian to direct coordinates"
        return np.dot(v, self.Ainv)

    def fold_to_str(self,v):
        "Fold a position into the lammps cell (semi open), return a tuple of str"
        # Two-stage fold, first into box, then into semi-open interval
        # (within the given precission).
        d = [x % (1-self.dir_prec) for x in
             map(dec.Decimal, map(repr, np.mod(self.car2dir(v) + self.eps, 1.0)))]
        return tuple([self.f2qs(x) for x in
                      self.dir2car(map(float, d))])

    def get_lammps_prism(self):
        A = self.A
        return (A[0,0], A[1,1], A[2,2], A[1,0], A[2,0], A[2,1])

    def get_lammps_prism_str(self):
        "Return a tuple of strings"
        p = self.get_lammps_prism()
        return tuple([self.f2s(x) for x in p])

    def pos_to_lammps_str(self, position):
        "Rotate an ase-cell postion to the lammps cell orientation, return tuple of strs"
        return tuple([self.f2s(x) for x in np.dot(position, self.R)])

    def pos_to_lammps_fold_str(self, position):
        "Rotate and fold an ase-cell postion into the lammps cell, return tuple of strs"
        return self.fold_to_str(np.dot(position, self.R))

    def is_skewed(self):
        acc = self.acc
        prism = self.get_lammps_prism()
        axy, axz, ayz = [np.abs(x) for x in prism[3:]]
        return (axy >= acc) or (axz >= acc) or (ayz >= acc)

def order(atoms,specorder):
    """Returns ordered species list"""
    if specorder == None:
         symbols = atoms.get_chemical_symbols()
         species = sorted(list(set(symbols)))
    else:
         species=specorder
    return species


class pylammps:
    """This is a calculator that allows the use of some features of Lammps through ASE"""

    def __init__(self, lammpsdir=None, **kwargs):

        if lammpsdir == None:
                self.lammpsdir = os.getcwd()

        else:
                self.lammpsdir = lammpsdir
        self.lammpsdir = os.path.expanduser(self.lammpsdir)
        self.cwd = os.getcwd()
        self.kwargs = kwargs

        if lammpsdir == None:
                self.initialize(**self.kwargs)

    def __enter__(self):
        """
        On enter, make sure directory exists, create it if necessary,
        and change into the directory. Then return the calculator
        """
        # Make directory if it doesn't already exist
        if not os.path.isdir(self.lammpsdir):
            os.makedirs(self.lammpsdir)

        # Now change into new working dir
        os.chdir(self.lammpsdir)
        self.initialize(**self.kwargs)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        on exit, change back to the original directory
        """

        os.chdir(self.cwd)
        return


    def initialize(self, atoms=None, **kwargs):
        '''We need an extra initialize since a lot of the things we need to do
        can only be done once we're inside the directory, which happens after
        the initial __init__'''

        # At this point, we want to determine the state of the directory to
        # TODO decide whether we need to start a new calculation, restart the calculation or read data

        self.write_lammps_in(**self.kwargs)
        self.write_lammps_data(**self.kwargs)
        return


    def calculate(self):
        """Generate necessary files in working directory and run QuantumEspresso

        The method first writes a [name].in file. Then it
        """
   #     if self.status == 'running':
   #         raise LammpsRunning('Running', os.getcwd())
   #     if (self.status == 'done'
   #         and self.converged == False):
   #         raise LammpsNotConverged('Not Converged', os.getcwd())

  #      if self.calculation_required(force=force):
  #          self.write_input()
        script = '''#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -V
#$ -N test
#$ -P cr_liion_materials
#$ -l excl=true
#$ -l h_rt=1:00:00
#$ -q regular
#$ -pe openmpi 4

MPI_CMD="mpirun"
LAMMPS=$LAMMPS_COMMAND
$MPI_CMD $LAMMPS < input.lmp'''

        f = paropen('qscript', 'w')
        f.write(script)
        f.close()

        p = Popen(['qsub qscript'],shell=True,stdout=PIPE,stderr=PIPE)
        out, err = p.communicate()
        if out=='' or err !='':
            raise Exception('something went wrong in qsub:\n\n{0}'.format(err))
        f = paropen('jobid','w')
        f.write(out)
        f.close()
        self.status = 'running'
        return

    def write_lammps_in(self, atoms=None,lammps_in='input.lmp',lammps_data='data.lmp',specorder=None,run_type='MD',**kwargs):
        """Method which writes a LAMMPS in file with run parameters and settings."""

        parameters = kwargs
        if isinstance(lammps_in, str):
            f = paropen(lammps_in, 'w')
            close_in_file = True
        else:
            # Expect lammps_in to be a file-like object
            f = lammps_in
            close_in_file = False

        f.write('# (written by pyLammps)\n\n')

        pbc = atoms.get_pbc()
        if 'atom_style' in parameters:
            f.write('atom_style %s\n' %parameters['atom_style'])
        else:
            f.write('atom_syle atomic \n')
        f.write('units metal \n')
        if ('boundary' in parameters):
            f.write('boundary %s \n' % parameters['boundary'])
        else:
            f.write('boundary %c %c %c \n' % tuple('sp'[x] for x in pbc))

        f.write('read_data %s\n\n' % lammps_data)
        species = order(atoms,specorder)
        #Groups
        f.write('# Groups\n')
        for i,spec in enumerate(species):
            f.write('group {0} type {1}\n'.format(spec,i+1))

        # Write interaction stuff
        f.write('\n### interactions \n')
        if ( ('kspace_style' in parameters) and ('pair_style' in parameters) and ('pair_coeff' in parameters)):
            kspace_style = parameters['kspace_style']
            pair_style = parameters['pair_style']
            f.write('kspace_style %s \n' % kspace_style)
            f.write('pair_style %s \n' % pair_style)

            for pair_coeff in parameters['pair_coeff']:
                if pair_coeff[0] != '*':
                    pair_coeff[0] = species.index(pair_coeff[0])+1
                if pair_coeff[1] != '*':
                    pair_coeff[1] = species.index(pair_coeff[1])+1
                if pair_coeff[0] !='*' and pair_coeff[1]!='*':
                    if pair_coeff[0] > pair_coeff[1]:
                        pair_coeff[0], pair_coeff[1] = pair_coeff[1], pair_coeff[0]
                f.write('pair_coeff %s %s %s\n' % tuple(pair_coeff))

        else:
            # default parameters
            # that should always make the LAMMPS calculation run
            f.write('kspace_style	ewald  1.0e-5 \n')
            f.write('pair_style      buck/coul/long 8.0 \n')
            f.write('pair_coeff      * *     0.00  0.100000  0.000000 \n')

        if 'velocity' in parameters:
            f.write('\nvelocity %s \n' %parameters['velocity'])

        if 'fix' in parameters:
            f.write('\nfix %s \n' %parameters['fix'])
        else:
            if 'temp' in parameters:
                temp = parameters['temp']
            else:
                temp = 298
            if temp > 500:
                tdamp = 100
            else:
                tdamp = 40
            f.write('\nfix 1 all nvt temp {0} {0} {1}\n'.format(temp,tdamp))

        if run_type == 'MD' :
            f.write('compute msd all msd\n')
            for spec in species:
                f.write('compute msd{0} {0} msd\n'.format(spec))
            if 'thermo' in parameters:
                thermo = parameters['thermo']
            else:
                thermo = 100
                f.write('\nthermo {0}\n'.format(thermo))

            if 'thermo_style' in parameters:
                thermo_style = parameters['thermo_style']
            else:
                thermo_style = 'custom step time temp pe ke etotal press'

            f.write('thermo_style %s\n' % thermo_style)
            f.write('dump            1 all custom %s xyz.dat id type x y z\n' %thermo)
            f.write('dump            2 all custom %s velocity.dat id type vx vy vz\n' %thermo)
            f.write('dump            3 all custom %s forces.dat id type fx fy fz\n\n' %thermo)
            if 'dt' in parameters:
                dt = parameters['dt']
                f.write('timestep %s' %dt)
            # writing variables to print mean square displacements
            f.write('variable t equal step*dt\n')
            f.write('variable msd_tot equal c_msd[4]\n')
            f.write('fix tot_msd all print %s  "${t}    ${msd_tot}" file total.msd screen no \n' %thermo)
            for spec in species:
                f.write('variable msd_{0} equal c_msd{0}[4]\n'.format(spec))
                f.write('fix {0}_msd all print {1} "${{t}}    ${{msd_{0}}}" file {0}.msd screen no \n'.format(spec,thermo))
            if 'run' in parameters:
                run = parameters['run']
            else:
                run = 10000
            f.write('\nrun %s\n\n' %run)


        f.write('# END')

        if close_in_file:
            f.close()


    def write_lammps_data(self, lammps_data='data.lmp', atoms=None, specorder=None,atom_style='atomic', masses=True, **kwargs):
        """Method which writes atomic structure data to a LAMMPS data file."""


        if isinstance(lammps_data, str):
            f = paropen(lammps_data, 'w')
            close_file = True
        else:
            # Presume fileobj acts like a fileobj
            f = lammps_data
            close_file = False

        if isinstance(atoms, list):
            if len(atoms) > 1:
                raise ValueError('Can only write one configuration to a lammps data file!')
            atoms = atoms[0]
        formula = atoms.get_chemical_formula()
        f.write(f.name + ' (written by ASE) \n')
        f.write('# Formula: %s\n' % formula)
        symbols = atoms.get_chemical_symbols()
        initial_charges=atoms.get_initial_charges()

        species = order(atoms, specorder)

        for i, spec in enumerate(species):
            f.write('# %d %s\n' % (i+1, spec))
        n_atoms = len(symbols)
        f.write('%d \t atoms \n' % n_atoms)
        n_atom_types = len(species)
        f.write('%d  atom types\n' % n_atom_types)

        p = prism(atoms.get_cell())

        xhi, yhi, zhi, xy, xz, yz = p.get_lammps_prism_str()

        f.write('0.0 %s  xlo xhi\n' % xhi)
        f.write('0.0 %s  ylo yhi\n' % yhi)
        f.write('0.0 %s  zlo zhi\n' % zhi)

        if p.is_skewed():
            f.write('%s %s %s  xy xz yz\n' % (xy, xz, yz))
        f.write('\n\n')

        f.write('Atoms \n\n')
        for i, r in enumerate(map(p.pos_to_lammps_str,
                              atoms.get_positions())):
            s = species.index(symbols[i]) + 1
            if atom_style == 'charge':
                q = initial_charges[i]
                f.write('%6d %3d  %6s   %16s %16s %16s\n' % ((i+1, s, q)+tuple(r)))
            else:
                f.write('%6d %3d %16s %16s %16s\n' % ((i+1, s)+tuple(r)))

        if masses == True:
            d = dict(list(set(zip(atoms.get_chemical_symbols(),atoms.get_masses()))))
            f.write('\n\n')
            f.write('Masses\n\n')
            for i,spec in enumerate(species):
                f.write('%d %s\n' %(i+1, d[spec]))
        if close_file:
            f.close()
