
import os
import numpy as np
from ase import Atom, Atoms
def read_out(outfile, pbc =True):

        def read_cell(i, line, lines):
            new_cell = []
            if line.lower().startswith('cell_parameters'):
                alat = float(line.split()[-1].translate(None, '()')) * 0.529177249
                for j in range(1, 4):
                    lat = np.array((float(lines[i + j].split()[0]) * alat,
                                    float(lines[i + j].split()[1]) * alat,
                                    float(lines[i + j].split()[2]) * alat))
                    new_cell.append(lat)
                return new_cell
            else:
                return None

        def read_positions(i, line, lines):
            if line.lower().startswith('atomic_positions'):
                new_pos = []
                j = 1
                new_symbol=[]
                while len(lines[i + j].split()) > 3:
                    atom_symbol = lines[i + j].split()[0]

                    new_symbol.append(atom_symbol)
                    atom_pos = np.array((float(lines[i + j].split()[1]),
                                         float(lines[i + j].split()[2]),
                                         float(lines[i + j].split()[3])))
                    new_pos.append(atom_pos)
                    j += 1

                return new_symbol, new_pos
            else:
                return None, None

        out_file = open(outfile, 'r')
        lines = out_file.readlines()


        for i, line in enumerate(lines):
            _cell = read_cell(i, line, lines)
            if not _cell == None:
                cell = _cell

            _symbols, _scaled_pos = read_positions(i, line, lines)
            if not _symbols == None:
                   symbols= _symbols
            if not _scaled_pos == None:
                   scaled_pos=_scaled_pos

        atoms = Atoms(symbols=symbols, cell = cell, scaled_positions=scaled_pos, pbc=pbc)
        return atoms
