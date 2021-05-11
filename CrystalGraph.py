# input : cif file
# output : crystal graph, atom feature vectors, bond feature vectors

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import sys

import encoding

## cif directory
cifdir = 'cif/'


# bond length data
# B. Cordero, "Covalent radii revisited", Dalton Transactions, 2008
atom_features = {'Sc': {'radius': 1.70, 'group': 3, 'period': 4}, \
                 'Ti': {'radius': 1.60, 'group': 4, 'period': 4}, \
                 'V': {'radius': 1.53, 'group': 5, 'period': 4}, \
                 'Cr': {'radius': 1.39, 'group': 6, 'period': 4}, \
                 'Mn': {'radius': 1.39, 'group': 7, 'period': 4}, \
                 'Fe': {'radius': 1.32, 'group': 8, 'period': 4}, \
                 'Co': {'radius': 1.26, 'group': 9, 'period': 4}, \
                 'Ni': {'radius': 1.24, 'group': 10, 'period': 4}, \
                 'Cu': {'radius': 1.32, 'group': 11, 'period': 4}, \
                 'Zn': {'radius': 1.22, 'group': 12, 'period': 4}, \
                 'Y': {'radius': 1.90, 'group': 3, 'period': 5}, \
                 'Zr': {'radius': 1.75, 'group': 4, 'period': 5}, \
                 'Nb': {'radius': 1.64, 'group': 5, 'period': 5}, \
                 'Mo': {'radius': 1.54, 'group': 6, 'period': 5}, \
                 'Tc': {'radius': 1.47, 'group': 7, 'period': 5}, \
                 'Ru': {'radius': 1.46, 'group': 8, 'period': 5}, \
                 'Rh': {'radius': 1.42, 'group': 9, 'period': 5}, \
                 'Pd': {'radius': 1.39, 'group': 10, 'period': 5}, \
                 'Ag': {'radius': 1.45, 'group': 11, 'period': 5}, \
                 'Cd': {'radius': 1.44, 'group': 12, 'period': 5}, \
                 'Hf': {'radius': 1.75, 'group': 4, 'period': 6}, \
                 'Ta': {'radius': 1.70, 'group': 5, 'period': 6}, \
                 'W': {'radius': 1.62, 'group': 6, 'period': 6}, \
                 'Re': {'radius': 1.51, 'group': 7, 'period': 6}, \
                 'Os': {'radius': 1.44, 'group': 8, 'period': 6}, \
                 'Ir': {'radius': 1.41, 'group': 9, 'period': 6}, \
                 'Pt': {'radius': 1.36, 'group': 10, 'period': 6}, \
                 'Au': {'radius': 1.36, 'group': 11, 'period': 6}, \
                 'Hg': {'radius': 1.32, 'group': 12, 'period': 6}}


CATEGORY_NUM = encoding.CATEGORY_NUM
NEIGHBOR_CATEGORY_NUM = 40

TOTAL_CATEGORY_NUM = 2*CATEGORY_NUM + NEIGHBOR_CATEGORY_NUM

fnlist = np.loadtxt('feature_num.txt',dtype=int)
print(fnlist)
if len(np.shape(fnlist)) == 0:
	fnlist = np.array([fnlist])
print(fnlist)
fnstr = '%d' % fnlist[0]
for i in range(1,len(fnlist)):
    fnstr+='_%d' % fnlist[i]

def basis_change(a,b,c,alpha,beta,gamma):
    trans_matrix = np.zeros([3,3],dtype=np.float32)
    v1 = np.array([a,0,0], dtype=np.float32)
    v2 = np.array([b*np.cos(gamma), b*np.sin(gamma), 0])
    v3 = np.array([c*np.cos(beta), c*(np.cos(alpha)-np.cos(gamma)*np.cos(beta))/np.sin(gamma), \
                   c/np.sin(gamma)*np.sqrt(1-np.cos(alpha)**2-np.cos(beta)**2-np.cos(gamma)**2+2*np.cos(alpha)*np.cos(beta)*np.cos(gamma))])
    trans_matrix[0] = v1
    trans_matrix[1] = v2
    trans_matrix[2] = v3
    return np.transpose(trans_matrix)

def unit_cell_expansion(lattices, matrix):
    num = len(lattices[0])
    #num=1
    expanded_lattices = np.zeros([3,num*27],dtype=np.float32)
    coeff = [0, -1, 1]
    count = 0
    for i in coeff:
        for j in coeff:
            for k in coeff:
                new_lattices = np.copy(lattices)
                translation = np.reshape(matrix[:,0] * i + matrix[:,1] * j + matrix[:,2] * k, [3,1])
                new_lattices = new_lattices + translation
                expanded_lattices[:,count*num:(count+1)*num] = new_lattices
                count = count + 1
    return expanded_lattices

def find_neighbor(lattices, expanded_lattices, elements):
    lattice_num = len(lattices[0])
    num = len(expanded_lattices[0])
    connectivity = []
    distance = []
    for i in range(lattice_num):
        neighbor_num = 0
        for j in range(num):
            if i==j:
                continue
            cond1 = np.linalg.norm(lattices[:,i] - expanded_lattices[:,j]) < 6
            cond2 = np.linalg.norm(lattices[:,i] - expanded_lattices[:,j]) < (atom_features[elements[i]]['radius'] + atom_features[elements[j%len(elements)]]['radius'] + 0.25)
            if cond1 and cond2:
                connectivity.append([i,j])
                distance.append(np.linalg.norm(lattices[:,i] - expanded_lattices[:,j]))
                neighbor_num = neighbor_num + 1
        #print(neighbor_num)

    return connectivity, distance

def atom_encoding(a):
    a_one_hot = encoding.atom_encoding(a)
    return a_one_hot

def bond_encoding(d): 
    # 2.4 - 3.4
    # 20 categories
    d_one_hot = np.zeros(NEIGHBOR_CATEGORY_NUM, dtype=np.int32)
    index = int(round((d - 2.4) / 0.025)) #round.. not int
    d_one_hot[index] = 1
    return d_one_hot

def bond_construction(elements,connectivity,distance):  ## connectivity: list of bonded pair [i,j]
    bond_vectors = []
    bond_indices = []
    atom_num = len(elements)
    neighbor_num = np.zeros(atom_num,dtype=np.int32) ## number of neighbor atoms
    for connection in connectivity:
        neighbor_num[connection[0]] += 1
    count = 0
    for i in range(len(elements)): ## atoms in cell
        bond_vector = np.zeros([neighbor_num[i], NEIGHBOR_CATEGORY_NUM], dtype=np.int32)
        bond_index = np.zeros([neighbor_num[i], 2], dtype=np.int32)
        for j in range(neighbor_num[i]): ## atoms bonded to atom i
            #neighbor_atom = elements[connectivity[count][1]%atom_num]  ## connectivity가 무조건 첫번째 atom 번호 작은 것부터 올라가서 그냥 count를 iterator? 로 쓸 수 있는듯
            bond_vector[j] = bond_encoding(distance[count])
            #bond_index.append(connectivity[count][1]%atom_num)
            bond_index[j][0] = connectivity[count][0] % atom_num
            bond_index[j][1] = connectivity[count][1] % atom_num
            count += 1
        bond_vectors.append(bond_vector)
        #print(i,'th atom','appended bond index:',bond_index)
        bond_indices.append(bond_index)

    return bond_vectors, bond_indices

def plot3d(trans_lattices, trans_matrix, elements, connectivity):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    elem_num = len(trans_lattices[0])
    color_list = ['b','r','g','c','m','y','k','w']
    color = 0
    elements.append(elements[-1])
    for i in range(elem_num):
        expanded_lattices = unit_cell_expansion(np.reshape(trans_lattices[:,i],[3,1]), trans_matrix)
        ax.scatter(expanded_lattices[0], expanded_lattices[1], expanded_lattices[2], color = color_list[color])
        if elements[i] != elements[i+1]:
            color = color + 1
    elements.pop()

    total_expanded_lattices = unit_cell_expansion(trans_lattices, trans_matrix)
    for i in range(len(connectivity)):
        ax.plot([trans_lattices[0,connectivity[i][0]], total_expanded_lattices[0,connectivity[i][1]]],[trans_lattices[1,connectivity[i][0]], total_expanded_lattices[1,connectivity[i][1]]],[trans_lattices[2,connectivity[i][0]], total_expanded_lattices[2,connectivity[i][1]]], 'k-')
    plt.show()

def cif_to_vector(filename): #TODO: distinguish input type (numpy array or string)

    try:
        npzfile = np.load(cifdir + filename + '.b40.feature%s.npz' % fnstr)
        atom_vectors = npzfile['atoms']
        bond_vectors = npzfile['bonds']
        bond_indices = npzfile['indices']

        return atom_vectors, bond_vectors, bond_indices

    except FileNotFoundError as e:
        #print('## not npz,',filename)
        f = open(cifdir + filename + '.cif', 'r')
        occupancy = False
        atom_num = 0


        lattices = []
        a, b, c, alpha, beta, gamma = 0, 0, 0, 0, 0, 0
        elements = []

        #Data extrcaction from cif file
        lines = f.readlines()
        for line in lines:
            if occupancy:
                str_list = line.strip().split()
                lattices.append(str_list[2:5])
                elements.append(str_list[-1])
                atom_num = atom_num + 1

            if '_cell_length_a' in line:
                a = float(line.strip().split(' ')[-1])
            elif '_cell_length_b' in line:
                b = float(line.strip().split(' ')[-1])
            elif '_cell_length_c' in line:
                c = float(line.strip().split(' ')[-1])
            elif '_cell_angle_alpha' in line:
                alpha = np.deg2rad(float(line.strip().split(' ')[-1]))
            elif '_cell_angle_beta' in line:
                beta = np.deg2rad(float(line.strip().split(' ')[-1]))
            elif '_cell_angle_gamma' in line:
                gamma = np.deg2rad(float(line.strip().split(' ')[-1]))
            elif '_atom_site_type_symbol' in line: ## only for cif from ase... be careful!!!!!
                occupancy = True
        f.close()

        bond_num = np.zeros([atom_num],dtype=np.int32)
        #print('## lattices:',lattices)
        lattices = np.transpose(np.array(lattices, dtype=np.float32))

        trans_matrix = basis_change(a,b,c,alpha,beta,gamma)
        trans_lattices = np.matmul(trans_matrix,lattices)
        expanded_lattices = unit_cell_expansion(trans_lattices, trans_matrix)
        #print(trans_matrix)

        connectivity, distance = find_neighbor(trans_lattices, expanded_lattices, elements)

        atom_vectors = np.zeros([atom_num, CATEGORY_NUM], dtype=np.float32)
        for i in range(atom_num):
            atom_vectors[i] = atom_encoding(elements[i])

        bond_vectors, bond_indices = bond_construction(elements,connectivity,distance)
        for i in range(len(bond_vectors)):
            bond_num[i] = bond_vectors[i].shape[0]

        #plot3d(trans_lattices,trans_matrix,elements, connectivity)
        npzfile = cifdir + filename + '.b40.feature%s.npz' % fnstr
        np.savez(npzfile, atoms=atom_vectors, bonds=bond_vectors, indices=bond_indices)
        return atom_vectors, bond_vectors, bond_indices

def main():
    filename = 'Pt13'
    if len(sys.argv) == 2:
        filename = sys.argv[1]
    atom_vectors, bond_vectors,bond_indices = cif_to_vector(filename)
    #print(atom_vectors)
    #print(bond_vectors)
    print(np.shape(atom_vectors))
    for i in range(len(bond_vectors)):
        print(i,':',np.shape(bond_vectors[i]))
    #print(bond_vectors[1].shape[0])

if __name__ == "__main__":
    main()