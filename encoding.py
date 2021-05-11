import numpy as np

csvdir = './'

atom_features = {}
with open(csvdir+'/feature.csv','r') as f:
    index = f.readline().strip().split(sep=',')
    k=0
    for line in f.readlines():
        values = line.strip().split(sep=',')
        dict_int = {index[j] : int(values[j]) for j in range(1,4)}
        dict_float = {index[j] : float(values[j]) for j in range(4,len(values))}
        dict_int.update(dict_float)

        atom_features[values[0]] = dict_int


feature_list = {'group' : 12, 'period' : 5, 'electronegativity' : 10, 'ionization' : 10, 'affinity':10, 'volume' :  10, 'radius' : 10, 'atomic number' : 32, 'weight' : 10, 'melting' : 10, 'boiling' : 10,\
                'density' : 10, 'Zeff' : 10, 'polarizability' : 10, 'resistivity' : 10, 'fusion' : 10, 'vaporization' : 10, 'atomization' : 10, 'capacity' : 10, 'valence' : 11, 'd-electron':11}



fnlist = np.loadtxt('feature_num.txt',dtype=int)

if len(np.shape(fnlist)) == 0:
	fnlist = np.array([fnlist])

fnstr = '%d' % fnlist[0]
for i in range(1,len(fnlist)):
    fnstr+='_%d' % fnlist[i]
features0 = list(feature_list.keys())

features= []
for fn in fnlist:
	features.append(features0[fn])

CATEGORY_NUM = 0
for i in range(len(features)):
    CATEGORY_NUM += feature_list[features[i]]

BOND_CATEGORY_NUM = 20

NEIGHBOR_CATEGORY_NUM = CATEGORY_NUM + BOND_CATEGORY_NUM
TOTAL_CATEGORY_NUM = CATEGORY_NUM + NEIGHBOR_CATEGORY_NUM


def atom_encoding(a):
    c = 0
    a_one_hot = np.zeros(CATEGORY_NUM, dtype=np.int32)
    for i in range(len(features)):
        index = 0
        if features[i] =='group':
            index = atom_features[a]['group'] - 1
        elif features[i] == 'period':
            index = atom_features[a]['period'] - 2
        elif features[i] == 'd-electron':
            index = int(atom_features[a]['d-electron'])
        elif features[i] == 'electronegativity':
            index = int((atom_features[a]['electronegativity'] - 1.2) / 0.2)
        elif features[i] == 'ionization':
            if atom_features[a]['ionization'] == atom_features['H']['ionization']:
                index = 0
            elif atom_features[a]['ionization'] == atom_features['N']['ionization']:
                index = 1
            else:
                index = int((np.log(atom_features[a]['ionization']) - 1.65) / (0.6/8))
        elif features[i] == 'affinity':
            index = int((atom_features[a]['affinity'] + 0.8) / 0.32)
        elif features[i] == 'volume':
            index = int((atom_features[a]['volume'] - 5) / 2.5)
        elif features[i] == 'radius':
            if atom_features[a]['radius'] - 1.15 < 0:
                index = 0
            else:
                index = int((atom_features[a]['radius'] - 1.15) / 0.05)
        elif features[i] == 'atomic number':
            index = atom_features[a]['atomic number'] - 1
        elif features[i] == 'weight':
            if atom_features[a]['weight'] < 2:
                index = 0
            elif atom_features[a]['weight'] < 15:
                index = 1
            else:
                index = int((atom_features[a]['weight']-40)/20)

        elif features[i] == 'melting':
            if atom_features[a]['melting'] == atom_features['H']['melting']:
                index = 0
            elif atom_features[a]['melting'] == atom_features['N']['melting']:
                index = 1
            else:
                index = int((atom_features[a]['melting']-300)/400)

        elif features[i] == 'boiling':
            if atom_features[a]['boiling'] == atom_features['H']['boiling']:
                index = 0
            elif atom_features[a]['boiling'] == atom_features['N']['boiling']:
                index = 1
            else:
                index = int((atom_features[a]['boiling']-700)/(4900/8))
        elif features[i] == 'density':
            if atom_features[a]['density'] == atom_features['H']['density']:
                index = 0
            elif atom_features[a]['density'] == atom_features['N']['density']:
                index = 1
            else:
                index = int((atom_features[a]['density']-2)/(21/8))
        elif features[i] == 'Zeff':
            if atom_features[a]['Zeff'] == atom_features['H']['Zeff']:
                index = 0
            elif atom_features[a]['Zeff'] == atom_features['N']['Zeff']:
                index = 1
            else:
                index = int((atom_features[a]['Zeff']-2.7)/(2.6/8))
        elif features[i] == 'polarizability':
            if atom_features[a]['polarizability'] == atom_features['H']['polarizability']:
                index = 0
            elif atom_features[a]['polarizability'] == atom_features['N']['polarizability']:
                index = 1
            else:
                index = int((atom_features[a]['polarizability']-4)/(19/8))
        elif features[i] == 'resistivity':
            if atom_features[a]['resistivity'] == atom_features['H']['resistivity']:
                index = 0
            elif atom_features[a]['resistivity'] == atom_features['N']['resistivity']:
                index = 1
            else:
                index = int(atom_features[a]['resistivity']/20)

        elif features[i] == 'fusion':
            if atom_features[a]['fusion'] == atom_features['H']['fusion']:
                index = 0
            elif atom_features[a]['fusion'] == atom_features['N']['fusion']:
                index = 1
            else:
                index = int((atom_features[a]['fusion']-6)/(30/8))

        elif features[i] == 'vaporization':
            if atom_features[a]['vaporization'] == atom_features['H']['vaporization']:
                index = 0
            elif atom_features[a]['vaporization'] == atom_features['N']['vaporization']:
                index = 1
            else:
                index = int((atom_features[a]['vaporization']-100)/(750/8))

        elif features[i] == 'heat capacity':
            if atom_features[a]['heat capacity'] == atom_features['H']['heat capacity']:
                index = 0
            elif atom_features[a]['heat capacity'] == atom_features['N']['heat capacity']:
                index = 1
            else:
                index = int((atom_features[a]['heat capacity']-0.12)/(0.45/8))

        elif features[i] == 'valence':
            index = int(atom_features[a]['valence'])
        index = index + c
        a_one_hot[index] = 1
        c = c + feature_list[features[i]]
    return a_one_hot

def bond_encoding(d):
    d_one_hot = np.zeros(BOND_CATEGORY_NUM, dtype=np.float32)
    if d < 2.4:
        d = 2.4
    elif d >= 4.0:
        d = 4.0
    #sigma = (4.0 - 2.4) / (BOND_CATEGORY_NUM-1)
    sigma = 0.2
    mu_0 = 2.4
    for i in range(BOND_CATEGORY_NUM):
        mu = mu_0 + i*sigma
        d_one_hot[i] = np.exp(-(d-mu)**2/sigma**2)
    return d_one_hot

if __name__ == "__main__":
    print(fnstr)
    print(fnlist)
    print(features)
    print(CATEGORY_NUM)