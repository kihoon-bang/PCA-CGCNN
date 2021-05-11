import sys
import time
import datetime
import numpy as np
from pca import * 
import math
import os

def load_data(fNameIn):

	data = []
	# Open the DOS file
	f = open(fNameIn, 'r')
	lineNr = 0
	flines = f.readlines()
	if flines[0][0] == '#':
		data = np.loadtxt(fNameIn)
		if np.shape(data)[1] > 3:
			for i in range(3,np.shape(data)[1]-1,2):
				data[:,1] += data[:,i]
				data[:,2] += data[:,i+1]
	else:
		data = np.loadtxt(fNameIn)

	f.close()

	return data

def sample(maxE,minE,N_E,input):
	
	interval_E = float(maxE-minE)/float(N_E)
	xcoor = np.arange(minE,maxE,interval_E)

	output = np.interp(xcoor,input[:,0],input[:,1])

	return output

### size
maxE = 3.0
minE = -8.0
N_E = 200

if __name__ == "__main__":

	dosdir = 'DOS'
	dosdir_md = 'DOS_MD'
	training_list = ['19','38','44','55','55_ih','62','79','85'] 

	arrNew = None		

	interval_E = float(maxE-minE)/float(N_E)
	xcoor = np.arange(minE,maxE,interval_E)
	print('xcoor length:',len(xcoor))

	for npstr in training_list:
                if 'ih' in npstr:
                        natom = int(npstr[0:-3])
                        print('## icosahedron,',npstr,'natom:',natom)
                else:
                        natom = int(npstr)
                for n in range(natom):
                        dos_data = load_data(dosdir+'/Pt%s/dosfile_%d' % (npstr,n))
                        dos_arr = sample(maxE,minE,N_E,dos_data)
                        if arrNew is None:
                                arrNew = np.c_[dos_arr]
                        else:
                                arrNew = np.c_[arrNew,dos_arr]
	
	md_list = list(range(30,41,5))
	
	for natom in md_list:
		npstr = 'Pt%d' % natom
		md_name = npstr
		print(md_name)		
		for n in range(natom):
			dos_data = load_data(dosdir_md+'/%s/dosfile_%d' % (md_name,n))
			dos_arr = sample(maxE,minE,N_E,dos_data)
			if arrNew is None:
				arrNew = np.c_[dos_arr]
			else:
				arrNew = np.c_[arrNew,dos_arr]

	
	print('arrNew shape:',np.shape(arrNew))	

	arr2 = arrNew.T
	mn2 = np.mean(arr2,axis=0)
	std = np.std(arr2,axis=0)
	arr2 = (arr2-mn2)/std
	arrNew = arr2.T
	
	tdir = 'pca_Pt'
	if not tdir in os.listdir('./'):
		os.mkdir(tdir)

	np.savetxt(tdir+'/mn2.txt',mn2)
	np.savetxt(tdir+'/std.txt',std)	
	
	signals, PC, V, mn = pca_fnc(arrNew)
	print(np.shape(signals),np.shape(PC),np.shape(V),np.shape(mn))
	np.savetxt(tdir+'/signals.txt',signals)
	np.savetxt(tdir+'/PC.txt',PC)
	np.savetxt(tdir+'/mn.txt',mn)
	np.savetxt(tdir+'/V.txt',V.real)

