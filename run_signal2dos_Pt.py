#from signal2dos_notgrid import s2d_mnstd
import numpy as np
import os

def desample(maxE,minE,N_E,input,mn):
		
	for i in range(len(input)):
		input[i] = input[i] + mn[i]
		
	interval_E = float(maxE-minE)/float(N_E)
	xcoor = np.arange(minE,maxE,interval_E)
	output=np.c_[xcoor,input]	
	return output

def write_data(dataOut1,fNameOut):
    

    f = open(fNameOut,'w')

    for i in range(len(dataOut1[:,0])):
        f.write('%20.8f %20.8f\n' %(dataOut1[i,0],dataOut1[i,1]))

def s2d_mnstd(maxE,minE,N_E,maxDOS,minDOS,N_DOS,signals,PC,mn,dirname,fname,mn2,std):
    k = np.shape(signals)[0]
    atomn = np.shape(signals)[1]
    totdos = None
    if not os.path.basename(dirname) in os.listdir(os.path.dirname(dirname)):
        os.mkdir(dirname)
    for i in range(atomn):
        DOSout1 = desample(maxE,minE,N_E,np.dot(PC[:,0:k],signals[:,i]),mn)
        DOSout1[:,1] *= std
        DOSout1[:,1] += mn2
        write_data(DOSout1,dirname+'/dosfile_%d' %i)
        if totdos is None:
            totdos = DOSout1
        else:
            totdos[:,1] += DOSout1[:,1]
    totdos[:,1] /= atomn
    write_data(totdos,fname)
    return totdos




### grid size
maxE = 3.0
minE = -8.0
N_E = 200

maxDOS = 14.0
minDOS = 0.0
N_DOS = 200

suffix = 'Pt'
datadir = 'pca_Pt'
signaldir = 'signal_Pt'
PC = np.loadtxt(datadir+'/PC.txt')
mn = np.loadtxt(datadir+'/mn.txt')
mn2= np.loadtxt(datadir+'/mn2.txt')
std= np.loadtxt(datadir+'/std.txt')


			
print(suffix)

for npn in [19,38,44,55,62,79,85,108,116,140]:
#for npn in [85,108,116]:
	signals = np.loadtxt(signaldir+'/'+'Pt%d_signal_%s.txt' % (npn,suffix))
	output = s2d_mnstd(maxE,minE,N_E,maxDOS,minDOS,N_DOS,signals,PC,mn,signaldir+'/'+'Pt%d' % npn,signaldir+'/'+'Pt%d.dos' % (npn),mn2,std)

mddir = 'MD'
md_list = []
for i in range(30,41,5):
	md_list.append('Pt%d' % i)
for npstr in md_list:

		natom = int(npstr[2:])
		
		md_name = mddir+'/'+npstr
		signals = np.loadtxt(signaldir+'/'+'%s_signal_%s.txt' % (md_name,suffix))
		output = s2d_mnstd(maxE,minE,N_E,maxDOS,minDOS,N_DOS,signals,PC,mn,signaldir+'/'+mddir+'/'+npstr+'_%d' % i,signaldir+'/'+'%s.dos' % (md_name),mn2,std)
		