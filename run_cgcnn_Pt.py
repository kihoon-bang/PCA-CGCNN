import tensorflow as tf
import numpy as np
from CrystalGraph import *
import time
import os
import math

def data_preparation(X):

	atom_number = 0
	bond_number = 0
	data_num = len(X)
    

	for i in range(data_num):
		atom_vectors, bond_vectors, bond_indices = cif_to_vector(X[i])
		n = atom_vectors.shape[0]
		m = 0
		if atom_number < n:
			atom_number = n
		for j in range(n):

			m = bond_vectors[j].shape[0]
			if bond_number < m:
				bond_number = m
    
	reshape_num = data_num*atom_number
	atoms = np.zeros([data_num, atom_number, CATEGORY_NUM], dtype=np.int32)
	bonds = np.zeros([data_num, atom_number, bond_number, NEIGHBOR_CATEGORY_NUM], dtype=np.int32)
	atom_num = np.zeros([len(X)], dtype=np.int32)
	bond_indices1 = np.full([data_num, atom_number, bond_number], reshape_num, dtype=np.int32)
	bond_indices2 = np.full([data_num, atom_number, bond_number], reshape_num, dtype=np.int32)

	for i in range(data_num):
		atom_vectors, bond_vectors,bond_indices = cif_to_vector(X[i])
		n = atom_vectors.shape[0]
		atoms[i][:n] = atom_vectors
		atom_num[i] = n

		for j in range(n):
			m = bond_vectors[j].shape[0]
			for k in range(m):
				bonds[i][j][k] = bond_vectors[j][k] 
				
				bond_indices1[i][j][k] = bond_indices[j][k][0] + i*atom_number
				bond_indices2[i][j][k] = bond_indices[j][k][1] + i*atom_number

	return atoms, bonds, atom_num, bond_indices1, bond_indices2



learning_rate = 10**-3
dropout_prob = 0.9 ## keep probablity
reg_coeff = 10**-3
epochs = 1000

logdir = 'log_Pt'
if not logdir in os.listdir('./'):
	os.mkdir(logdir)
modeldir = 'model'
pcadir = 'pca_Pt'
signaldir = 'signal_Pt'
if not signaldir in os.listdir('./'):
	os.mkdir(signaldir)
if not logdir in os.listdir('./'):
	os.mkdir(logdir)


for conv_num in [2]:
	for num_layer in [3]:
		for hidden_dim in [63]:
			for output_dim in [41]:
				suffix = 'Pt'

				
				print('\n####################################\n')
				print('model:',suffix)
				print()

				initial_time = time.time()

				cif_total = []
				index_total = []

				##
				training_list = ['19','38','44','55','55_ih','62','79','85']
				
				for npstr in training_list:
						if 'ih' in npstr:
								natom = int(npstr[0:-3])
								print('## icosahedron,',npstr,'natom:',natom)
						else:
								natom = int(npstr)
						for n in range(natom):
							cif_total.append('Pt'+npstr)
							index_total.append(n)
				
				md_list=[]
				for i in range(30,41,5):
					md_list.append('Pt%d' % i)
				
				mddir = 'MD'
				for npstr in md_list:
						natom = int(npstr[2:])
						
						#if npstr =='Pt38': ####
						#		strnum = 14
						
						md_name = mddir+'/'+npstr

						for n in range(natom):
							cif_total.append(md_name)
							index_total.append(n)

				## read signal output
				signal_all = np.loadtxt(pcadir+'/signals.txt')[:output_dim,:].transpose()

				print('## cif shape',np.shape(cif_total))
				print('## signal shape',np.shape(signal_all))
				
				signal_total = signal_all[:np.shape(cif_total)[0],:]

				signal_mean = np.mean(signal_all,axis=0)
				signal_std = np.std(signal_all,axis=0)

				signal_total -= signal_mean
				signal_total /= signal_std

				rand_int = np.arange(len(signal_total))
				np.random.shuffle(rand_int)
				cif_total = np.array(cif_total)
				cif_total = cif_total[rand_int]
				signal_total = signal_total[rand_int]
				index_total = np.array(index_total)
				index_total = index_total[rand_int]
				
				
				total_data_num = len(signal_total)
				batch_size = 32

				

				#
				training_num = int(total_data_num*0.8)
				cif_training = cif_total
				signal_training = signal_total
				index_training = index_total

				atom_training, bond_training, atom_num_training ,indices1_training, indices2_training = data_preparation(cif_training)

				training_y = signal_total
				
				cif_valid = cif_total[training_num:training_num+int(total_data_num*0.1)]
				signal_valid = signal_total[training_num:training_num+int(total_data_num*0.1)]
				index_valid = index_total[training_num:training_num+int(total_data_num*0.1)]
				Y_valid = signal_valid
				atom_valid, bond_valid, atom_num_valid, indices1_valid, indices2_valid = data_preparation(cif_valid)

				cif_test = cif_total[training_num+int(total_data_num*0.1):]
				signal_test = signal_total[training_num+int(total_data_num*0.1):]
				index_test = index_total[training_num+int(total_data_num*0.1):]
				Y_test = signal_test
				test_atom, test_bond, test_atom_num,test_indices1,test_indices2 = data_preparation(cif_test)


				tf.reset_default_graph()

				X_atoms = tf.placeholder(tf.float32, [None, None, CATEGORY_NUM])
				X_bonds = tf.placeholder(tf.float32, [None, None, None , NEIGHBOR_CATEGORY_NUM])
				X_atom_num = tf.placeholder(tf.float32, [None])
				X_indices1= tf.placeholder(tf.int32, [None,None,None])
				X_indices2= tf.placeholder(tf.int32, [None,None,None])
				y = tf.placeholder(tf.float32, [None,output_dim])
				keep_prob = tf.placeholder(tf.float32)

				X_atom_index = tf.placeholder(tf.int32,[None])

				a = tf.shape(X_bonds)[1]
				b = tf.shape(X_bonds)[2]

				def concat_atom_bond(X_atoms,X_bonds,X_atom_num,X_indices1,X_indices2):
					atoms_dummy = tf.zeros([1, CATEGORY_NUM], dtype=tf.float32)

					X_atoms_reshape = tf.reshape(X_atoms, [-1, CATEGORY_NUM])
					X_atoms_reshape = tf.concat([X_atoms_reshape, atoms_dummy], axis=0)
					X_bonds_reshape = tf.reshape(X_bonds, [-1, NEIGHBOR_CATEGORY_NUM])
					X_indices1_reshape = tf.reshape(X_indices1, [-1, ])
					X_indices2_reshape = tf.reshape(X_indices2, [-1, ])
					
					concat_1 = tf.gather(X_atoms_reshape, X_indices1_reshape)
					concat_2 = tf.gather(X_atoms_reshape, X_indices2_reshape)
					X_bonds_concat = tf.concat([concat_1,concat_2,X_bonds_reshape], axis=1)

					return X_bonds_concat

				atoms_dummy = tf.zeros([1, CATEGORY_NUM], dtype=tf.float32) 

				X_atoms_reshape = tf.reshape(X_atoms, [-1, CATEGORY_NUM])
				X_atoms_reshape = tf.concat([X_atoms_reshape, atoms_dummy], axis=0)
				X_bonds_reshape = tf.reshape(X_bonds, [-1, NEIGHBOR_CATEGORY_NUM])
				X_indices1_reshape = tf.reshape(X_indices1, [-1, ])
				X_indices2_reshape = tf.reshape(X_indices2, [-1, ])

				## convolution part

				#print(a,b)

				def cgcnn_conv(X_atoms,X_bonds_reshaped):

					Wf_1 = tf.Variable(tf.random_normal([TOTAL_CATEGORY_NUM, CATEGORY_NUM], stddev=0.1),dtype=tf.float32)
					bf_1 = tf.Variable(tf.random_normal([CATEGORY_NUM], stddev=0.1), dtype=tf.float32)
					Ws_1 = tf.Variable(tf.random_normal([TOTAL_CATEGORY_NUM, CATEGORY_NUM], stddev=0.1), dtype=tf.float32)
					bs_1 = tf.Variable(tf.random_normal([CATEGORY_NUM], stddev=0.1), dtype=tf.float32)
				 
					## convolution part

					#print(a,b)

					sig_term = tf.sigmoid(tf.matmul(X_bonds_reshaped, Wf_1) + tf.reshape(bf_1, [1, CATEGORY_NUM]))
					relu_term = tf.nn.relu(tf.matmul(X_bonds_reshaped, Ws_1) + tf.reshape(bs_1, [1, CATEGORY_NUM]))
					conv_term = tf.multiply(sig_term, relu_term)
					conv_term = tf.reshape(conv_term, [-1, a,b, CATEGORY_NUM])
					X_atoms_conv = X_atoms + tf.reduce_sum(conv_term, axis=2)
					return X_atoms_conv

				X_atoms_conv = X_atoms
				for i in range(conv_num):
					X_bonds_concat = concat_atom_bond(X_atoms_conv,X_bonds,X_atom_num,X_indices1,X_indices2)
					X_atoms_conv = cgcnn_conv(X_atoms_conv,X_bonds_concat)

				##pooling part

				range_tmp = tf.range(tf.shape(X_atom_index)[0])
				index_tmp = tf.stack([range_tmp,X_atom_index],axis=-1)

				X_atom_pool = tf.gather_nd(X_atoms_conv,index_tmp)
				X_atom_pool = tf.reshape(X_atom_pool,[-1,CATEGORY_NUM])


				###fully connected part

				###fully connected part

				global_step = tf.Variable(0, trainable=False, name='global_step')
				is_training = tf.placeholder(tf.bool, shape=())

				regularizer = tf.contrib.layers.l2_regularizer(scale=0.01)
				h = X_atom_pool
				for i in range(num_layer-1):
					h = tf.layers.dense(h, 
										units=hidden_dim, 
										use_bias=True, 
										activation=tf.nn.relu, 
										kernel_initializer=tf.contrib.layers.xavier_initializer(),
										kernel_regularizer=regularizer,
										bias_regularizer=regularizer)
					h = tf.layers.dropout(h, 
										  rate=1.0-keep_prob, 
										  training=is_training)
				h = tf.layers.dense(h, 
									units=hidden_dim, 
									use_bias=True, 
									activation=tf.nn.relu, 
									kernel_initializer=tf.contrib.layers.xavier_initializer(),
									kernel_regularizer=regularizer,
									bias_regularizer=regularizer)
				h = tf.layers.dropout(h, 
										  rate=1.0-keep_prob, 
										  training=is_training)
				y_pred = tf.layers.dense(h, 
										 units=output_dim, 
										 use_bias=True, 
										 kernel_initializer=tf.contrib.layers.xavier_initializer(),
										 kernel_regularizer=regularizer,
										 bias_regularizer=regularizer)


				#regularizer = tf.nn.l2_loss(Wl_2)

				reg_loss = tf.losses.get_regularization_loss()
				cost = tf.losses.mean_squared_error(y, y_pred) + reg_loss*reg_coeff
				optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost, global_step=global_step)

				#acc_mae = tf.metrics.mean_absolute_error(y, y_pred)
				acc_mse = tf.losses.mean_squared_error(y, y_pred)

				sess= tf.Session()
				sess.run(tf.global_variables_initializer())
				print("learning start")



				stime = time.time()
				fo = open(logdir+'/'+suffix+'.log','w')
				
				fo.write('## training\n')
				for epoch in range(epochs):
					mse_training = 0

					total_batch = math.ceil(total_data_num/batch_size)
					for i in range(total_batch):
						X_batch = cif_training[i*batch_size:(i+1)*batch_size]
						Y_batch = signal_training[i*batch_size:(i+1)*batch_size]
						index_batch = index_training[i*batch_size:(i+1)*batch_size]
						atom_batch = atom_training[i*batch_size:(i+1)*batch_size]
						bond_batch = bond_training[i*batch_size:(i+1)*batch_size]
						atom_num_batch = atom_num_training[i*batch_size:(i+1)*batch_size]
						indices1_batch = indices1_training[i*batch_size:(i+1)*batch_size]
						indices2_batch = indices2_training[i*batch_size:(i+1)*batch_size]
						
						feed_dict = {X_atoms: atom_batch, X_bonds: bond_batch, X_atom_num : atom_num_batch, \
									 X_indices1: indices1_batch, X_indices2: indices2_batch, \
									 X_atom_index: index_batch, y: Y_batch, keep_prob : dropout_prob, is_training : True}
						
						c, _,pred_training = sess.run([cost, optimizer,y_pred], feed_dict=feed_dict)
						
						mse_training += sess.run(acc_mse, feed_dict=feed_dict)
					mse_training /= total_batch

					
					X_batch = cif_valid
					Y_batch = Y_valid
					index_batch = index_valid
					atom_batch = atom_valid
					bond_batch = bond_valid
					atom_num_batch = atom_num_valid
					indices1_batch = indices1_valid
					indices2_batch = indices2_valid
					feed_dict = {X_atoms: atom_batch, X_bonds: bond_batch, X_atom_num : atom_num_batch, \
									 X_indices1: indices1_batch, X_indices2: indices2_batch, \
									 X_atom_index: index_batch, y: Y_batch, keep_prob : 1.0, is_training : False}
					
					Y_valid_p, mse_valid = sess.run([y_pred,acc_mse], feed_dict=feed_dict)
					fo.write('Epoch: %d training MSE = %s validation MSE = %s\n' % (epoch+1,mse_training,mse_valid))
					if (epoch+1) % 100 == 0:
						print('Epoch:', '%d' % (epoch+1), 'training MSE = ', '{:.5f}'.format(mse_training),'validation MSE = ','{:.5f}'.format(mse_valid))
						print('time:',time.time()-stime)
						stime = time.time()
				# accuracy test
				print('## accuracy_test')
				fo.write('## test\n')
				

				saver = tf.train.Saver()
				saver.save(sess, modeldir+'/'+suffix+'.ckpt')


				test_dict = {X_atoms: test_atom, X_bonds: test_bond, X_atom_num : test_atom_num, \
							 X_indices1: test_indices1, X_indices2: test_indices2, \
							 X_atom_index :index_test, y: Y_test, keep_prob : 1, is_training : False}
				y_pred_data = sess.run(y_pred, feed_dict=test_dict)
				mse = sess.run(acc_mse, feed_dict=test_dict)


				print('Test Accuracy : MSE = ', mse)
				fo.write('Test Accuracy: MSE = %s\n' % mse)
				total_time = time.time()-initial_time
				print('Total time:',total_time)
				fo.close()


				def write_signal(signal,fname):
					signal = signal*signal_std+signal_mean
					signal = signal.transpose()
					size = np.shape(signal)
					fo = open(signaldir+'/'+fname,'w')
					for i in range(size[0]):
						for j in range(size[1]):
							fo.write('%s ' % signal[i][j])
						fo.write('\n')
					fo.close()
				
				
				fo = open('time.txt','w')
				fo.write('np t\n')
				for n in [19,38,44,55,62,79,85,108,116,140]:
					itime = time.time()
					cif_np = []
					index_np = []
					for i in range(n):
						cif_np.append('Pt%d' % (n))
						index_np.append(i)
					cif_np = np.array(cif_np)
					index_np = np.array(index_np)
					test_atom, test_bond, test_atom_num, test_indices1,test_indices2 = data_preparation(cif_np)
					test_dict = {X_atoms: test_atom, X_bonds: test_bond, X_atom_num : test_atom_num, \
								 X_indices1: test_indices1, X_indices2: test_indices2, \
								 X_atom_index : index_np, keep_prob:1,is_training : False}
					signal_np = sess.run(y_pred, feed_dict = test_dict)

					write_signal(signal_np,'Pt%d_signal_%s.txt' % (n,suffix))
					fo.write('%s %s\n' % ('Pt%d' % n,total_time+time.time()-itime))
				fo.close()
				
				if not mddir in os.listdir(signaldir):
					os.mkdir(signaldir+'/'+mddir)
				md_list=[]
				for i in range(30,41,5):
					md_list.append('Pt%d' % i)

				for npstr in md_list:

						natom = int(npstr[2:])
						

						
						cif_MD = []
						index_MD = []
						md_name = mddir+'/'+npstr

						for n in range(natom):
							
							cif_MD.append(md_name)
							index_MD.append(n)
			
						cif_MD = np.array(cif_MD)
						index_MD = np.array(index_MD)
						test_atom, test_bond, test_atom_num, test_indices1,test_indices2 = data_preparation(cif_MD)
						test_dict = {X_atoms: test_atom, X_bonds: test_bond, X_atom_num : test_atom_num, \
									 X_indices1: test_indices1, X_indices2: test_indices2, \
									 X_atom_index : index_MD, keep_prob:1,is_training : False}
						signal_np = sess.run(y_pred, feed_dict = test_dict)

						write_signal(signal_np,'%s_signal_%s.txt' % (md_name,suffix))
							
				sess.close()