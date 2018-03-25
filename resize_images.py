#paths 
org_path = 'raw-files/data'
new_path = 'data-299'

#Import relevant modules
import numpy as np 
from scipy.misc import imread, imresize, imsave 
from sklearn.datasets import load_files
from time import time 
from multiprocessing import Pool as ThreadPool


#Read files paths into a numpy array & clean 
all_paths = load_files(org_path, load_content=False)['filenames']
f = np.vectorize(lambda x: x[-1] in ['g','G'])
all_paths = all_paths[f(all_paths)]


def partition(paths,thread_id):
	size = len(paths)
	#reduce function 
	def read_resize_save(img_path):
		img = imread(img_path)
		img = imresize(img, (299,299,3))
		imsave(img_path.replace(org_path,new_path), img)

	#Mapping 
	t0 = time()
	for i in range(len(paths)): 
		read_resize_save(paths[i])
		if i%10==0:
			print ('Thread#%i: Completed %i/%i Image Transformations.		Time Elapsed: %0.1f Seconds'%(thread_id,i,size, (time()-t0)))
	print ('Thread#%i: Completed %i/%i Image Transformations.		Time Elapsed: %0.1f Seconds'%(thread_id,i,size, (time()-t0)))


#Threading
n = 7
paths_list = np.array_split(all_paths, n)
ids = [i for i in range(n)]
pool = ThreadPool(n)
pool.starmap(partition,zip(paths_list,ids))


