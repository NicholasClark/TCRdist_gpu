# load libraries/packages --------------

#import cupy as mx #cuda python backend, connect to T4 runtime or other with GPU
import mlx.core as mx #use this for apple silicon
import numpy as np
#import numpy as mx #use this for CPU only
import pandas as pd
import time
import scipy
import os
#from numba import njit, prange
from multiprocessing import Pool
pool = Pool()

# helper functions --------------

# function to pad center with gaps until target length
def pad_center(seq, target_length):
   seq_length = len(seq)
   if seq_length >= target_length:
       return seq[:target_length]
   else:
       total_padding = target_length - seq_length
       first_half = seq[:seq_length // 2]
       second_half = seq[seq_length // 2:]
       return first_half + ['_'] * total_padding + second_half

# functions to load data, parameter mapping dataframe, and substitution matrix
def load_TCR_file_small():
    tst_tcr = pd.read_csv(os.path.join("data","tmp_tcr.tsv"), sep="\t")
    return(tst_tcr)

def load_TCR_file_vdjdb():
    tst_tcr = pd.read_csv(os.path.join("data", "vdjdb_for_nick.tsv"), sep = "\t")
    return(tst_tcr)

def load_TCR_file():
    tst_tcr = pd.read_parquet(os.path.join("data", "covid_pairs_filtered_selected_columns.parquet"))
    #tst_tcr = pd.read_csv("covid_pairs_sub_filter.tsv", sep="\t")
    #tst_tcr = pd.read_csv("covid_pairs_sub.tsv", sep="\t") ## un-filtered pairs -- contains V segments not in the lookup table
    return(tst_tcr)

def load_params_file():
    params_df = pd.read_csv(os.path.join("data", "params_v2.tsv"), sep="\t", header=None, names=["feature", "value"], dtype={'feature' : 'string', 'value' : 'uint8'})
    params_vec = dict(zip(params_df["feature"], params_df["value"]))
    return( params_df, params_vec )

def load_substitution_matrix():
    submat = mx.array(np.loadtxt(os.path.join("data", 'TCRdist_matrix_mega.tsv'), delimiter='\t', dtype=np.int16))
    return(submat)

#def load_files(example = "small"):
#    load_TCR_file(example = example)
#    load_params_file(example = example)
#    load_substitution_matrix()

# function to encode TCR regions (va, vb, cdr3a, cdr3b) as integers and load onto GPU
def encode_TCRs(tcr, params_vec, n_max=np.inf):
    n_max = min(n_max, tcr.shape[0])
    cdr3amat = np.array([pad_center(list(seq), 29) for seq in tcr['cdr3a'][:n_max] ])
    cdr3amatint = np.vectorize(params_vec.get)(cdr3amat)
    cdr3bmat = np.array([pad_center(list(seq), 29) for seq in tcr['cdr3b'][:n_max] ])
    cdr3bmatint = np.vectorize(params_vec.get)(cdr3bmat)
    
    cols_to_use = slice(3, -2) #truncate CDR3s
    
    encoded = np.column_stack([
        np.vectorize(params_vec.get)(tcr['va'][:n_max]),
        cdr3amatint[:,cols_to_use],
        np.vectorize(params_vec.get)(tcr['vb'][:n_max]),
        cdr3bmatint[:,cols_to_use]
    ])
    tcrs1=mx.array(encoded).astype(mx.uint8)
    #tcrs2=mx.array(encoded).astype(mx.uint8) # could be a different dataset, like a database
    return(encoded, tcrs1)


# naive TCRdist functions --------------

#### Naive no-loop TCRdist function for GPU -- returns full distance matrix
def TCRdist_no_loop(tcr1, tcr2, submat):
    if tcr2 is None:
        tcr2 = tcr1
    start_time = time.time()
    #result=mx.zeros((tcrs1.shape[0],tcrs2.shape[0]),dtype=mx.uint32)
    result = mx.sum(submat[tcr1[:, None, :], tcr2[ None,:, :]],axis=2)
    #result[row_range,:]=mx.argpartition(temp_scores_chunk,axis=1,kth=kbest)[:,0:kbest] #note that this does not guarantee elements are sorted within partition! It is also annoying that sum produce 32 bit ints here, 16 bit would be enough
    #result[row_range,:]=mx.sum(submat[tcrs1[row_range, None, :], tcrs2[ None,:, :]],axis=2)
    
    end_time = time.time()
    
    print(result)
    print(f"Time taken: {end_time - start_time:.6f} seconds") #6.7 seconds on T4 GPU, 120 seconds on T4 CPU
    ### 0.0009 seconds on local Apple Silicon GPU
    return(result)

#### Naive no-loop TCRdist function for GPU with sparse output -- either sparse csr_matrix or pandas dataframe

#### Returns only edges with TCRdist less than or equal to cutoff (default = 90)
#### Returns dataframe with 3 columns: 'row' (row_index), 'col' (column index), and 'TCRdist' (TCRdist value)
#### Can also return a sparse matrix of type scipy.sparse.csr_matrix 
def TCRdist_simple(tcr1, tcr2, submat, tcrdist_cutoff=90, ch1=0, ch2=0, output = "edge_list"):
    result = mx.sum(submat[tcr1[:, None, :], tcr2[ None,:, :]],axis=2)
    #zeros = mx.subtract(mx.equal(result, 0), mx.eye(n = result.shape[0], m = result.shape[1])) ## matrix keeping track of off-diagonal zeros
    #result = mx.add(result, mx.multiply(zeros, -1)) ## set off-diagonal zeros to -1
    less_or_equal = mx.less_equal(result, tcrdist_cutoff)
    result = result*less_or_equal
    if output in ["sparse", "both", "edge_list"]:
        score_dtype = np.int16
        #if tcrdist_cutoff <= 255:
        #    score_dtype = np.uint16
        result_sparse = scipy.sparse.csr_matrix(result, dtype = score_dtype)
    if output in ["edge_list", "both"]:
        coo_mat = result_sparse.tocoo()
        result_list = pd.DataFrame({'edge1_0index': coo_mat.row+ch1, 'edge2_0index': coo_mat.col+ch2, 'TCRdist': coo_mat.data})
        result_list = result_list[result_list['edge1_0index'] != result_list['edge2_0index']]
        #result_list.loc[result_list.TCRdist == -1, "TCRdist"] = 0
    if output == "both":
        return(result_sparse, result_list)
    elif output == "edge_list":
        return(result_list)
    elif output == "sparse":
        return(result_sparse)

# TCRdist batch functions ---------------

### TCRdist function for GPU with batching for tcr1 and tcr2 lists and sparse output

### Returns a pandas data frame of all edges with TCRdist less than cutoff (default = 90)
### Returns dataframe with 3 columns: 'row' (row_index), 'col' (column index), and 'TCRdist' (TCRdist value)
def TCRdist_batch(tcr1, tcr2, submat, tcrdist_cutoff=90, output = "edge_list", chunk_size=10000, print_chunk_size=10000, print_res = True):
    n1 = tcr1.shape[0]
    n2 = tcr2.shape[0]
    num_chunks1 = np.int64(np.ceil(n1//chunk_size))
    num_chunks2 = np.int64(np.ceil(n2//chunk_size))
    if print_res:
        print('total number of chunks (rows):', num_chunks1)
        print('total number of chunks (cols):', num_chunks2)
    start_time = time.time()
    res_list = []
    for ch in range(0, n1, chunk_size):
        if print_res:
            if ch % print_chunk_size == 0:
                print('Processing chunk (rows)', ch)
        chunk_end = min(ch + chunk_size, n1)
        row_range1 = slice(ch, chunk_end)
        tcr1_tmp = tcr1[row_range1,:]
        for ch2 in range(0, n2, chunk_size):
            chunk_end2 = min(ch2 + chunk_size, n2)
            row_range2 = slice(ch2, chunk_end2)
            tcr2_tmp = tcr2[row_range2,:]
            edges_tmp = TCRdist_simple(tcr1=tcr1_tmp, tcr2=tcr2_tmp, submat=submat, tcrdist_cutoff=tcrdist_cutoff,
                                       ch1=ch, ch2=ch2, output="edge_list")
            res_list.append(edges_tmp)
    res = pd.concat(res_list)
    end_time = time.time()
    if print_res:
        res
        print(f"Time taken: {end_time - start_time:.6f} seconds")
    return(res)

## Original function from Misha (slightly modified)

## Runs TCRdist on chunks of tcr1 list against whole tcr2 list -- runs out of memory for larger datasets
def TCRdist_orig_loop(tcrs1, tcrs2=None, chunk_size=1000, kbest=1000, ignore_same_TCR=True):
    if tcrs2 is None:
        tcrs2 = tcrs1
    #kbest=1000 # nbest neighbour we are looking for
    #chunk_size=min(tcrs1.shape[0],20000000//tcrs2.shape[0]) # decrease magic constant to limit memory consumption by temporaty 3d tensor. Increase if you have more powerful gpu
    #chunk_size = 1000
    #num_chunks = tcrs1.shape[0]/chunk_size
    num_chunks = np.int64(np.ceil(tcrs1.shape[0]/chunk_size))
    print('total number of chunks', num_chunks)
    #### Original loop
    start_time = time.time()
    #result=mx.zeros((tcrs1.shape[0],kbest),dtype=mx.uint32) #initialize result array for indices
    #result=mx.zeros((tcrs1.shape[0],tcrs2.shape[0]),dtype=mx.uint32) #initialize result array for indices
    result=mx.zeros((tcrs1.shape[0],kbest),dtype=mx.uint32)
    for ch in range(0, tcrs1.shape[0], chunk_size): #we process in chunks across tcr1 to not run out of memory
        print('Processing chunk', ch)
        chunk_end = min(ch + chunk_size, tcrs1.shape[0])
        row_range = slice(ch, chunk_end)
        #mx.sum(submat[tcrs1[row_range, None, :], tcrs2[ None,:, :]],axis=2)# if you just want TCRdist matrix for this chunk'
        temp_scores_chunk = mx.sum(submat[tcrs1[row_range, None, :], tcrs2[ None,:, :]],axis=2)
        if ignore_same_TCR:
            temp_scores_chunk = mx.where(temp_scores_chunk==0, mx.inf, temp_scores_chunk)
        result[row_range,:]=mx.argpartition(temp_scores_chunk,axis=1,kth=kbest)[:,0:kbest] #note that this does not guarantee elements are sorted within partition! It is also annoying that sum produce 32 bit ints here, 16 bit would be enough
        #result[row_range,:]=mx.sum(submat[tcrs1[row_range, None, :], tcrs2[ None,:, :]],axis=2)
    
    end_time = time.time()
    
    print(result)
    print(f"Time taken: {end_time - start_time:.6f} seconds") #6.7 seconds on T4 GPU, 120 seconds on T4 CPU
    ### 0.0009 seconds on local Apple Silicon GPU
    return(result)

### testing with multi-GPU
def TCRdist_batch_multi(tcr1, tcr2, submat, tcrdist_cutoff=90, output = "edge_list", chunk_size=10000, print_chunk_size=10000, print_res = True):
    n1 = tcr1.shape[0]
    n2 = tcr2.shape[0]
    num_chunks1 = np.int64(np.ceil(n1//chunk_size))
    num_chunks2 = np.int64(np.ceil(n2//chunk_size))
    if print_res:
        print('total number of chunks (rows):', num_chunks1)
        print('total number of chunks (cols):', num_chunks2)
    start_time = time.time()
    res_list = []
    for ch in range(0, n1, chunk_size):
        if print_res:
            if ch % print_chunk_size == 0:
                print('Processing chunk (rows)', ch)
        chunk_end = min(ch + chunk_size, n1)
        row_range1 = slice(ch, chunk_end)
        tcr1_tmp = tcr1[row_range1,:]
        for ch2 in range(0, n2, chunk_size):
            chunk_end2 = min(ch2 + chunk_size, n2)
            row_range2 = slice(ch2, chunk_end2)
            tcr2_tmp = tcr2[row_range2,:]
            edges_tmp = TCRdist_simple(tcr1=tcr1_tmp, tcr2=tcr2_tmp, submat=submat, tcrdist_cutoff=tcrdist_cutoff,
                                       ch1=ch, ch2=ch2, output="edge_list")
            res_list.append(edges_tmp)
    res = pd.concat(res_list)
    end_time = time.time()
    if print_res:
        res
        print(f"Time taken: {end_time - start_time:.6f} seconds")
    return(res)