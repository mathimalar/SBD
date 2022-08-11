import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import SBD
import plotting
import bench
import time
from SBD import Y_factory, measurement_to_activation, RTRM, crop_to_center, recovery_error
from bench import BenchmarkInfo
from tqdm import tqdm

def job(model, defect_density, kernel_size, samples, SNR):
    errors = np.zeros(samples)
    for k, sample in enumerate(tqdm(range(samples),desc="Samples Loop",leave=False)):
        # print(i,j,k)
        Y, A, X = Y_factory(1, (256, 256), (kernel_size, kernel_size), defect_density, SNR)

        X_guess = measurement_to_activation(Y, model=model)
        kernel_size = kernel_size if type(kernel_size) is int else int(kernel_size.item(0))
        A_rand = np.random.normal(0, 1, (1, 2 * kernel_size, 2 * kernel_size))
        A_rand = A_rand / np.linalg.norm(A_rand)

        A_solved = RTRM(1e-5, X_guess, Y, A_rand)
        A_solved = crop_to_center(A_solved, (1, kernel_size, kernel_size))
        A_solved = A_solved / np.linalg.norm(A_solved)  # norm = 1

        errors[sample] = recovery_error(A_solved[0], A[0])
    return errors



bench_info = BenchmarkInfo(sample_num=20,
                            resolution=20,
                            max_defect_density=0.5,
                            min_defect_density=0.5 * (10 ** -4),
                            max_kernel_size=62,
                            min_kernel_size=8)

import argparse

# Create the parser and add arguments
parser = argparse.ArgumentParser()
parser.add_argument('--jxrs', type=int, default=-1, help="job x range start")
parser.add_argument('--jxre', type=int, default=-1, help="job x range end")
parser.add_argument('--jyrs', type=int, default=-1, help="job y range start")
parser.add_argument('--jyre', type=int, default=-1, help="job y range end")

parser.add_argument('--job_file', type=str, default="", help="string file containing jobs to run")

# Parse and print the results
args = parser.parse_args()
defect_density_range = bench_info.defect_range()#range(args.jxrs, args.jxre)#[args.jxrs:args.jxre]
kernel_size_range = bench_info.kernel_range()#[args.jyrs:args.jyre]
model='lista'
SNR=2
error_matrix = np.zeros([len(defect_density_range), len(kernel_size_range)])
start = time.time()
if args.job_file == "":

    for i in tqdm(defect_density_range, desc="Defect Density Loop"):
        defect_density = defect_density_range[i]
        # print(f'loop {i + 1} out of {len(defect_density_range)} \n')
        for j in tqdm(kernel_size_range, desc="Kernel Size Loop", leave=False):
            kernel_size = kernel_size_range[j]
            # start = time.time()
            e=job(model, defect_density,kernel_size, bench_info.sample_num, SNR)
            np.save(f"out/errors_{i}_{j}.npy",e)

        # tqdm.write(f"Job {i} {j} finished!")
else:
    import json
    job_indices = None
    with open(args.job_file) as f:
        job_indices = json.loads(f.read()) 
    # print(job_indices)
    for j in tqdm(job_indices, desc="Jobs loop"):
        tqdm.write(str(j) +" started ")
        a, b = j
        defect_density = defect_density_range[a]
        kernel_size = kernel_size_range[b]
        e=job(model, defect_density,kernel_size, bench_info.sample_num, SNR)
        np.save(f"out/errors_{a}_{b}.npy",e)
    
# print(f"time taken: {(time.time()-start)/60}")
# with Pool(processes=8) as pool:
#     pool.starmap(job, args)
# for item in args:
#     job(*item)

