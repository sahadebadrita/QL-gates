import numpy as np
import os
import matplotlib.pyplot as plt

def save_Rg_npz(Rg, t, outdir):
    os.makedirs(outdir, exist_ok=True)

    fname = f"Rg_t{t:06d}.npz"   # time-stamped filename
    path = os.path.join(outdir, fname)

    np.savez_compressed(path, Rg=Rg)


def count_edges_per_block(A, num_blocks):
    """
    A : (4n^2 x 4n^2) adjacency matrix
    n : original parameter (block size is n^2)

    Returns:
        4x4 matrix where entry (i,j) is the number of nonzero
        entries in block (i,j)
    """
    T = A.shape[0]
    block_size = int(T/num_blocks)
    print(block_size)
    counts = np.zeros((num_blocks, num_blocks), dtype=int)

    for i in range(num_blocks):
        for j in range(num_blocks):
            block = A[
                i*block_size:(i+1)*block_size,
                j*block_size:(j+1)*block_size
            ]
            counts[i, j] = (np.count_nonzero(np.abs(block) > 1e-12))

    return counts

def visadjmat(A, num_blocks, fname):
    T = A.shape[0]
    block_size = int(T/num_blocks)
    plt.imshow(A, cmap="Greys")
    plt.colorbar()
    #plt.spy(A, markersize=1,cmaps='Greys')
    for k in range(num_blocks):
        plt.axhline((k+1)*block_size, color='red')
        plt.axvline((k+1)*block_size, color='red')
    plt.savefig(f'{fname}.png',dpi=900)
    plt.close()

def check_complex(A,num_blocks, tol=1e-12):
    T = A.shape[0]
    block_size = int(T/num_blocks)
    complex_blocks = np.zeros((num_blocks, num_blocks), dtype=bool)

    for i in range(num_blocks):
        for j in range(num_blocks):

            block = A[
                i*block_size:(i+1)*block_size,
                j*block_size:(j+1)*block_size
            ]

            complex_blocks[i, j] = np.any(np.abs(np.imag(block)) > tol)

    return complex_blocks


