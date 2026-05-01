import numpy as np
import os, sys
#import qlgates

#folder_path1 = os.path.abspath('../')
#sys.path.append(folder_path1)
from qlgates.utils import getRzzgate   # adjust import
#from utils import getRzzgate   # adjust import

def test_rzz_shape():
    n = 2
    U = getRzzgate(n, NQL=2, theta1=0.5)
    assert U.shape == (4*n*n, 4*n*n)

def test_rzz_theta_zero_identity():
    n = 2
    U = getRzzgate(n, NQL=2, theta1=0.0)
    I = np.eye(4*n*n)
    assert np.allclose(U, I)
