import numpy as np
from model import Model
from solver import Solver
from utils import *

RANDOM_SEED = 1
INPUT_DIM = 3
HIDDEN_DIM = 2
if __name__ == '__main__':
    np.random.seed(RANDOM_SEED)
    model = Model(INPUT_DIM, HIDDEN_DIM)
    print("Real model:")
    print(model)
    
    solver = Solver(model)
    steal_model = solver.solve()
    print("Steal model:")
    print(steal_model)
    
    # Validate final results
    DATA_NUM = 1000
    random_input = solver._input_generator(DATA_NUM)
    error = 0.0
    for x in random_input:
        error += abs(model(x) - steal_model(x))
    error /= DATA_NUM
    
    print(f"Mean abs error of {len(random_input)} random generated data: {error}\n")