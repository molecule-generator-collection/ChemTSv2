import importlib
from importlib import import_module
import pickle
import sys
import os

sys.path.append('./')

def calc():
    
    calc_obj = sys.argv[1]
    print(os.getcwd(), calc_obj)
    
    with open(calc_obj, mode='rb') as f:
        mol, conf = pickle.load(f)
    print(conf)
    
    rs = conf['reward_setting']
    reward_calculator = getattr(import_module(rs["reward_module"]), rs["reward_class"])
    
    print(reward_calculator)
    values = [f(mol) for f in reward_calculator.get_objective_functions(conf)]
    print('values', values)
    
    gid = conf['gid']
    result_dir = os.path.join(conf['output_dir'], 'gaussian_result')
    calc_dir = f'InputMolopt{gid}'
    
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    if not os.path.exists(os.path.join(result_dir, calc_dir)):
        os.mkdir(os.path.join(result_dir, calc_dir))
    
    with open(os.path.join(result_dir, calc_dir, 'values.pickle'), mode='wb') as f:
        pickle.dump(values, f)
    os.remove(calc_obj)

if __name__ == "__main__":
    calc()
