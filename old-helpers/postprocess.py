#!/usr/bin/env python3
'''
Semi-automate postporocessing of last cycle metrics.
'''

import os

cmd = '''
python compute_last_cycle_metrics_from_vtu.py \
      --folder {} \
      --dt 0.0025 \
      --T 0.8 \
      --pattern "result_*.vtu" \
      --out {}
'''

RECOMPUTE = False

home_path = '/home/group4/Challenge3/simulation_db'
data_path = '/data/simulation_db'

# first round
'''
# folders in ~/Challenge3/simulation_db/
pz_home = [19, 32, 36, 76, 85, 88, 89, 92, 94, 95, 96, 98]

# folders in /data/simulation_db/
pz_data = [21, 31, 59, 80, 105]
'''

# second round
# folders in ~/Challenge3/simulation_db/
pz_home = [21, 25, 26, 29, 53, 54, 97, 101, 104, 107, 112]

# folders in /data/simulation_db/
pz_data = []

corrects = []
errors = []

for pz in pz_home:
    inpath = f'{home_path}/pz{pz:03d}/Simulations/pz{pz:03d}/72-procs'
    outpath = f'{home_path}/pz{pz:03d}/Simulations/pz{pz:03d}/72-procs/last_cycle_metrics.vtp'
    
    if not os.path.isdir(inpath):
        print('ERROR: Dir does not exist: ' + inpath)
        errors.append(pz)
        continue
    
    if RECOMPUTE or not os.path.isfile(outpath):
        print(f'Starting to compute last cycle metrics for pz{pz:03d}...')
        res = os.system(cmd.format(inpath, outpath))
        if res != 0:
            print(f'ERROR: Failed to compute last cycle metrics for pz{pz:03d}!')
            errors.append(pz)
        else:
            print(f'\tComputed last cycle metrics for pz{pz:03d}')
            corrects.append(pz)
    else:
        print(f'Skipping pz{pz:03d}.')



for pz in pz_data:
    inpath = f'{data_path}/pz{pz:03d}/Simulations/pz{pz:03d}/72-procs'
    outpath = f'{data_path}/pz{pz:03d}/Simulations/pz{pz:03d}/72-procs/last_cycle_metrics.vtp'
    
    if not os.path.isdir(inpath):
        print('ERROR: Dir does not exist: ' + inpath)
        errors.append(pz)
        continue
    
    if RECOMPUTE or not os.path.isfile(outpath):
        print(f'Starting to compute last cycle metrics for pz{pz:03d}...')
        res = os.system(cmd.format(inpath, outpath))
        if res != 0:
            print(f'ERROR: Failed to compute last cycle metrics for pz{pz:03d}!')
            errors.append(pz)
        else:
            print(f'\tComputed last cycle metrics for pz{pz:03d}')
            corrects.append(pz)
    else:
        print(f'Skipping pz{pz:03d}.')


print('Finished computing last cycle metrics for all cases.')
print('Successfully computed last cycle metrics for the following cases:\n\t' + ', '.join([f'pz{pz:03d}' for pz in corrects]))
if errors:
    print('Errors occurred for the following cases: ' + ', '.join([f'pz{pz:03d}' for pz in errors]))

'''
python compute_last_cycle_metrics_from_vtu.py \
      --folder pz001/Simulations/pz001/72-procs \
      --dt 0.0025 \
      --T 0.8 \
      --pattern "result_*.vtu" \
      --out test_last_cycle_metrics.vtp
'''