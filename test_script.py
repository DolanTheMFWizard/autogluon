import os
import uuid
import numpy as np

reuse_list = [True, False]
threshold_list = np.round(list(np.linspace(0.5, 1, 5, endpoint=False)), 2)

for t in threshold_list:
    for r in reuse_list:
        id = str(uuid.uuid4())[:8]
        command = f'python test.py --id={id} --threshold={t} --reuse={r}'
        os.system(command)