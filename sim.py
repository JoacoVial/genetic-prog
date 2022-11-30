import os

MaxLenPop = [640,1280]
CxPb = [0.7,0.9]
MutPb = [0.01,0.03]
MaxDepth = [3,4,5]

for pop in MaxLenPop:
        for cx in CxPb:
            for mut in MutPb:
                for depth in MaxDepth:
                    print('------------------SIMULACION MaxLenPop: {} -- CxPb: {} -- MutPb: {} -- MaxDepth: {} -----------------'.format(pop,cx,mut,depth))
                    os.system('python3 pg.py {} {} {} {}'.format(pop,cx,mut,depth))
