'''
    Here we test the formula for optimal position and check what should be for samples containing two elements.
'''

import numpy as np

from find_min import find_min


class neg_sharpe:


    def __init__(self, samples):
        self.samples = samples


    def cost(self, params):
        if len(params) != len(self.samples):
            print 'Wrong number of parameters.'
            exit()

        n = len(params)
        profits = []
        for i in range(n):
            position = params[i]
            sample = self.samples[i]
            profits += [position * change for change in sample]

        m = np.mean(profits)
        s = np.std(profits)

        if m > 0.0 and s == 0.0:
            return -np.inf
        elif m < 0.0 and s == 0.0:
            return np.inf
        elif m == 0.0 and s == 0.0:
            return 0.0
        else:
            return -1.0 * m / s




    def get_opos(self):
        opos = []
        for sample in self.samples:
            m = np.mean(sample)
            s = np.std(sample)
            p = m / (pow(m,2) + pow(s,2))
            opos.append(p)
        return opos



if __name__ == '__main__':

    import random
    import pandas as pd

    #case = 'test_formula'
    case = 'test_monotono'

    if case == 'test_formula':

        samples = [[3.0], [-5.0, 2.0], [-2.0], [10.0, 8.0]]
        ns = neg_sharpe(samples)
        opos = ns.get_opos()
        print opos

        o = find_min(ns.cost, [0.0, 0.0, 0.0, 0.0], verbose = False)
        opos2 = o['params']

        print np.array(opos2) / np.array(opos)

        print ns.cost(opos)
        print ns.cost(opos2)
        print o['score']


    if case == 'test_monotono':
        '''
        * Generate random changes.
        * Find presumably optimal monotonic function using the "mono_dyn" approach.
        * Try to find somethng better using a random shooting.
        '''

        n_points = 5

        for exp_ind in range(100):
            xs = [random.uniform(-1.0, 1.0) for i in range(n_points)]
            xs.sort()
            ys = [random.uniform(-1.0, 1.0) for i in range(n_points)]

            df = pd.DataFrame({'x':xs, 'y':ys})

            from mono_dyn import *

            m = mono_model('x', 'y', -1, w_opos)
            m.fit(df)
            df = m.predict(df)
            #print df.sort_values('x')

            profits = df.y * df.p
            ref_s = np.mean(profits) / np.std(profits)

            best_s = None
            for i in range(100000):

                p = [random.uniform(-1.0, 1.0) for k in range(n_points)]
                p.sort()
                p.reverse()

                profits = df.y * p

                s = np.mean(profits) / np.std(profits)

                if best_s == None or s > best_s:
                    best_s = s
                    #print i, best_s, p
                    best_params = p[:]

            label = ':)'
            if best_s >= ref_s:
                label = '?'*10

                #from IPython import embed
                #embed()

            ratio = best_s / ref_s
            print 'Reference:', ref_s, '\tBest found:', best_s, '\tPerc:', ratio, '\t\t', label
