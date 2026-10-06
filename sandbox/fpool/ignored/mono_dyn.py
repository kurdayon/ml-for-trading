
# golden
'''
    The class "mono_dyn" finds a 1D monotonic function minimizing squared deviations.
    At instantiation of the model we need to specify: feature, target and direction (sign).
        1 means that funciton growa and -1 means that it falls.

    Dependency:

        mono_model.fit
            -> get_levels
                -> mono_dyn
                    -> full_correct
                        -> correct
                            -> w_mean

'''

import numpy as np
import pandas as pd
from math import sqrt

from linear_fill import linear_fill

def w_mean(vws):
    '''
    Takes a list of tuples: (value, weight) and return the weighted mean.
    '''
    vs = [vw[0] for vw in vws]
    ws = [vw[1] for vw in vws]
    return np.average(vs, weights = ws)


def w_opos(vws):
    vs = [vw[0] for vw in vws]
    ws = [vw[1] for vw in vws]
    m = np.average(vs, weights = ws)
    sdevs = np.power(vs - m, 2)
    s = sqrt(np.average(sdevs, weights = ws))
    return m / (pow(m,2) + pow(s,2))


def correct(chunks, level_func):
    '''
    Combine the last 2 chunks into one if the mean of the last chunk
    is not larger (smaller or equal) than the mean of the previous chunk.
    The second element of the returned tuple indicate if the correction
    (combination of chunk) was performed.
    '''
    if level_func(chunks[-1]) <= level_func(chunks[-2]):
        return chunks[:-2] + [chunks[-2] + chunks[-1]], True
    else:
        return chunks, False


def full_correct(chunks, level_func):
    ''' Correct chunks untill its needed.
    '''
    modified = True
    while modified:
        chunks, modified = correct(chunks, level_func)
        if len(chunks) == 1:
            break
    return chunks


def mono_dyn(ys, level_func):
    ''' Take a list of values and returns chunks.
    In other words, it take a list and returns a list of lists.
    '''
    chunks = [[ys[0]]]
    for i in range(1, len(ys)):
        chunks.append([ys[i]])
        chunks = full_correct(chunks, level_func)
    return chunks


def get_levels(ys, level_func):
    ''' For each y in ys generate the corresponding level.
    So, the output is the list of the same size as ys.
    '''
    chunks = mono_dyn(ys, level_func)
    out = []
    for chunk in chunks:
        m = level_func(chunk)
        out += [m for _ in chunk]
    return out


class mono_model:


    def __init__(self, feature, target, sign = 1, level_func = w_mean):
        '''
        If sign ==  1 we search a growing  function.
        If sign == -1 we search a decaying function.
        '''
        self.sign = sign
        self.feature = feature
        self.target = target
        self.level_func = level_func


    def fit(self, inp_df):
        df = inp_df.copy()
        df.dropna(inplace = True)

        # define a data-frame with the following columns: self.feature, 'm' (mean), 'w' weight
        gr_m = df.groupby(self.feature, as_index = False).mean().rename(columns = {self.target:'m'})
        gr_n = df.groupby(self.feature, as_index = False).count().rename(columns = {self.target:'w'})
        gr = pd.merge(gr_m, gr_n, how = 'inner', on = [self.feature])
        gr.loc[:,'w'] = gr.w * 1.0
        gr.sort_values(self.feature, inplace = True)

        self.xs = gr[self.feature].values
        self.ys = self.sign * gr.m.values
        self.ws = gr.w.values

        # Python 3 zip returns an iterator; mono_dyn indexes this sequence.
        self.ms = get_levels(list(zip(self.ys, self.ws)), self.level_func)

        # save the result into a data-frame to use it in predict
        self.fit_df = pd.DataFrame({self.feature : self.xs, 'p' : self.ms, 'label' : 'fit'})


    def fast_step_predict(self, inp_df):
        '''
        This function is fast but it generates predictions to "the left" from the points.
        In other words, if we change the level, it happens not beteween the points but
        immediately after the last point of the interval / level.
        '''
        df = inp_df.copy()
        xs = df[self.feature].values
        inds = np.searchsorted(self.xs, xs)
        P =  self.sign * np.array(self.ms + [self.ms[-1]])[inds]
        df.loc[:,'p'] = P
        return df


    def predict(self, inp_df):
        '''
        '''
        df = inp_df.copy()
        out_df = inp_df.copy()

        df = df[[self.feature]]
        df['label'] = 'pre'
        df['p'] = np.nan
        ind = [i for i in range(len(df))]
        df['ind'] = ind[:]

        fp_df = pd.concat([self.fit_df, df], sort = False)
        fp_df = linear_fill(fp_df, self.feature, 'p', 'p')
        fp_df = fp_df[fp_df.label == 'pre'].copy()
        fp_df.sort_values('ind', inplace = True)

        out_df['p'] = self.sign * fp_df.p.values[:]

        return out_df




if __name__ == '__main__':
    '''
    Just run this code in the command line to get a self explaining figure that demonstrate performance of the method.
    '''

    import random
    import matplotlib.pyplot as plt

    def x_to_y(x):
        if x < 8.5:
            return x + random.uniform(-2.0, 2.0)
        else:
            return x + random.uniform(-1.0, 1.0) - 5.0

    # generate sochastically growing values.
    n_points = 10
    xs1 = [i for i in range(n_points)]
    xs1 += [xs1[-1]] * 100
    random.shuffle(xs1)
    ys1 = [x_to_y(x) for x in xs1]
    df1 = pd.DataFrame({'x':xs1, 'y':ys1})

    xs2 = [random.uniform(-1.0, n_points) for i in range(100 * n_points)]
    random.shuffle(xs2)
    df2 = pd.DataFrame({'x':xs2})

    # fit & predict
    m = mono_model('x', 'y', 1, w_mean)
    m.fit(df1)
    df1 = m.fast_step_predict(df1)
    df2 = m.fast_step_predict(df2)

    df3 = m.predict(df2)

    # generate a figure
    plt.figure(figsize = (16,8))
    plt.plot(df1.x, df1.y, marker = 'o', linestyle = '', color = 'g', markersize = 20, label = 'targets')
    plt.plot(df2.x, df2.p,  marker = 'o', linestyle = '', color = 'r',                  label = 'out of sample predictions')
    plt.plot(df1.x, df1.p,  marker = 'o', linestyle = '', color = 'b', markersize = 15, label = 'predictions for targets')
    plt.plot(df3.x, df3.p,  marker = 'o', linestyle = '', color = 'y', alpha = 0.3)
    plt.legend()
    plt.show()
