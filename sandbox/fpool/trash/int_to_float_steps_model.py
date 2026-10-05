import itertools
import numpy as np
import random
import pandas as pd
pd.options.mode.chained_assignment = None
from scipy.stats import binom

class int_to_float_steps_model:


    def __init__(self, input_col, output_col, n_ranges, verbose = False):
        self.input_col = input_col
        self.output_col = output_col
        self.verbose = verbose
        self.n_ranges = n_ranges


    def fit(self, df):
        '''
        Take a data frame and find the best split for a given number of splits.
        As a result we set "self.best_splitter" and "self.best_means" fealds.
        This method is used in the "_get_scores" method.
        '''

        self.inps = df[self.input_col].unique()
        self.inps = list(self.inps)
        self.inps.sort()

        splitters = itertools.combinations(self.inps[1:], self.n_ranges - 1)

        best_score = None

        splits_count = 0
        for sub_splitter in splitters:
            splits_count += 1

            splitter = [self.inps[0]] + list(sub_splitter) + [self.inps[-1] + 1]

            # get predictions for each split
            df['preds'] = 0.0
            means = []
            for i in range(len(splitter) - 1):

                i1 = splitter[i]
                i2 = splitter[i+1]
                condition = (df[self.input_col] >= i1) & (df[self.input_col] < i2)
                mean = df[condition][self.output_col].mean()
                df['preds'] = np.where(condition, mean, df.preds)
                means.append(mean)

            score = np.nanmean(np.power(df[self.output_col] - df.preds, 2))

            label = ''
            if best_score == None or score < best_score:
                label = '***'
                best_score = score
                best_splitter = splitter[:]
                best_means = means[:]
            if self.verbose: print splitter, score, label

        self.best_splitter = best_splitter
        self.best_means = best_means[:]


    def predict(self, inp_df):
        '''
        Use the current "expertises" to generate predictions.
        '''
        df = inp_df.copy()
        n = len(self.best_splitter)
        df['preds'] = np.zeros(len(df))
        for i in range(n - 1):
            i1 = self.best_splitter[i]
            i2 = self.best_splitter[i+1]
            condition = (df[self.input_col] >= i1) & (df[self.input_col] < i2)
            df['preds'] = np.where(condition, self.best_means[i], df.preds)
        i1 = self.best_splitter[-1]
        i2 = self.best_splitter[0]
        condition = (df[self.input_col] >= i1) | (df[self.input_col] < i2)
        df['preds'] = np.where(condition, self.best_means[-1], df.preds)
        return df['preds']


if __name__ == '__main__':

    import pandas as pd
    import random
    import matplotlib.pyplot as plt

    xs = []
    ys = []
    epsilon = 3.0
    n_per_x = 20
    for i in range(n_per_x):
        xs.append(random.choice([1.0, 2.0, 3.0]))
        ys.append(20.0 + random.uniform(-epsilon, epsilon))

    for i in range(n_per_x):
        xs.append(random.choice([4.0, 5.0]))
        ys.append(10.0 + random.uniform(-epsilon, epsilon))

    for i in range(n_per_x):
        xs.append(random.choice([6.0, 7.0, 8.0]))
        ys.append(30.0 + random.uniform(-epsilon, epsilon))

    for i in range(n_per_x):
        xs.append(random.choice([9.0, 10.0, 11.0, 12.0]))
        ys.append(40.0 + random.uniform(-epsilon, epsilon))

    df = pd.DataFrame({'x':xs, 'y':ys}, columns = ['x', 'y'])

    model = int_to_float_steps_model(
        input_col = 'x',
        output_col = 'y',
        n_ranges = 4,
        verbose = True
        )

    model.fit(df)
    preds = model.predict(df)
    print 'Check the plots...'
    plt.figure()
    plt.scatter(df.x, df.y, color = 'g')
    plt.scatter(df.x, preds, color = 'r')
    plt.show()
