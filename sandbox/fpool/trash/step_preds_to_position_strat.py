from int_to_float_steps_model import int_to_float_steps_model

class step_preds_to_position_strat:
    '''
    '''

    def __init__(self, input_col, output_col, n_ranges, trans_func):
        self.pred_model = int_to_float_steps_model(input_col, output_col, n_ranges)
        self.trans_func = trans_func

    def fit(self, df):
        self.pred_model.fit(df)

    def predict(self, df):
        preds = self.pred_model.predict(df)
        return self.trans_func(preds)
