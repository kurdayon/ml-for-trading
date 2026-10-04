'''
    An efficient way to get out-of-sample performance of a model is to use "n_fold" function.
    This functio splits the data set into N folds and for each fold it "predicts" using
        an instance of a model trained on the remaining N - 1 folds.
    In this way we get out-of-sample predictions for the whole data set.

    We can use "n_fold" to evaluate many models. Then we choose the best model.
    However, these results are not representative since we chose what was the best
        and part of its good performance is beacuse of "lack" / "chance".

    To solve this problem we create this function. It works as follows:
        * As before, split the data set into N folds.
        * For each split we do the following:
            * On the remainig N - 1 folds we run the "n_fold" function to choose
              the model that performes the best (out-of-sample).
            * The chosen model is retrained on the N - 1 folds.
            * The trained model is used to predict for the considered fold.
'''
