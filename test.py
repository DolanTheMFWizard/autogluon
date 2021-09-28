from autogluon.tabular import TabularDataset, TabularPredictor

train_data_og = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv')
subsample_size = 5  # subsample subset of data for faster demo, try setting this to much larger values
train_data = train_data_og.sample(n=subsample_size, random_state=0)
labeled_pseudo_data = train_data_og.sample(n=subsample_size, random_state=1)
train_data.head()
label = 'class'
print("Summary of class variable: \n", train_data[label].describe())
predictor = TabularPredictor(label=label).fit(train_data, tuning_data=labeled_pseudo_data)
predictor.pseudo_label_fit(labeled_pseudo_data)
