import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline

OUTPUT_TEMPLATE = (
    'Logistic Regression (acc): {log_acc:.3f}\n'
    'Naive Bayes (acc): {nb_acc:.3f}\n'
    'Random Forest (acc): {rf_acc:.3f}\n'
    'kNN (acc): {knn_acc:.3f}\n'
    'ANN (acc): {ann_acc:.3f}\n'
)

def main():
    data = pd.read_csv('merge_cleaned/dataset.csv')
    data['Date'] = pd.to_datetime(data['Date'])
    feature_cols = ['avg_sentiment', 'mention_count']
    
    # uses history to predict future
    # splits the data into start to middle of date range as history (train) and middle of date range to end as future (test)
    # reference: https://apxml.com/courses/time-series-analysis-forecasting/chapter-6-model-evaluation-selection/train-test-split-time-series
    train, test = [], []
    for ticker in data['ticker'].unique():
        d = data[data['ticker'] == ticker].sort_values('Date')
        split = int(len(d) * 0.5)
        train.append(d.iloc[:split])
        test.append(d.iloc[split:])
    train = pd.concat(train)
    test = pd.concat(test)

    X_train = train[feature_cols]
    y_train = train['high_vol']
    X_test = test[feature_cols]
    y_test = test['high_vol']

    # training models
    # reference: https://ggbaker.ca/data-science/content/ml.html
    # reference: https://medium.com/@ssadullah.celik/comparing-machine-learning-algorithms-in-python-logistic-regression-svm-knn-neural-networks-6aa6a551ab30
    logistic = make_pipeline(StandardScaler(), LogisticRegression())
    nb = make_pipeline(StandardScaler(), GaussianNB())
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    knn = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
    ann = make_pipeline(StandardScaler(), MLPClassifier(max_iter=1000))

    # fit
    for m in (logistic, nb, rf, knn, ann):
        m.fit(X_train, y_train)

    # accuracy of each model
    log_acc = logistic.score(X_test, y_test)
    nb_acc = nb.score(X_test, y_test)
    rf_acc = rf.score(X_test, y_test)
    knn_acc = knn.score(X_test, y_test)
    ann_acc = ann.score(X_test, y_test)

    print(OUTPUT_TEMPLATE.format(
        log_acc=log_acc, nb_acc=nb_acc, rf_acc=rf_acc, knn_acc=knn_acc, ann_acc=ann_acc
    ))

    # random forest was seen to be the best model
    rf_pred = rf.predict(X_test)
    out = test[['Date', 'ticker', 'high_vol']].copy()
    out['predicted'] = rf_pred
    out.to_csv('stats/predictions.csv', index=False)

if __name__ == '__main__':
    main()
