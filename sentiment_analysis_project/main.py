import pandas as pd
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from src.model_pipeline import pipeline_function

def main(): 
    df = pd.read_csv('data/YoutubeCommentsDataSet.csv')

    # removes rows without comments 
    df = df.loc[~df.Comment.isna()]

    # removes neutral comments
    df = df.loc[~(df.Sentiment == 'neutral')]

    # transforms target to numeric
    df['Sentiment'] = df['Sentiment'].map({'positive' : 1, 'negative' : 0})

    # divides text and target
    X, y = df['Comment'], df['Sentiment']

    # splits data into train and test partitions
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # list of models
    models = {'logreg' : LogisticRegression(), 'xgboost' : XGBClassifier()}

    # initializes de prediction versus actual dataframe
    pred_vs_real = pd.DataFrame({'actual' : y_test})

    # fits models and generates predictions on test data
    pipelines_list = []
    for m in models.items():
        # trains the model
        print('Fitting model')
        pipe  = pipeline_function(m[1])
        pipe.fit(X_train, y_train)

        print('Making predictions')
        # generates predictions
        predictions = pipe.predict(X_test) 
        probabilities = pipe.predict_proba(X_test)

        pipelines_list.append(pipe)

        # updates prediction versus actual dataframe 
        pred_vs_real[m[0]] = predictions
        pred_vs_real[m[0] + '_prob'] = probabilities[:, 1]

    return pred_vs_real

if __name__ == "__main__":
    pred_vs_real = main()
