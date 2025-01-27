from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score 
import joblib 
import pandas as pd 
import os 


def xgbc_train():
    df = pd.read_csv("./data/Iris.csv")
    
    if os.path.exists("./data_update/Iris_new.csv"):
        df_new = pd.read_csv("./data_update/Iris_new.csv")
        df = pd.concat([df, df_new], axis=0)
        print("%%%%% in new data %%%%%")

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    y = y.astype("category").cat.codes

    X_train,X_test,y_train,y_test = train_test_split(X, y, train_size=0.1, random_state=42)

    clf_model = XGBClassifier()
    clf_model.fit(X_train,y_train)

    pred = clf_model.predict(X_test)

    accu_model = accuracy_score(y_test, pred)
    accu_model = round(accu_model,2)

    if accu_model >= 0.85:
        final_clf_model = XGBClassifier()
        final_clf_model.fit(X,y)
        print(X.columns)
        
        with open("./model/xgbmodel.pkl", "wb") as var:
            joblib.dump(final_clf_model,var)
        
        return f"New Model is selected with accuracy {accu_model}"
        
    else:
        return f"old model is better, as accuarcy is lower than 85 i.e. {accu_model}"
    
if __name__ == "__main__":
    print(xgbc_train())
    
    

