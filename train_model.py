import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
from utils import load_data

X_train,X_test,y_train,y_test=load_data('../data/sample_students.csv')
lr=LogisticRegression(max_iter=1000)
dt=DecisionTreeClassifier(max_depth=5,random_state=42)
ensemble=VotingClassifier([('lr',lr),('dt',dt)])
ensemble.fit(X_train,y_train)
pred=ensemble.predict(X_test)
print('acc',accuracy_score(y_test,pred))
joblib.dump(ensemble,'model.pkl')
