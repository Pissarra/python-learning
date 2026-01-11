import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import seaborn as sn
import matplotlib.pyplot as plt


df = pd.read_excel('../dataset-test/DadosRegressao2.xlsx')
df.head()


df.info

X = df[['Escrita', 'Ingles','Artigos Publicados']]
y = df['Resultado']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)


logistic_regression= LogisticRegression()
logistic_regression.fit(X_train,y_train)
y_pred=logistic_regression.predict(X_test)


confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
sn.heatmap(confusion_matrix, annot=True)


print('Accuracy: ',metrics.accuracy_score(y_test, y_pred))
print('Precision: ',metrics.precision_score(y_test, y_pred))
print('Recall: ',metrics.recall_score(y_test, y_pred))
print('F-Score: ',metrics.f1_score(y_test, y_pred))
plt.show()


teste = {'Escrita': 60, 'Ingles': 6, 'Artigos Publicados': 10}
dft = pd.DataFrame(data = teste,index=[0])
print(dft)
resultado = logistic_regression.predict(dft)
print(resultado)