from sklearn import linear_model
from pandas import read_csv
from os import path as ospath, listdir
from sklearn.model_selection import cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from numpy import array
import matplotlib.pyplot as plt


class Lasso:

    def __init__(self, dt_path):  # Cria as variáveis da classe
        """
        dt_path: STR
            caminho do dataset em questão
        """
        dict_datasets = {r'work_01\docs\50_Startups.csv': 0, r'work_01\docs\qsar_fish_toxicity.csv': 1,
                         r'work_01\docs\Salary.csv': 2}
        self.dataset = read_csv(dt_path, sep=';')  # Recebe o dataset da planilha em csv (;)
        self.X = self.dataset.iloc[:, :-1].values
        self.i = dict_datasets[dt_path]
        if 'Startups' in dt_path:  # Se for o benchmark das startups, codifica o input em string
            ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
            self.X = array(ct.fit_transform(self.X))

        self.Y = self.dataset.iloc[:, -1].values
        self.X_train = self.X[0:int(0.7*len(self.dataset))]  # Recebe 70% dos dados de entrada para treino
        self.Y_train = self.Y[0:int(0.7*len(self.dataset))]  # Recebe 70% dos dados de saída para treino

        self.clf = linear_model.LassoCV(cv=5)  # Cria o classificador para gerar o modelo

        self.model = self.clf.fit(self.X_train, self.Y_train)  # Gera o modelo a partir dos dados de treino

    def five_fold_val(self):
        # Retorna a validação com cinco testes para o dataset de treino (70% dos dados)
        self.plot_mdl()  # Plota os gráficos
        return cross_val_score(self.clf, self.X_train, self.Y_train, cv=5, scoring='r2')

    def plot_mdl(self):
        plt.figure(self.i)
        plt.semilogx(self.model.alphas_, self.model.mse_path_, ':')
        plt.plot(self.model.alphas_, self.model.mse_path_.mean(axis=-1), 'k', label='Average across the folds')
        plt.axvline(self.model.alpha_, linestyle='--', color='k', label='alpha: CV estimate')
        plt.legend()
        plt.show()


def run_lasso():
    path = r'work_01\docs'
    dict_lasso = dict()
    for dataset in listdir(path):  # Varre os datasets do caminho
        if '.csv' in dataset:
            dict_lasso[dataset] = Lasso(ospath.join(path, dataset)).five_fold_val()

    return dict_lasso
