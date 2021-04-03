from sklearn import linear_model
from pandas import read_csv, DataFrame
from os import path as ospath, getcwd, listdir


class Lasso:

    def __init__(self, dt_path):  # Cria as variáveis da classe
        """
        dt_path: STR
            caminho do dataset
        """
        cols = ['I0', 'I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'O']  # Cria as colunas do dataset
        # Recebe o dataset da planilha em csv (;)
        dataset = read_csv(dt_path, sep=';', names=cols)
        self.X = dataset[cols[0:6]]  # Cria a matrix de input
        self.Y = dataset[cols[7]]  # Cria o vetor de output
        self.X_train = self.X[0:int(0.7*len(dataset))]  # Recebe 70% dos dados de entrada para treino
        self.Y_train = self.Y[0:int(0.7*len(dataset))]  # Recebe 70% dos dados de saída para treino

        clf = linear_model.Lasso(alpha=0.1)  # Cria o classificador para gerar o modelo

        self.model = clf.fit(self.X_train, self.Y_train)  # Gera o modelo a partir dos dados de treino
        result = self.model.predict(dataset[cols[0:5]][int(0.7*len(dataset)):len(dataset)])


if __name__ == '__main__':
    path = r'work_01\docs'
    for dataset in listdir(path):  # Varre os datasets do caminho
        Lasso(ospath.join(path, dataset))
