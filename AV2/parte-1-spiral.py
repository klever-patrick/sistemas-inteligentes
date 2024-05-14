import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def sign(u):
    if u >= 0:
        return 1
    else:
        return -1


Data = np.loadtxt("spiral.csv", delimiter=",")
X = Data[:, 0:2]
N, p = X.shape
y = Data[:, 2].reshape(N, 1)

# GRAFICO DE ESPALHAMENTO
red_points = X[Data[:, 2] == 1]
blue_points = X[Data[:, 2] == -1]
plt.scatter(red_points[:, 0], red_points[:, 1], color="cyan", edgecolors="k")
plt.scatter(blue_points[:, 0], blue_points[:, 1], color="magenta", edgecolors="k")
plt.show()

# NOVA DIMENSÃO DE X
X = X.T
X = np.concatenate((-np.ones((1, N)), X))

# PASSO DE APRENDIZAGEM
LR = 0.001
# EMBARALHAMENTO
Rodadas = 0
AcuraciaGeral = []
SensibilidadeGeral = []
EspecificidadeGeral = []
Y_rodadas_tudo = []
Y_teste_tudo = []
while Rodadas < 10:
    seed = np.random.permutation(N)
    X_random = X[:, seed]
    y_random = y[seed, :]
    # Divida X e Y em (X_treino,Y_treino) e (X_teste,Y_teste)
    X_treino = X_random[:, 0 : int(N * 0.8)]
    y_treino = y_random[0 : int(N * 0.8), :]

    X_teste = X_random[:, int(N * 0.8) :]
    y_teste = y_random[int(N * 0.8) :, :]

    y_teste_mapped = [0 if valor == -1 else 1 for valor in y_teste]
    Y_teste_tudo.append(y_teste_mapped)
    # TREINAMENTO
    epoch = 0
    w = np.zeros((p + 1, 1))
    erro = True
    R, v = X_treino.shape
    while erro == True and epoch < 100:
        erro = False
        for t in range(v):
            x_t = X_treino[:, t].reshape((p + 1, 1))
            u_t = (w.T @ x_t)[0, 0]
            y_t = sign(u_t)
            d_t = y_treino[t, 0]
            e_t = int(d_t - y_t)
            w = w + (e_t * x_t * LR) / 2
            if y_t != d_t:
                erro = True
        epoch += 1
    # CABOU TREINO
    bp = 1
    # TESTE
    L, j = X_teste.shape
    acertos = 0
    contadorVP = 0
    contadorVN = 0
    contadorFP = 0
    contadorFN = 0
    Y_rodadas = []
    for t in range(j):
        u = w.T @ X_teste[:, t]
        y_t_teste = sign(u)
        d_t_teste = y_teste[t, 0]
        if y_t_teste == -1:
            y_t_teste_mapped = 0
            Y_rodadas.append(y_t_teste_mapped)
        else:
            y_t_teste_mapped = 1
            Y_rodadas.append(y_t_teste_mapped)

        if y_t_teste == d_t_teste:
            acertos += 1
            if y_t_teste == 1:
                contadorVP += 1
            else:
                contadorVN += 1
        else:
            if d_t_teste == 1:
                contadorFP += 1
            else:
                contadorFN += 1
    Y_rodadas_tudo.append(Y_rodadas)

    bp1 = 1
    Rodadas += 1
    acuracia = (acertos / j) * 100
    AcuraciaGeral.append(acuracia)
    Sensibilidade = (contadorVP / (contadorVP + contadorFN)) * 100
    SensibilidadeGeral.append(Sensibilidade)
    Especificidade = (contadorVN / (contadorVN + contadorFP)) * 100
    EspecificidadeGeral.append(Especificidade)

bp3 = 0
# ACURACIA
MediaAcuracia = np.mean(AcuraciaGeral)
DesvioPadraoAcuracia = np.std(AcuraciaGeral)
menorAcuracia = np.min(AcuraciaGeral)
maiorAcuracia = np.max(AcuraciaGeral)
# SENSIBILIDADE
SensibilidadeMedia = np.mean(SensibilidadeGeral)
DesvioPadraoSensibilidade = np.std(SensibilidadeGeral)
menorSensibilidade = np.min(SensibilidadeGeral)
maiorSensibilidade = np.max(SensibilidadeGeral)
# ESPECIFIDADE
EspecifidadeMedia = np.mean(EspecificidadeGeral)
DesvioPadraoEspecifidade = np.std(EspecificidadeGeral)
menorEspecifidade = np.min(EspecificidadeGeral)
maiorEspecifidade = np.max(EspecificidadeGeral)


print(
    "MediaAcuracia ",
    MediaAcuracia,
    " DesvioPadrao ",
    DesvioPadraoAcuracia,
    " Menor acuracia ",
    menorAcuracia,
    " MaiorAcuracia ",
    maiorAcuracia,
)
print(
    "MediaSensibilidade ",
    SensibilidadeMedia,
    " DesvioPadraoSensibilidade ",
    DesvioPadraoSensibilidade,
    " Menor Sensibilidade ",
    menorSensibilidade,
    " Maior Sensibilidade ",
    maiorSensibilidade,
)
print(
    "MediaEspecifidade ",
    EspecifidadeMedia,
    " DesvioPadraoEspecifidade ",
    DesvioPadraoEspecifidade,
    " Menor Especifidade ",
    menorEspecifidade,
    " Maior Especifidade ",
    maiorEspecifidade,
)

melhor_acuracia_idx = np.argmax(AcuraciaGeral)
pior_acuracia_idx = np.argmin(AcuraciaGeral)


# Matriz de Confusão para a melhor acurácia
melhor_acuracia_confusion = confusion_matrix(
    Y_teste_tudo[melhor_acuracia_idx], Y_rodadas_tudo[melhor_acuracia_idx]
)
sns.heatmap(melhor_acuracia_confusion, annot=True, fmt="d")
plt.title("Matriz de Confusão - Melhor Acurácia")
plt.xlabel("Predito")
plt.ylabel("Real")
plt.show()

# Matriz de Confusão para a pior acurácia
pior_acuracia_confusion = confusion_matrix(
    Y_teste_tudo[pior_acuracia_idx], Y_rodadas_tudo[pior_acuracia_idx]
)
sns.heatmap(pior_acuracia_confusion, annot=True, fmt="d")
plt.title("Matriz de Confusão - Pior Acurácia")
plt.xlabel("Predito")
plt.ylabel("Real")
plt.show()

x_linha = np.linspace(np.min(X[1, :]), np.max(X[1, :]), 100)
y_linha = -(w[0] + w[1] * x_linha) / w[2]

red_points = X[Data[2, :] == 1]
blue_points = X[Data[2, :] == -1]
plt.scatter(red_points[:, 0], red_points[:, 1], color="cyan", edgecolors="k")
plt.scatter(blue_points[:, 0], blue_points[:, 1], color="magenta", edgecolors="k")
plt.plot(x_linha, y_linha, color="red", label="Linha de Decisão")
plt.show()
