import numpy as np
import pysindy as ps
import csv
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as mse
import json
import time

def Transformar(string):
    func_split = string.split(' + ')
    terms=[]
    for i in func_split:
        terms.append(i.split(' '))
    for i in terms:
        for id,j in enumerate(i):
            if '^' in j:
                aux = j.split('^')
                i[id] = 'pow({},{})'.format(aux[0],aux[1])

    func = ''
    for id_term,term in enumerate(terms):
        for id_var,var in enumerate(term):
            if id_var == len(term)-1:
                func += var
            else:
                func += var+'*'
        if id_term != len(terms)-1:
            func += '+'

    funcion = lambda x0,x1,x2,x3,x4,x5,x6,x7: eval(func)

    return funcion

def CalcularV(datos,func):
    sol = []
    x_next=datos[0][0]
    sol.append(x_next)
    
    for i,row in enumerate(datos):
        if i != len(datos)-1:
            h = func(row[0],row[1],row[2],row[3],row[4],row[5],row[6],row[7])
            x_next = x_next + h
            sol.append(x_next)
    return sol

training_data = [
                    {'name': '../data/gsk80.csv', 'test':1, 'train':1},
                    {'name': '../data/gsk85.csv', 'test':1, 'train':1},
                    {'name': '../data/gsk90.csv', 'test':1, 'train':1},
                    {'name': '../data/gsk95.csv', 'test':1, 'train':1},
                    {'name': '../data/gsk100.csv', 'test':1, 'train':1}
]

path = './resultados'

treshold_scan = [0.000000022, 0.000000044, 0.000000037, 0.000000025, 0.000000023]
alphas = [0.23, 0.18, 0.27, 0.16, 0.29]
feature_library=ps.PolynomialLibrary(degree=3)
differentiation_method=ps.SmoothedFiniteDifference()

for data,treshold,alpha in zip(training_data,treshold_scan,alphas):

    x_train = []
    y_train = []
    x_test = []
    y_test = []
    archivo = data['name'][8:13]
    error = []
    min_error = 1000
    best_model = None
    best_alpha = 0
    best_treshold = 0
    best_Vdot = None

    with open(data['name'],'r') as datos:
        reader = csv.reader(datos,delimiter=';')

        for row in reader:
            if data['train'] == 1:
                x_train.append(row[0:len(row)])
                y_train.append(row[0])

            if data['test'] == 1:
                x_test.append(row[0:len(row)])
                y_test.append(row[0])

    for i in range(0,len(y_train)):
        x_train[i] = [float(j) for j in x_train[i]]
        y_train[i] = float(y_train[i])

    for i in range(0,len(y_train)):
        x_train[i] = [float(j) for j in x_train[i]]
        y_train[i] = float(y_train[i])

    for i in range(0,len(y_test)):
        x_test[i] = [float(j) for j in x_test[i]]
        y_test[i] = float(y_test[i])

    t = np.arange(start=0,stop=4000, step=1)
    t = t[0:len(x_train)]

    start_time = time.time()
    model = ps.SINDy(
        differentiation_method=differentiation_method,
        optimizer=ps.STLSQ(threshold=treshold, alpha=alpha), 
        feature_library=feature_library
    )

    model.fit(np.array(x_train),t=t)

    tiempo_ejecucion = time.time() - start_time

    Vdot = model.equations()
    Vdot = Vdot[0]

    sim = CalcularV(x_test,Transformar(Vdot))

    error_sim = mse(y_true=y_test,y_pred=sim,squared=True)
    if error_sim < min_error:
        min_error = error_sim
        best_model = model
        best_alpha = alpha 
        best_Vdot = Vdot
        best_treshold = treshold

    error.append(mse(y_test,sim,squared=True))

    plt.plot(sim, label='sim')
    plt.xlabel('Secuencia (i)')
    plt.ylabel('Voltaje (mV)')
    plt.savefig(archivo+'.png')
    plt.clf()