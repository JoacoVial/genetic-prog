import random as rd
import csv
from deap import base, creator, gp, tools
from sklearn.metrics import mean_squared_error as mse
import warnings
from ParamsFijos import *
import json
import time
import operator
import os
import multiprocessing
import sys

warnings.filterwarnings('ignore')

x_train = []
y_train = []

x_test = []
y_test = []

for data in training_data:
    with open(data['name'],'r') as datos:
        reader = csv.reader(datos,delimiter=';')

        for row in reader:
            if data['train'] == 1:
                x_train.append(row[1:len(row)])
                y_train.append(row[0])

            if data['test'] == 1:
                x_test.append(row[1:len(row)])
                y_test.append(row[0])

    for i in range(0,len(y_train)):
        x_train[i] = [float(j) for j in x_train[i]]
        y_train[i] = float(y_train[i])

    for i in range(0,len(y_test)):
        x_test[i] = [float(j) for j in x_test[i]]
        y_test[i] = float(y_test[i])
    
def Evaluar(tree,pset):
    func = gp.compile(tree,pset)
    y_pred = []
    for x_true in x_train:
        y_pred.append(func(*x_true))
    error=mse([y_train],[y_pred], squared=True)
    return error,
    
def div(a,b):
    if b == 0:
        return 0
    else:
        return a/b

args = sys.argv

if len(args) >= 2:
    MaxLenPop = int(args[1])
    CxPb = float(args[2])
    MutPb = float(args[3])
    MaxDepth = int(args[4])

    directory = '../{}/DEPTH_{}/LENPOP_{}/'.format(file,MaxDepth,MaxLenPop)
    path = directory+'CxPb:{} MutPb:{}'.format(CxPb,MutPb)

if not os.path.exists(path):
    os.makedirs(path)

params = open('{}/params.txt'.format(path),'w')
params.write('sum,res,mult,div\n')
params.write('MaxGen: {}\nMaxGenHOF: {}\nMaxDepth {}\nCxPb: {}\nMutPb: {}\nMaxLenPop: {}\n'.format(MaxGen, MaxGenHOF, MaxDepth, CxPb,MutPb, MaxLenPop))
params.close()

pset = gp.PrimitiveSet("MAIN", arity = 7)
pset.addPrimitive(operator.add, arity = 2)
pset.addPrimitive(operator.sub, arity = 2)
pset.addPrimitive(operator.mul, arity = 2)
pset.addPrimitive(div, arity = 2)

toolbox = base.Toolbox()
hof = tools.HallOfFame(1)

creator.create("FitnessMin", base.Fitness, weights=(-1.0,)) #clase FitnessMin, no entiendo el peso
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin,pset=pset) #Clase Individual, el cual es un arbol con el conjunto de operandos definidos

toolbox.register("expr", gp.genFull, pset=pset, min_=MaxDepth, max_=MaxDepth) 
toolbox.register("individual", tools.initIterate,creator.Individual,toolbox.expr) #Registra la inicializacion de los individuos
toolbox.register("population",tools.initRepeat,list,toolbox.individual)
toolbox.register("select", tools.selTournament, tournsize = 2)
toolbox.register("evaluate", Evaluar, pset= pset)
toolbox.register("mate", gp.cxOnePointLeafBiased, termpb=0.1)
toolbox.register("expr_mut", gp.genFull, min_=1, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

toolbox.decorate("mate",gp.staticLimit(operator.attrgetter('height'),MaxDepth))
toolbox.decorate("mutate",gp.staticLimit(operator.attrgetter('height'),MaxDepth))

if __name__ == '__main__':
    pool = multiprocessing.Pool(processes=18)
    toolbox.register("map", pool.map)
    for id_sim in range(0,N_SIM):

        seed = os.urandom(20)
        rd.seed(seed)

        actualHOF = None
        globalHOF = []

        startTime = time.time()

        Gen = 1
        GenHOF = 0
        pop = toolbox.population(MaxLenPop)

        fitnesses = toolbox.map(toolbox.evaluate, pop)
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        hof.update(pop)
        GenHOF += 1

        actualHOF = hof[0]

        globalHOF.append(actualHOF.fitness.values[0])
        
        while True:
            Gen += 1
            LenPop = 0

            newPop = []

            while LenPop < MaxLenPop:
                if rd.random() <= 1 - CxPb or LenPop == MaxLenPop -1:
                    select = toolbox.select(individuals = pop, k = 1)
                    offspring = toolbox.map(toolbox.clone, select)
                    newPop.append(offspring[0])
                    LenPop += 1
                else:
                    select = toolbox.select(individuals = pop, k = 2)
                    offspring = toolbox.map(toolbox.clone, select)
                    ind1, ind2 = toolbox.mate(offspring[0], offspring[1])
                    if rd.random() < MutPb:
                        ind1, = toolbox.mutate(ind1)
                    newPop.append(ind1)
                    if rd.random() < MutPb:
                        ind2, = toolbox.mutate(ind2)
                    newPop.append(ind2)
                    
                    LenPop += 2

            fitnesses = toolbox.map(toolbox.evaluate, newPop)
            for ind, fit in zip(newPop, fitnesses):
                ind.fitness.values = fit
            
            hof.update(newPop)

            if hof[0] != actualHOF:
                GenHOF = 0
                actualHOF = hof[0]
            else:
                GenHOF+=1

            globalHOF.append(actualHOF.fitness.values[0])

            if Gen >=MaxGen or GenHOF >= MaxGenHOF:
                break
            else:
                pop = newPop

        tiempoEvolucion = time.time() - startTime

        final_tree = hof[0]

        func = gp.compile(final_tree,pset)

        y_pred = []

        for x in x_test:
            y_pred.append(func(*x))

        with open('{}/ID{}.txt'.format(path,id_sim),'w') as params_file:
            params = {
                'tiempo_evolucion' : tiempoEvolucion,
                'Gen' : Gen,
                'fitnessHOF' : actualHOF.fitness.values[0],
                'seed' : str(seed),
                'depth' : final_tree.height,
                'hof_mse' : mse(y_true=y_test, y_pred=y_pred,squared=True),
                'tree' : str(final_tree)
            }
            json.dump(params, params_file)
            params_file.close()

        globalHOF = []

    pool.close()
