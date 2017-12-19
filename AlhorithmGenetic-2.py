#import ClassGenetAlg as genAlg
import random
import copy
from operator import attrgetter
import math
from six.moves import range

def read(file, lst):
    f=open(file)    
    for row in f:
        lst.append(row.replace('\n','').split(' '))
    f.close()
dataList=[]
read('./27data.txt',dataList)

massGlob=float(dataList[0][0])
vGlob=float(dataList[0][1])
dataSet=dataList[1:]
print('Mass: '+str(massGlob))
print('V: '+str(vGlob))
print('DATA:')
for i in range(0,len(dataSet)):
    print(str(i)+': '+'; '.join(map(str, dataSet[i])))
class GeneticAlgorithm(object):
    """Genetic Algorithm class."""
    def __init__(self,
                 seed_data,
                 seed_massGlob,
                 seed_vGlob,
                 population_size=60,
                 generations=20,
                 crossover_probability=1,
                 mutation_probability=0.20,
                 maximise_fitness=True):
        self.seed_data = seed_data
        self.population_size = population_size
        self.generations = generations
        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability
        self.elitism = True
        self.maximise_fitness = maximise_fitness
        self.current_generation = []
        self.max_val=100000000
        self.dataVals=[] #list значений для расчета расстояния в OK
        self.lenOK=[]#матрица расстояний между вещами рюкзака
        self.massGlob=seed_massGlob
        self.vGlob=seed_vGlob
        """Return individual's fitness value"""
        def fitness (individual, seed_data):
            fitnessMass = 0
            fitnessV = 0
            fitnessCost = 0
            fitnessMassAll = 0
            fitnessVAll = 0
            for (selected, (profit)) in zip(individual, seed_data):
                if selected:
                    fitnessMassAll +=float(profit[0])
                    fitnessVAll +=float(profit[1])
                    fitnessMass += 1/(float(profit[0])/10000)
                    fitnessV += 1/(float(profit[1])*10)*100
                    fitnessCost += float(profit[2])
                    #print(profit[0]+' -> '+str(fitnessMass))
            avgfitness=fitnessMass+fitnessV+fitnessCost
            if fitnessMassAll>self.massGlob or fitnessVAll>self.vGlob:
                fitness=math.log(avgfitness,2)#AVGfitness([fitnessMass,fitnessV,fitnessCost]),2);
            else:
                fitness = avgfitness#AVGfitness([fitnessMass,fitnessV,fitnessCost])
            return fitness
        """Create array OK"""
        def AVGfitness(item):
            mn=2/5;
            return mn*float(item[0])+mn*float(item[1])-(1-2*mn)*float(item[2])
        """Create array OK"""
        def create_LenOK_dataVals():    #функция перевода v+mass+cost = приспособленность
            self.dataVals=[]
            self.lenOK=[]
            for item in self.seed_data:
                self.dataVals.append(AVGfitness(item))
            minVal=abs(min(self.dataVals))+1
            for i in range(0,len(self.dataVals)):
                self.dataVals[i]+=minVal
            for i in range(0,len(self.dataVals)):
                row = []
                for j in range(0,len(self.dataVals)):
                    if i==j:
                        row.append(self.max_val) # if index1==index2 val = inf
                    else:
                        row.append(abs(self.dataVals[i]-self.dataVals[j]))
                self.lenOK.append(row)
        """Create new hromosome"""
        def create_individual(seed_data):
            IndexHromosome = [random.randint(0, len(seed_data)-1)]
            create_LenOK_dataVals()
            lenOKCopy=self.lenOK[:][:]
            theMass=0
            for item in lenOKCopy:
                item[IndexHromosome[0]]=self.max_val
            boolHromosomeFind=0 
            while(not boolHromosomeFind):
                lastIndex=IndexHromosome[len(IndexHromosome)-1]
                currentOKrow = lenOKCopy[lastIndex]
                find=0        
                while(not find):            
                    currentMinIndex = currentOKrow.index(min(currentOKrow))
                    for item in lenOKCopy:
                        item[currentMinIndex]=self.max_val
                    theMass = float(seed_data[currentMinIndex][0])
                    theV = float(seed_data[currentMinIndex][1])
                    for i in IndexHromosome:
                        theMass += float(seed_data[i][0])
                        theV += float(seed_data[i][1])
                    if(theMass>self.massGlob or theV>self.vGlob):
                        boolHromosomeFind=1
                    else:
                        IndexHromosome.append(currentMinIndex)
                    find=1
            Hromosome=[0]*len(seed_data)
            for i in IndexHromosome:
                Hromosome[i]=1
            return Hromosome
        """Compare individuals"""
        def EquelMembers(member1,member2):  #Сравнение 2х генов
            pairs=zip(member1,member2)
            for pair in pairs:
                if pair[0]!=pair[1]:
                    return 0
            return 1
        """Return only unique individuals"""
        def uniquePopulation(population):   #Оставляем только уникальных особей в population
            new_population=[population[0]]
            for member in population:
                i=0
                for member_new in new_population:
                    if not EquelMembers(member,member_new):
                        i+=1   
                if i==len(new_population):
                   new_population.append(member) 
            return new_population
        """Every bit from an random parent"""
        def crossover(parent_1, parent_2):
            child_1 = []
            child_2 = []
            parents=[parent_1,parent_2]
            for i in range(0,len(parent_1)):
                crossover_parent = random.randint(0, 1)
                child_1.append(parents[crossover_parent][i])
                crossover_parent = random.randint(0, 1)
                child_2.append(parents[crossover_parent][i])
            return child_1, child_2
        """Mutate individual, add one random thing"""
        def mutate(individual):
            mutate_index = random.randrange(len(individual))
            while individual[mutate_index] == 1:
                mutate_index = random.randrange(len(individual))
            individual[mutate_index] == 1
            return individual
        """Select and return a random member of the population."""
        def random_selection(population):
            return random.choice(population)
        """Selection function. Return new chromosome"""
        def tournament_selection(population):
            this_population = []
            for member in population:
                this_population.append(member.genes)
            unique_population = uniquePopulation(this_population)
            #высчитываем приспособленность для уникальных особей
            fitnessPopulation = []
            sum_fitnessPopulation = 0
            for member in unique_population:
                member_fitness = fitness(member,self.seed_data)
                fitnessPopulation.append([member_fitness,member])
                sum_fitnessPopulation += member_fitness
            #высчитываем P для особей
            N = len(unique_population)
            population_P=[]
            for i in range(0,N):
                population_P.append(fitnessPopulation[i][0]/sum_fitnessPopulation)
            #Бросаем рулетку
            roulette = random.random();
            #Находим i-й диапазон
            i = 0
            minVal=0
            maxVal=population_P[i]
            while not (minVal <= roulette < maxVal):
                minVal = minVal+population_P[i]
                i+=1
                maxVal= minVal+population_P[i]
                #print(str(minVal)+' -> '+str(maxVal)+' :'+str(i))
            return Chromosome(unique_population[i])
        self.fitness_function = fitness
        self.tournament_selection = tournament_selection
        self.tournament_size = self.population_size // 10
        self.random_selection = random_selection
        self.create_individual = create_individual
        self.crossover_function = crossover
        self.mutate_function = mutate
        self.selection_function = self.tournament_selection
    """Create members of the first population randomly."""
    def create_initial_population(self):
        initial_population = []
        for _ in range(self.population_size):
            genes = self.create_individual(self.seed_data)
            individual = Chromosome(genes)
            initial_population.append(individual)
        self.current_generation = initial_population
    """Calculate the fitness of every member of the given population using the supplied fitness_function. """
    def calculate_population_fitness(self):
        for individual in self.current_generation:
            individual.fitness = self.fitness_function(
                individual.genes, self.seed_data)
    """Sort the population by fitness according to the order defined by maximise_fitness. """
    def rank_population(self):
        self.current_generation.sort(
            key=attrgetter('fitness'), reverse=self.maximise_fitness)
        
    """Create a new population using the genetic operators (selection, crossover, and mutation) supplied. """
    def create_new_population(self):
        new_population = []
        elite = copy.deepcopy(self.current_generation[0])
        selection = self.selection_function
        while len(new_population) < self.population_size:
            parent_1 = copy.deepcopy(selection(self.current_generation))
            parent_2 = copy.deepcopy(selection(self.current_generation))
            child_1, child_2 = parent_1, parent_2
            child_1.fitness, child_2.fitness = 0, 0
            can_crossover = random.random() < self.crossover_probability
            if can_crossover:
                child_1.genes, child_2.genes = self.crossover_function(
                    parent_1.genes, parent_2.genes)
            new_population.append(child_1)
            if len(new_population) < self.population_size:
                new_population.append(child_2)
        persent10=int(self.population_size*self.mutation_probability)
        for i in range(0,persent10):
            self.mutate_function(new_population[i].genes)  
        if self.elitism:
            new_population[0] = elite
            self.current_generation = new_population

    """ Create the first population, calculate the population's fitness and
        rank the population by fitness according to the order specified. """
    def create_first_generation(self):
        self.create_initial_population()
        self.calculate_population_fitness()
        self.rank_population()

    """ Create subsequent populations, calculate the population fitness and
        rank the population by fitness in the order specified. """
    def create_next_generation(self):
        self.create_new_population()
        self.calculate_population_fitness()
        self.rank_population()
    """ Run (solve) the Genetic Algorithm."""
    def run(self):
        self.create_first_generation()

        for _ in range(1, self.generations):
            print(_)
            self.create_next_generation()
    """ Return the individual with the best fitness in the current generation. """
    def best_choice(self):
        best = self.current_generation[0]
        return (best.fitness, best.genes)
    
    """ Return members of the last generation as a generator function."""
    def last_generation(self):
        return ((member.fitness, member.genes) for member
                in self.current_generation)


class Chromosome(object):
    def __init__(self, genes):
        self.genes = genes
        self.fitness = 0
    def __repr__(self):
        return repr((self.fitness, self.genes))

ga = GeneticAlgorithm(dataSet, massGlob, vGlob)
ga.run()
print (ga.best_choice())
mass=0
V=0
cost=0
ans = ga.best_choice()

for (select, (profit)) in zip(ans[1], dataSet):
        if select:
            mass += float(profit[0])
            V += float(profit[1])
            cost += float(profit[2])
            
print('Mass: '+str(massGlob))
print('V: '+str(vGlob))
print('Mass_ans: '+str(mass))
print('V_ans: '+str(V))
print('Cost_ans: '+str(cost))
