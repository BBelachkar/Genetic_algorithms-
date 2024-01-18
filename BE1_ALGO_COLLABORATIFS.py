#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.optimize import minimize
import time


# In[2]:


def f(x):
    # La fonction à optimiser
    condition = (x <= 1) & (x >= -1)
    result = np.zeros_like(x)
    result[condition] = -x[condition]**2 * ((2 + np.sin(10*x[condition]))**2)
    return result


# In[3]:


x0 = np.array([0.0])

# Utiliser la fonction minimize pour trouver le minimum
result = minimize(f, x0)
print(result)


# In[4]:


# Générer les valeurs de x et y pour tracer la fonction
x = np.linspace(-2, 2, 100)
y = np.array([f(xi) for xi in x])

# Tracer la fonction
plt.plot(x, y)

# Ajouter une légende et un titre
plt.xlabel("x")
plt.ylabel("y")
plt.title("Tracé de la fonction f(x)")

# Afficher le graphique
plt.show()


# In[5]:


def mutation(ind, em, pm):
    # Mutation
    if np.random.random() < pm:
        # Petite mutation
        return ind + np.random.uniform(-em, em)
    else:
        # Pas de mutation
        return ind

def crossover(ind1, ind2):
    # Crossover
    return 0.5 * (ind1 + ind2)

def evolutionary_algorithm(prec, Npop, pM, pm, ps, em, n_runs=10):
    times = []
    bests = []
    for _ in range(n_runs):
        # Initialisation aléatoire de la population
        pop = np.random.uniform(-1, 1, size=(Npop,))
        # Calcul de l'erreur initiale
        err = f(pop).min()
        start_time = time.time()
        while err > prec:
            # Tri de la population en fonction de l'erreur
            pop_sorted = np.sort(pop, axis=0)
            # Sélection des individus
            best = pop_sorted[:int(pM * Npop)]
            good = pop_sorted[int(pM * Npop):int((pM + pm) * Npop)]
            rest = pop_sorted[int((pM + pm) * Npop):]
            # Mutation des individus
            best = np.array([mutation(x, em) for x in best])
            good = np.array([mutation(x, em) for x in good])
            # Crossover des individus
            ps_size = int(ps * Npop)
            pairs = np.random.choice(rest, (ps_size, 2))
            sex = np.array([crossover(p[0], p[1]) for p in pairs])
            # Nouvelle population
            pop = np.concatenate((best, good, sex))
            # Calcul de l'erreur
            err = f(pop).min()
        # Enregistrement du meilleur individu et du temps d'exécution
        bests.append(pop.min())
        times.append(time.time() - start_time)
    # Retourne le temps moyen et le meilleur individu trouvé
    return np.mean(times), np.min(bests)


# In[6]:


Npop = 10
pM = 0.8
pm = 0
ps = 0
em = 0.1

precisions = [10**(-8/i) for i in range(1, 11)]
temps_moyens = []

for prec in precisions:
    t_debut = time.time()
    meilleur_x = evolutionary_algorithm(prec, Npop, pM, pm, ps, em, n_runs=10)
    temps_moyens.append(time.time() - t_debut)

plt.plot([-8/i for i in range(1, 11)], np.log10(temps_moyens))
plt.xlabel('log10(precision)')
plt.ylabel('log10(temps moyen)')
plt.title('Temps moyen d\'atteinte de la précision en fonction de la précision')
plt.show()


# In[57]:


Npop = 1000
pM = 0.8
pm = 0.1
ps = 0
em = 0.05
prec = 1e-8

meilleur_x = evolutionary_algorithm(prec, Npop, pM, pm, ps, em)
print("Meilleur x trouvé :", meilleur_x)


# In[58]:


Npop = 10
pM = 0.8
pm = 0.3
ps = 0
em = 0.01
prec = 1e-8

meilleur_x = evolutionary_algorithm(prec, Npop, pM, pm, ps, em)
print("Meilleur x trouvé :", meilleur_x)


# In[59]:


Npop_values = [10**i for i in range(1,6)]
pm = ps = pM = (5-1)/3
em = 0.01
prec = 10**(-8)
temps_moyens = []

for Npop in Npop_values:
    t_debut = time.time()
    temps = []
    for i in range(10):
        temps.append(evolutionary_algorithm(prec, Npop, pM, pm, ps, em))
    temps_moyens.append(np.mean(temps))

plt.plot(np.log10(Npop_values), np.log10(temps_moyens), 'bo-')
plt.xlabel('log10(Npop)')
plt.ylabel('log10(Temps moyen)')
plt.show()


# In[60]:


Npop_values = [10**i for i in range(1,6)]
pm = ps = pM = (5-1)/3
em = 0.01
prec = 10**(-8)
temps_moyens = []

for Npop in Npop_values:
    t_debut = time.time()
    temps = []
    for i in range(10):
        temps.append(evolutionary_algorithm(prec, Npop, pM, pm, ps, em))
    temps_moyens.append(np.mean(temps))

plt.plot(np.log10(Npop_values), np.log10(temps_moyens), 'bo-')
plt.xlabel('log10(Npop)')
plt.ylabel('log10(Temps moyen)')
plt.show()


# In[2]:


def esperance1(P1,P2):
    return P1*P2 + (1-P1)*(1-P2) - P1*(1-P2) - P2*(1-P1) 


# In[3]:


def esperance2(P1, P2):
    return -esperance1(P1, P2)


# In[4]:


def mutation(p):
    s = -1.0

    while ((s > 1) or (s < 0)):
        r = np.random.uniform(-0.5, 0.5)
        s = p * (1 + r)

    return s


# In[10]:


C=[0]*tmax
N1=10
N2=10
tmax=1000
A = np.zeros(tmax) # va contenir les probabilités maximales pour la population 1
B = np.zeros(tmax)
Temps = np.arange(1, tmax+1)

# probabilité de dire pair pour la population 1
P1 = np.zeros(N1)
for i in range(N1):
    P1[i] = np.random.uniform(0,1)

# probabilité de dire pair pour la population 2
P2 = np.zeros(N2)
for i in range(N2):
    P2[i] = np.random.uniform(0,1)

E1 = [0] * N1
for i in range(N1):
    E1[i] = sum(esperance1(P1[i], P2)) # Espérances des joueurs 1 face à la population 2

E2 = [0] * N2
for j in range(N2):
    E2[j] = sum(esperance2(P1, P2[j])) # Espérances des joueurs 2 face à 1

imax1 = E1.index(max(E1)) # Meilleure stratégie parmi les joueurs 1
imax2 = E2.index(max(E2))
imin1 = E1.index(min(E1)) # Pire stratégie
imin2 = E2.index(min(E2))
Emax1 = max(E1)
Emax2 = max(E2)
Emin1 = min(E1)
Emin2 = min(E2)

for k in range(tmax):
    
    P1[imin1] = mutation(P1[imax1]) # on remplace dans population 1 le pire par une mutation du meilleur
    P2[imin2] = mutation(P2[imax2])

    for i in range(N1):
        E1[i] = esperance1(P1[i],P2[imax2]) # Espérances des joueurs 1 face au meilleur de 2 au temps précédent

    for j in range(N2):
        E2[j] = esperance2(P1[imax1],P2[j]) # Espérances des joueurs 2 face au meilleur de 1 au temps précédent
# On trouve les meilleures et pires stratégies
    imax1 = E1.index(max(E1)) # Meilleure stratégie parmi les joueurs 1
    imax2 = E2.index(max(E2))
    imin1 = E1.index(min(E1)) # Pire stratégie
    imin2 = E2.index(min(E2))
    Emax1 = max(E1)
    Emax2 = max(E2)
    Emin1 = min(E1)
    Emin2 = min(E2)
    C[k] = max(E1)
    A[k] = E1[imax1];
    B[k] = P2[imax2];
print(A[C.index(max(C))])
plt.plot(Temps, A)
#plt.plot(Temps, B, 'r')
plt.show()


# In[15]:


def esperance1(p, q, X, Y):
    B=10
    
    E1 = p * (1-q)
    
    if X > Y:
        E1 = E1 + (1+B)*p*q - 1 + p
    
    if X < Y:
        E1 = E1 - (1+B)*p*q - 1 + p
    
    return E1


# In[16]:


def esperance1_combat(j1, j2, P, Q):
    h = 1/10
    
    E = 0
    
    for i1 in range(10):
        for i2 in range(10):
            E = E + esperance1(P[i1][j1], Q[i2][j2], i1, i2)
    
    E = E*(1/h**2)
    
    return E


# In[21]:


def erreur(P1,P2):
    e=0
    for i in range(10):
        e = e + (P1[i]-P2[i])**2
    e = e**0.5
    return e


# In[25]:


def mutation(strategie_entree):
    strategie_sortie = np.zeros(10)
    for i in range(10):
        strategie_sortie[i] = -1
        while ((strategie_sortie[i] > 1) or (strategie_sortie[i] < 0)):
            r = np.random.uniform(-0.5, 0.5)
            strategie_sortie[i] = strategie_entree[i] * (1 + r)
    return strategie_sortie


# In[30]:


B = 10 # Mise
h = 1/10  # découpage en intervalles des valeurs de X et Y

N = 10
tmax = 10

# Initialisation

P = np.zeros((10,N))
Q = np.zeros((10,N))

for j in range(N):
    for i in range(10): # Découpage de l'intervale [0,1] en 10 segments
        P[i,j]= np.random.uniform(0, 1) # Une stratégie pj pour X dans l'intervalle [ih,(i+1)h] avec h = 1/10
        Q[i,j] = np.random.uniform(0, 1)

E1 = np.zeros(N)

for j1 in range(N):
    for j2 in range(N):
        E1[j1] = E1[j1] + esperance1_combat(j1,j2,P,Q) # espérance du joueur j1 face à toute la population 2

E2 = np.zeros(N)

for j2 in range(N):
    for j1 in range(N):
        E2[j2] = E2[j2] - esperance1_combat(j1,j2,P,Q) # espérance du joueur j2 face à toute la population 1

imax1 = np.argmax(E1) # Meilleure stratégie parmi les joueurs 1
imax2 = np.argmax(E2)
imin1 = np.argmin(E1) # Pire stratégie
imin2 = np.argmin(E2)
Emax1 = max(E1)
Emax2 = max(E2)
Emin1 = min(E1)
Emin2 = min(E2)

# Récurrence

print(Emin1)


# In[27]:



Erreur1 = np.zeros(tmax)
Erreur2 = np.zeros(tmax)

for t in range(tmax):
        
    P[:, imin1] = mutation(P[:, imax1])  # On remplace dans population 1 le pire par une mutation du meilleur
    Q[:, imin2] = mutation(Q[:, imax2])
        
    for i1 in range(N):
        E1[i1] = esperance1_combat(i1, imax2, P, Q)  # Espérances des joueurs 1 face au meilleur de 2 au temps précédent
    for i2 in range(N):
        E2[i2] = -esperance1_combat(imax1, i2, P, Q)  # Espérances des joueurs 2 face au meilleur de 1 au temps précédent
    
    # On trouve les meilleures et pires stratégies
    imax1bis = np.argmax(E1)
    imax2bis = np.argmax(E2)
    imin1 = np.argmin(E1)
    imin2 = np.argmin(E2)    

    e1 = erreur(P[:,imax1],P[:,imax1bis]) # On calcule l'écart de la meilleure stratégie par rapport à la meilleure stratégie au temps précédent
    e2 = erreur(Q[:,imax2],Q[:,imax2bis])
    
    imax1 = np.unravel_index(imax1bis, E1.shape)[0]
    imax2 = np.unravel_index(imax2bis, E2.shape)[0]
    
    Erreur1[t] = e1
    Erreur2[t] = e2
    
Temps = np.arange(tmax)

plt.plot(Temps, Erreur1)
plt.plot(Temps, Erreur2, 'r')
plt.show()


# In[ ]:




