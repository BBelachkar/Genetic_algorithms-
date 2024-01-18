#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from scipy.optimize import minimize
import time


# In[2]:


# La sortie Y=1 si x1=1 (ou bien y=x1 and x3)
# Pas de hidden couche, donc les résultats ne seront pas top.
import numpy as np
# La fonction de combinaison (ici une sigmoide)
def sigmoide(x) : return 1/(1+np.exp(-x))
# Et sa dérivée
def derivee_de_sigmoide(x) : return x*(1-x)
# Les entrées
X = np.array([ [0,0,0], [0,0,1], [0,1,0], [0,1,1], [1,0,0], [1,0,1], [1,1,0], [1,1,1] ])
# Les sorties (ici un vecteur)
Y = np.array([[0, 0, 0, 0, 1, 1, 1, 1]]).T # Transposé
# On utilise seed pour rendre les calculs déterministes.
np.random.seed(1)
# Initialisation aléatoire des poids (avec une moyenne = 0)
synapse0 = 4*np.random.random((3,1))-2
couche_entree = X
erreurs = []
for i in range(100): # On peut augmenter ! # propagation vers l’avant (forward)
    couche_sortie = sigmoide(np.dot(couche_entree,synapse0)) # dot multiplication 
    #print(couche_sortie)
    # Quelle est l’erreur (l’écart entre les sorties calculées et attendues) 
    erreur_couche_sortie = Y - couche_sortie
    # Multiplier l’erreur (l’écart) par la pente du ïsigmode pour les valeurs dans couche_sortie
    delta_couche_sortie = erreur_couche_sortie * derivee_de_sigmoide(couche_sortie)
    # Mise àjour des poids : rétropropagation
    synapse0 += np.dot(couche_entree.T,delta_couche_sortie)
    erreurs.append(np.mean(np.abs(erreur_couche_sortie)))
print ("Les sorties après l’apprentissage :")
print (couche_sortie)
# Affichage de la courbe des erreurs
plt.plot(erreurs)
plt.xlabel('Iterations')
plt.ylabel('Erreur')
plt.show()


# In[3]:


import numpy as np
def sigmoide_et_sa_derivee(x,deriv=False):
    if(deriv==True):
        return x* (1-x)
    return 1/(1+np.exp(-x))

X = np.array([[0,0,1],
[0,1,1],
[1,0,1],
[1,1,1]])
y = np.array([[0], [1], [1], [0]])
# On utilise seed pour rendre les calculs déterministes.
np.random.seed(1)
# Initialisation aléatoire des poids (avec une moyenne = 0 et écart−type=1)
# Ici, on met la moyenne àzéro, le std ne change pas.
# L’écriture X = b*np.random.random((3,4)) - a
# permet un tirage dans [a,b)], ici entre b=1 et a=−1 (donc moyenne=0)
synapse0 = 2 * np.random.random((3,4)) - 1
synapse1 = 2 * np.random.random((4,1)) - 1
couche_entree = X
nb_iterations = 100000
for j in range(nb_iterations):
    # propagation vers l’avant (forward)
    # couche_entree = X
    couche_cachee = sigmoide_et_sa_derivee(np.dot(couche_entree,synapse0))
    couche_sortie = sigmoide_et_sa_derivee(np.dot(couche_cachee,synapse1)) # erreur ?
    erreur_couche_sortie = y - couche_sortie
    if j % (nb_iterations//10) == 0: # des traces de l’erreur
        print("Moyenne Erreur couche sortie :" + str(np.mean(np.abs(erreur_couche_sortie))))
    # pondération par l’erreur (si pente douce, ne pas trop changer sinon, changer pondérations,
    delta_couche_sortie = erreur_couche_sortie * sigmoide_et_sa_derivee(couche_sortie,deriv=True)
    # Quelle est la contribution de couche_cachee àl’erreur de couche_sortie
    # (suivant les pondérations)?
    error_couche_cachee = delta_couche_sortie.dot(synapse1.T)
    # Quelle est la "direction" de couche_cachee (dérivée) ?
    # Si OK, ne pas trop changer la valeur.
    delta_couche_cachee = error_couche_cachee * sigmoide_et_sa_derivee(couche_cachee,deriv=True)
    synapse1 += couche_cachee.T.dot(delta_couche_sortie)
    synapse0 += couche_entree.T.dot(delta_couche_cachee)
print ("Résultat de l’apprentissage :")
print (couche_sortie)


# In[11]:


import numpy as np
def sigmoide_et_sa_derivee(x,deriv=False):
    if(deriv==True):
        return x* (1-x)
    return 1/(1+np.exp(-x))

def tanh_et_sa_derivee(x,deriv=False):
    if(deriv==True):
        return 1 - (2/(1+np.exp(-2*x)) - 1) **2
    return 2/(1+np.exp(-2*x)) - 1


X = np.array([[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]])
y = np.array([[0], [0], [0],[0],[0],[0],[1],[0]])
# On utilise seed pour rendre les calculs déterministes.
np.random.seed(1)
# Initialisation aléatoire des poids (avec une moyenne = 0 et écart−type=1)
# Ici, on met la moyenne àzéro, le std ne change pas.
# L’écriture X = b*np.random.random((3,4)) - a
# permet un tirage dans [a,b)], ici entre b=1 et a=−1 (donc moyenne=0)
synapse0 = 2 * np.random.random((3,4)) - 1
synapse1 = 2 * np.random.random((4,1)) - 1
couche_entree = X
nb_iterations = 100000
for j in range(nb_iterations):
    # propagation vers l’avant (forward)
    # couche_entree = X
    couche_cachee = tanh_et_sa_derivee(np.dot(couche_entree,synapse0))
    couche_sortie = tanh_et_sa_derivee(np.dot(couche_cachee,synapse1)) # erreur ?
    erreur_couche_sortie = y - couche_sortie
    if j % (nb_iterations//10) == 0: # des traces de l’erreur
        print("Moyenne Erreur couche sortie :" + str(np.mean(np.abs(erreur_couche_sortie))))
    # pondération par l’erreur (si pente douce, ne pas trop changer sinon, changer pondérations,
    delta_couche_sortie = erreur_couche_sortie * tanh_et_sa_derivee(couche_sortie,deriv=True)
    # Quelle est la contribution de couche_cachee àl’erreur de couche_sortie
    # (suivant les pondérations)?
    error_couche_cachee = delta_couche_sortie.dot(synapse1.T)
    # Quelle est la "direction" de couche_cachee (dérivée) ?
    # Si OK, ne pas trop changer la valeur.
    delta_couche_cachee = error_couche_cachee * tanh_et_sa_derivee(couche_cachee,deriv=True)
    synapse1 += couche_cachee.T.dot(delta_couche_sortie)
    synapse0 += couche_entree.T.dot(delta_couche_cachee)
print ("Résultat de l’apprentissage :")
print (couche_sortie)
couche_entree = X
erreurs = []
for i in range(100): # On peut augmenter ! # propagation vers l’avant (forward)
    couche_sortie = tanh_et_sa_derivee(np.dot(couche_entree,synapse0),False) # dot multiplication 
    #print(couche_sortie)
    # Quelle est l’erreur (l’écart entre les sorties calculées et attendues) 
    erreur_couche_sortie = y - couche_sortie
    # Multiplier l’erreur (l’écart) par la pente du ïsigmode pour les valeurs dans couche_sortie
    delta_couche_sortie = erreur_couche_sortie * tanh_et_sa_derivee(couche_sortie,True)
    # Mise àjour des poids : rétropropagation
    synapse0 += np.dot(couche_entree.T,delta_couche_sortie)
    erreurs.append(np.mean(np.abs(erreur_couche_sortie)))
print ("Les sorties après l’apprentissage :")
print (couche_sortie)
# Affichage de la courbe des erreurs
plt.plot(erreurs)
plt.xlabel('Iterations')
plt.ylabel('Erreur')
plt.show()


# In[7]:


import numpy as np
def sigmoide_et_sa_derivee(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))
X = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])
y = np.array([[0],[1],[1],[0]])
# On utilise seed pour rendre les calculs déterministes.
np.random.seed(1)
# Initialisation aléatoire des poids (avec une moyenne = 0 et écart−type=1)
# Ici, on met la moyenne àzéro, le std ne change pas.
# L’écriture X = b*np.random.random((3,4)) − a
# permet un tirage dans [a,b)], ici entre b=1 et a=−1 (donc moyenne=0)
synapse0 = 2*np.random.random((3,4)) - 1
synapse1 = 2*np.random.random((4,1)) - 1
couche_entree = X
nb_iterations = 100000
err=[]
for j in range(nb_iterations):
# propagation vers l’avant (forward)
# couche_entree = X
    couche_cachee = sigmoide_et_sa_derivee(np.dot(couche_entree,synapse0))
    couche_sortie = sigmoide_et_sa_derivee(np.dot(couche_cachee,synapse1))
# erreur ?
    erreur_couche_sortie = y - couche_sortie
    # Enregistrement de l’erreur
    moy=np.mean(np.abs(erreur_couche_sortie))
    err.append(abs(moy))
    if j % (nb_iterations//100) == 0: # des traces de l’erreur
        print("Moyenne Erreur couche sortie :" + str(moy))
# pondération par l’erreur (si pente douce, ne pas trop changer sinon, changer pondérations,
    delta_couche_sortie = erreur_couche_sortie*sigmoide_et_sa_derivee(couche_sortie,deriv=True)
# Quelle est la contribution de couche_cachee àl’erreur de couche_sortie
# (suivant les pondérations)?
    error_couche_cachee = delta_couche_sortie.dot(synapse1.T)
# Quelle est la "direction" de couche_cachee (dérivée) ?
# Si OK, ne pas trop changer la valeur.
    delta_couche_cachee = error_couche_cachee * sigmoide_et_sa_derivee(couche_cachee,deriv=True)
    synapse1 += couche_cachee.T.dot(delta_couche_sortie)
    synapse0 += couche_entree.T.dot(delta_couche_cachee)
print ("Résultat de l’apprentissage :")
print (couche_sortie)

# ——— La courbe de l’erreur ———–
import matplotlib.pyplot as plt
myvec=np.array([range(len(err)),err])
plt.plot(myvec[0,],myvec[1,]);plt.show()


# In[23]:


import numpy as np
def activation_function(x, function_type='sigmoid', deriv=False):
    if function_type == 'sigmoid':
        if deriv == True:
            return x*(1-x)
        return 1/(1+np.exp(-x))
    elif function_type == 'tanh':
        if deriv == True:
            return 1 - np.tanh(x)**2
        return np.tanh(x)
    elif function_type == 'gaussian':
        if deriv == True:
            return -2*x*np.exp(-x**2)
        return np.exp(-x**2)
    elif function_type == 'arctan':
        if deriv == True:
            return 1/(1+x**2)
        return np.arctan(x)

X = np.array([[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]])
y = np.array([[0], [0], [0],[0],[0],[0],[1],[0]])
# On utilise seed pour rendre les calculs déterministes.
np.random.seed(1)
# Initialisation aléatoire des poids (avec une moyenne = 0 et écart−type=1)
# Ici, on met la moyenne àzéro, le std ne change pas.
# L’écriture X = b*np.random.random((3,4)) − a
# permet un tirage dans [a,b)], ici entre b=1 et a=−1 (donc moyenne=0)
synapse0 = 2*np.random.random((3,4)) - 1
# Initialisation des synapses pour la deuxième couche cachée
synapse1 = 2*np.random.random((4,4)) - 1
# Initialisation des synapses pour la sortie
synapse2 = 2*np.random.random((4,1)) - 1
couche_entree = X
nb_iterations = 100000
err=[]
for j in range(nb_iterations):
# propagation vers l’avant (forward)
# couche_entree = X
    couche_cachee1 = activation_function(np.dot(couche_entree,synapse0),function_type='sigmoid')
    couche_cachee2 = activation_function(np.dot(couche_cachee1,synapse1),function_type='sigmoid')
    couche_sortie = activation_function(np.dot(couche_cachee2,synapse2),function_type='sigmoid')
# erreur ?
    erreur_couche_sortie = y - couche_sortie
    # Enregistrement de l’erreur
    moy=np.mean(np.abs(erreur_couche_sortie))
    err.append(abs(moy))
    if j % (nb_iterations//10) == 0: # des traces de l’erreur
        print("Moyenne Erreur couche sortie :" + str(moy))
        
    delta_couche_sortie = erreur_couche_sortie*activation_function(couche_sortie,function_type='sigmoid',deriv=True)
    delta_couche_cachee2 = delta_couche_sortie.dot(synapse2.T) * activation_function(couche_cachee2,function_type='sigmoid',deriv=True)
    delta_couche_cachee1 = delta_couche_cachee2.dot(synapse1.T) * activation_function(couche_cachee1,function_type='sigmoid',deriv=True)

    # Mise à jour des synapses
    synapse2 += couche_cachee2.T.dot(delta_couche_sortie)
    synapse1 += couche_cachee1.T.dot(delta_couche_cachee2)
    synapse0 += couche_entree.T.dot(delta_couche_cachee1)
print ("Résultat de l’apprentissage :")
print (couche_sortie)

# ——— La courbe de l’erreur ———–
import matplotlib.pyplot as plt
myvec=np.array([range(len(err)),err])
plt.plot(myvec[0,],myvec[1,]);plt.show()


# In[26]:





# In[3]:


import os
def lecture_matrice_X_et_Y_from_file(nom_fic) :
    os.chdir('C:\\Users\\LATITUDE\\Documents')
    if not nom_fic in os.listdir() :
        print("pb d’ouverture du fichier" + nom_fic)
        matrice_X_et_Y=[]
        return
# La matrice des données sous forme de caractères contenant des chiffres (des int)
    mat_tous_les_caracteres = open(nom_fic).read()
# matrice des données (les chiffres sont sous forme de caractères) : séparer les lignes
    matrix_cars = [item.split() for item in mat_tous_les_caracteres.split('\n')[:-1]]
    matrice_X_et_Y=[[int(row[i]) for i in range(len(row))] for row in matrix_cars]
    return matrice_X_et_Y #continet (pour BD chiffres) 48 bits et un chiffre 0..9
def code_Gray_repr_binaire_de_0_a_9(chiffre):
    """
    Convertit un chiffre de 0 à 9 en sa représentation binaire de 4 bits en code Gray.
    """
    if chiffre == 0:
        return [0,0, 0, 0]
    elif chiffre == 1:
        return [0, 0, 0, 1]
    elif chiffre == 2:
        return [0, 0, 1, 1]
    elif chiffre == 3:
        return [0, 0, 1, 0]
    elif chiffre == 4:
        return [0, 1, 1,0]
    elif chiffre == 5:
        return [0, 1, 1, 1]
    elif chiffre == 6:
        return [0, 1, 0, 1]
    elif chiffre == 7:
        return [0, 1, 0, 0]
    elif chiffre == 8:
        return [1 ,1, 0, 0]
    elif chiffre == 9:
        return [1, 1, 0, 1]
    else:
        return None

def code_Gray_repr_binaire_de_0_a_92(chiffre):
    """
    Convertit un chiffre décimal en code Gray binaire sur 4 bits
    
    Arguments:
    chiffre -- le chiffre à convertir (int)
    
    Returns:
    code_gray -- le code Gray binaire correspondant (array de 4 booléens)
    """
    binaire = bin(chiffre)[2:].zfill(4)  # Conversion en binaire sur 4 bits
    gray = binaire[0] + "".join([str(int(binaire[i]) ^ int(binaire[i+1])) for i in range(3)])  # Conversion en code Gray binaire sur 4 bits
    code_gray = np.array([bool(int(x)) for x in gray])  # Conversion en array de booléens
    return code_gray


# Définition de la fonction de préparation des données
def preparer_donnees(nom_fic) :
    global matrice_X
    global matrice_cible_y
    global L_Gray_repr_binaire_de_0_a_9
    global vecteur_chiffres_0_9_cible_y
    # Lecture de la matrice des données à partir du fichier
    matrice_X_et_Y=lecture_matrice_X_et_Y_from_file(nom_fic) # Vu ci−dessus
    if (matrice_X_et_Y==[]) :
        print("pb constitution de la matrice_X_et_Y")
        quit() # ou exit()
    # Ici, matrice_X_et_Y est prete : on enlève la décion (un chiffre 0..9 àla fin de la ligne)
    matrice_X = np.array([row[:-1] for row in matrice_X_et_Y])
# Et on transforme la décision 0..9 en 4 bits (code Gray)
    matrice_cible_y = np.array([code_Gray_repr_binaire_de_0_a_9(row[-1]) for row in matrice_X_et_Y])


matrice_X_et_Y=preparer_donnees("test.data")  # Chargement des données d'entraînement

print(matrice_X_et_Y)


# In[4]:


preparer_donnees("train.data")

print(matrice_X)
print(matrice_cible_y)


# In[35]:


import numpy as np
def activation_function(x, function_type='sigmoid', deriv=False):
    if function_type == 'sigmoid':
        if deriv == True:
            return x*(1-x)
        return 1/(1+np.exp(-x))
    elif function_type == 'tanh':
        if deriv == True:
            return 1 - np.tanh(x)**2
        return np.tanh(x)
    elif function_type == 'gaussian':
        if deriv == True:
            return -2*x*np.exp(-x**2)
        return np.exp(-x**2)
    elif function_type == 'arctan':
        if deriv == True:
            return 1/(1+x**2)
        return np.arctan(x)


# On utilise seed pour rendre les calculs déterministes.
np.random.seed(1)
# Initialisation aléatoire des poids (avec une moyenne = 0 et écart−type=1)
# Ici, on met la moyenne àzéro, le std ne change pas.
# L’écriture X = b*np.random.random((3,4)) − a
# permet un tirage dans [a,b)], ici entre b=1 et a=−1 (donc moyenne=0)
synapse0 = 2*np.random.random((48,96)) - 1
# Initialisation des synapses pour la deuxième couche cachée
synapse1 = 2*np.random.random((96,1)) - 1
couche_entree = X
nb_iterations = 100000
err=[]
for j in range(nb_iterations):
# propagation vers l’avant (forward)
# couche_entree = X
    couche_cachee1 = activation_function(np.dot(couche_entree,synapse0),function_type='sigmoid')
    couche_sortie = activation_function(np.dot(couche_sortie,synapse1),function_type='sigmoid')
# erreur ?
    erreur_couche_sortie = y - couche_sortie
    # Enregistrement de l’erreur
    moy=np.mean(np.abs(erreur_couche_sortie))
    err.append(abs(moy))
    if j % (nb_iterations//10) == 0: # des traces de l’erreur
        print("Moyenne Erreur couche sortie :" + str(moy))
        
    delta_couche_sortie = erreur_couche_sortie*activation_function(couche_sortie,function_type='sigmoid',deriv=True)
    delta_couche_cachee1 = delta_couche_sortie.dot(synapse1.T) * activation_function(couche_cachee1,function_type='sigmoid',deriv=True)

    # Mise à jour des synapses
    synapse1 += couche_cachee1.T.dot(delta_couche_sortie)
    synapse0 += couche_entree.T.dot(delta_couche_cachee1)
print ("Résultat de l’apprentissage :")
print (couche_sortie)

# ——— La courbe de l’erreur ———–
import matplotlib.pyplot as plt
myvec=np.array([range(len(err)),err])
plt.plot(myvec[0,],myvec[1,]);plt.show()


# In[5]:


def preparer_donnees(nom_fic) :
    """ pour l’efficacité, travailler avec les matrices en données globales """
    global matrice_X
    global matrice_cible_y
    global L_Gray_repr_binaire_de_0_a_9
    global vecteur_chiffres_0_9_cible_y
    matrice_X_et_Y = lecture_matrice_X_et_Y_from_file(nom_fic) # Vu ci−dessus
    if (matrice_X_et_Y==[]) :
        print("pb constitution de la matrice_X_et_Y")
        exit() # ou quit()
    # Ici, matrice_X_et_Y est prête : on enlève la décision (un chiffre 0..9 à la fin de la ligne)
    matrice_X = np.array([row[:-1] for row in matrice_X_et_Y])
    # Et on transforme la décision 0..9 en 4 bits (code Gray)
    matrice_cible_y = np.array([code_Gray_repr_binaire_de_0_a_9(row[-1]) for row in matrice_X_et_Y])


# In[47]:


def activation_function(x, function_type='sigmoid', deriv=False):
    if function_type == 'sigmoid':
        if deriv == True:
            return x*(1-x)
        return 1/(1+np.exp(-x))
    elif function_type == 'tanh':
        if deriv == True:
            return 1 - np.tanh(x)**2
        return np.tanh(x)
    elif function_type == 'gaussian':
        if deriv == True:
            return -2*x*np.exp(-x**2)
        return np.exp(-x**2)
    elif function_type == 'arctan':
        if deriv == True:
            return 1/(1+x**2)
        return np.arctan(x)
    
alphas = [0.001,0.01,0.1,1,10,100,1000]

# On utilise seed pour rendre les calculs déterministes.
np.random.seed(1)
# Initialisation aléatoire des poids (avec une moyenne = 0 et écart−type=1)
# Ici, on met la moyenne àzéro, le std ne change pas.
# L’écriture X = b*np.random.random((3,4)) - a
# permet un tirage dans [a,b)], ici entre b=1 et a=−1 (donc moyenne=0)
y=np.array(matrice_cible_y)
for alpha in alphas:
    print("\nApprentissage Avec Alpha:" + str(alpha))
    np.random.seed(1)
    # Init des pondérations (avec mu=0)
    synapse0 = 2 * np.random.random((48,32)) - 1
    synapse1 = 2 * np.random.random((32,32)) - 1
    couche_entree = np.array(matrice_X)
    nb_iterations = 100000
    synapse2 = 2 * np.random.random((32,4)) - 1
    for j in range(nb_iterations):
        # propagation vers l’avant (forward)
        # couche_entree = X
        
        couche_cachee1 = activation_function(np.dot(couche_entree,synapse0),function_type='sigmoid')
        couche_cachee2 = activation_function(np.dot(couche_cachee1,synapse1),function_type='sigmoid')
        couche_sortie = activation_function(np.dot(couche_cachee2,synapse2),function_type='sigmoid')
    # erreur ?
        erreur_couche_sortie = y - couche_sortie
        # Enregistrement de l’erreur
        moy=np.mean(np.abs(erreur_couche_sortie))
        if j % (nb_iterations//10) == 0: # des traces de l’erreur
            print("Moyenne Erreur couche sortie :" + str(moy))

        delta_couche_sortie = erreur_couche_sortie*activation_function(couche_sortie,function_type='sigmoid',deriv=True)
        delta_couche_cachee2 = delta_couche_sortie.dot(synapse2.T) * activation_function(couche_cachee2,function_type='sigmoid',deriv=True)
        delta_couche_cachee1 = delta_couche_cachee2.dot(synapse1.T) * activation_function(couche_cachee1,function_type='sigmoid',deriv=True)

        # Mise à jour des synapses
        synapse2 += couche_cachee2.T.dot(delta_couche_sortie)
        synapse1 += couche_cachee1.T.dot(delta_couche_cachee2)
        synapse0 += couche_entree.T.dot(delta_couche_cachee1)
    print ("Résultat de l’apprentissage :")
    print (couche_sortie)


# In[5]:


code_Gray_repr_binaire_de_0_a_9(1)


# In[6]:


def mutation(p):
    s = -1.0

    while ((s > 1) or (s < 0)):
        r = np.random.uniform(-0.5, 0.5)
        s = p * (1 + r)

    return s


# In[7]:


def optimisation(synapse0,synapse1,synapse2):
    N1=10
    N2=10
    N3=10
    tmax=1000
    A = np.zeros(tmax) # va contenir les probabilités maximales pour la population 1
    B = np.zeros(tmax)
    Temps = np.arange(1, tmax+1)
    Erreur1 = np.zeros(tmax)
    
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

        A[k] = P1[imax1];
        B[k] = P2[imax2];

    plt.plot(Temps, A)
    plt.plot(Temps, B, 'r')
    plt.show()


# In[8]:


def activation_function(x, function_type='sigmoid', deriv=False):
    if function_type == 'sigmoid':
        if deriv == True:
            return x*(1-x)
        return 1/(1+np.exp(-x))
    elif function_type == 'tanh':
        if deriv == True:
            return 1 - np.tanh(x)**2
        return np.tanh(x)
    elif function_type == 'gaussian':
        if deriv == True:
            return -2*x*np.exp(-x**2)
        return np.exp(-x**2)
    elif function_type == 'arctan':
        if deriv == True:
            return 1/(1+x**2)
        return np.arctan(x)
    
synapse0 = 2 * np.random.random((48,32)) - 1
synapse1 = 2 * np.random.random((32,32)) - 1
couche_entree = np.array(matrice_X)
nb_iterations = 10000
synapse2 = 2 * np.random.random((32,4)) - 1
N1=10
N2=10
N3=10
A = np.zeros(nb_iterations) # va contenir les probabilités maximales pour la population 1
B = np.zeros(nb_iterations)
erreur_couche_sortie = np.zeros(nb_iterations)
    
for j in range(nb_iterations):
        # propagation vers l’avant (forward)
        # couche_entree = X
        
    couche_cachee1 = activation_function(np.dot(couche_entree,synapse0),function_type='sigmoid')
    couche_cachee2 = activation_function(np.dot(couche_cachee1,synapse1),function_type='sigmoid')
    couche_sortie = activation_function(np.dot(couche_cachee2,synapse2),function_type='sigmoid')
    # erreur ?
    erreur_couche_sortie[j] = y - couche_sortie
        # Enregistrement de l’erreur
    moy=np.mean(np.abs(erreur_couche_sortie))
    if j % (nb_iterations//10) == 0: # des traces de l’erreur
        print("Moyenne Erreur couche sortie :" + str(moy))

    delta_couche_sortie = erreur_couche_sortie*activation_function(couche_sortie,function_type='sigmoid',deriv=True)
    delta_couche_cachee2 = delta_couche_sortie.dot(synapse2.T) * activation_function(couche_cachee2,function_type='sigmoid',deriv=True)
    delta_couche_cachee1 = delta_couche_cachee2.dot(synapse1.T) * activation_function(couche_cachee1,function_type='sigmoid',deriv=True)

        # Mise à jour des synapses
    synapse2 += couche_cachee2.T.dot(delta_couche_sortie)
    synapse1 += couche_cachee1.T.dot(delta_couche_cachee2)
    synapse0 += couche_entree.T.dot(delta_couche_cachee1)
print ("Résultat de l’apprentissage :")
print (couche_sortie)


# In[9]:


def fitness(population):
    """
    Calcule la fitness de chaque membre de la population.
    """
    fitness_scores = []
    for i in range(population_size):
        # Initialisation des pondérations
        synapse0 = population[i][:48*32].reshape(48, 32)
        synapse1 = population[i][48*32:48*32+32*32].reshape(32, 32)
        synapse2 = population[i][48*32+32*32:].reshape(32, 4)
        
        # propagation vers l’avant (forward)
        couche_cachee1 = activation_function(np.dot(couche_entree,synapse0),function_type='sigmoid')
        couche_cachee2 = activation_function(np.dot(couche_cachee1,synapse1),function_type='sigmoid')
        couche_sortie = activation_function(np.dot(couche_cachee2,synapse2),function_type='sigmoid')
        
        # calculer l'erreur
        erreur_couche_sortie = y - couche_sortie
        moy=np.mean(np.abs(erreur_couche_sortie))
        
        # enregistrer la fitness
        fitness_scores.append(moy)
    return fitness_scores
    # Calcul de la fitness pour chaque individu
for i in range(population_size):
        # Propagation vers l'avant
    couche_cachee1 = activation_function(np.dot(couche_entree, population[i]['weights1']), function_type='sigmoid')
    couche_cachee2 = activation_function(np.dot(couche_cachee1, population[i]['weights2']), function_type='sigmoid')
    couche_sortie = activation_function(np.dot(couche_cachee2, population[i]['weights3']), function_type='sigmoid')
        # Calcul de l'erreur
    erreur_couche_sortie = y - couche_sortie
    fitness = np.mean(np.abs(erreur_couche_sortie))
        # Mise à jour de la fitness pour l'individu i
    population[i]['fitness'] = fitness
        
    # Tri de la population en fonction de la fitness
population = sorted(population, key=lambda k: k['fitness'])
    
    # Sélection des parents pour la reproduction
parents = population[:num_parents]
    
    # Création de la nouvelle génération
new_generation = []
    
    # Reproduction
for i in range(num_parents):
    parent1 = parents[i]
    for j in range(num_offspring):
            # Sélection d'un second parent au hasard
        parent2 = parents[np.random.randint(0, num_parents)]
            # Création d'un nouvel individu à partir des parents
        child = {}
        child['weights1'] = crossover(parent1['weights1'], parent2['weights1'])
        child['weights2'] = crossover(parent1['weights2'], parent2['weights2'])
        child['weights3'] = crossover(parent1['weights3'], parent2['weights3'])
            # Mutation de l'individu avec une faible probabilité
        child = mutate(child, mutation_rate)
            # Ajout du nouvel individu à la nouvelle génération
        new_generation.append(child)
    
    # Ajout de la nouvelle génération à la population existante
population = parents + new_generation
    
print("Résultat de l'apprentissage:")
# Récupération de l'individu avec la meilleure fitness
best_individual = sorted(population, key=lambda k: k['fitness'])[0]
# Propagation vers l'avant avec les poids du meilleur individu
couche_cachee1 = activation_function(np.dot(couche_entree, best_individual['weights1']), function_type='sigmoid')
couche_cachee2 = activation_function(np.dot(couche_cachee1, best_individual['weights2']), function_type='sigmoid')
couche_sortie = activation_function(np.dot(couche_cachee2, best_individual['weights3']), function_type='sigmoid')
print(couche_sortie)


# In[10]:


import numpy as np

# Définition du réseau de neurones
input_size = 2
hidden_size = 3
output_size = 1

# Initialisation aléatoire des poids du réseau
weights = np.random.randn(input_size, hidden_size, output_size)

# Définition des paramètres de l'algorithme génétique
population_size = 20
mutation_rate = 0.01
num_generations = 100


# Boucle principale de l'algorithme génétique
for generation in range(num_generations):
    # Générer une nouvelle population de poids en faisant des mutations sur la population précédente
    population = [weights + np.random.randn(*weights.shape) * mutation_rate for i in range(population_size)]
    
    # Calculer le score de chaque individu dans la population
    fitness_scores = [evaluate_fitness(individual) for individual in population]
    
    # Garder les deux meilleurs individus pour la reproduction
    best_indices = np.argsort(fitness_scores)[-2:]
    parents = [population[i] for i in best_indices]
    
    # Garder les meilleurs synapses des deux parents pour la nouvelle population
    new_population = [parents[0], parents[1]]
    for i in range(2, population_size):
        # Mutations des synapses des mauvais individus en utilisant une mutation des meilleurs synapses
        child = np.zeros_like(weights)
        for j in range(weights.shape[0]):
            for k in range(weights.shape[1]):
                for l in range(weights.shape[2]):
                    if np.random.rand() < 0.5:
                        child[j,k,l] = parents[0][j,k,l] + np.random.randn() * mutation_rate
                    else:
                        child[j,k,l] = parents[1][j,k,l] + np.random.randn() * mutation_rate
        new_population.append(child)
        
    # Remplacer l'ancienne population par la nouvelle population
    population = new_population
    weights = population[0]
    
    # Afficher le score du meilleur individu de cette génération
    print('Generation {}: {}'.format(generation, fitness_scores[best_indices[0]]))
# Boucle principale de l'algorithme génétique


# In[14]:


import numpy as np
err = []
def sigmoid(x, deriv=False):
    if deriv == True:
        return x*(1-x)
    return 1/(1+np.exp(-x))

def initialise_individu(input_size,hidden_size,output_size):
    Individu = []
    synapse0 = 10 * np.random.random((hidden_size,input_size)) - 10
    synapse1 = 10 * np.random.random((output_size,hidden_size)) - 10
    Individu.append(synapse0)
    Individu.append(synapse1)
    return Individu

def initialiser_population(population_size, input_size, hidden_size, output_size):
    population = []
    for i in range(population_size):
        individu = initialise_individu(input_size,hidden_size,output_size)
        population.append(individu)
    return population 

def fitness(individu, X, y):
    couche_cachee1 = sigmoid(np.dot(X,individu[0]))
    couche_cachee2 = sigmoid(np.dot(couche_cachee1,individu[1]))
    couche_sortie = sigmoid(np.dot(couche_cachee2,individu[2]))
    erreur_couche_sortie = y - couche_sortie
    return np.mean(np.abs(erreur_couche_sortie))

def mutation(individu):
    individu_sortie = np.copy(individu)
    for i in range(3):
        for j in range(len(individu[i])):
            for k in range(len(individu[i][j])):
                individu_sortie[i][j][k] = -1
                while ((individu_sortie[i][j][k] > 1) or (individu_sortie[i][j][k] < -1)):
                    r = np.random.uniform(-0.5, 0.5)
                    individu_sortie[i][j][k] = individu[i][j][k] * (1 + r)          
    return individu_sortie

preparer_donnees("train.data")
Bonne_population = []
# Préparation des données

# Paramètres de l'algorithme génétique
population_size = 1000

# Initialisation aléatoire de la population
population = initialiser_population(population_size, 48, 96, 4)

# Algorithme

err = []
for j in population:
    err.append(fitness(j, matrice_X, matrice_cible_y))

i_bon = err.index(min(err)) # Meilleure stratégie parmi les joueurs 1
i_mauvais = err.index(max(err))
Err_bonne = min(err)
Err_mauvaise = max(err)

for k in range(200):
    err = []
    population[i_mauvais] = mutation(population[i_bon]) # réinitialiser la population sauf pour la meilleure stratégie
    for j in population:
        err.append(fitness(j, matrice_X, matrice_cible_y))

    i_bon = err.index(min(err)) # Meilleure stratégie parmi les joueurs 1
    i_mauvais = err.index(max(err))
    Err_bonne = min(err)
    Err_mauvaise = max(err)

print("minimum des erreur est :")
print(Err_bonne)
  


# In[50]:


# Algorithme
for i in range(100):
    for j in population:
        err.append(fitness(j, matrice_X, matrice_cible_y))
    
    i_bon = err.index(min(err)) # Meilleure stratégie parmi les joueurs 1
    i_mauvais = err.index(max(err))
    Err_bonne = min(err)
    Err_mauvaise = max(err)
    
    # réinitialiser la population sauf pour la meilleure stratégie
    nouvelle_population = [population[i_bon][0]
    for k in range(1, len(population)):
        synapse0 = mutation(population[i_bon][0])
        synapse1 = mutation(population[i_bon][1])
        synapse2 = mutation(population[i_bon][2])
        nouvelle_population.append([synapse0, synapse1, synapse2])
    population = nouvelle_population


# In[43]:



individu = np.array(initialise_individu(2,3,1))

synapse0 = 2 * np.random.random((2,3)) - 1
print(synapse0)
synapse1 = 2 * np.random.random((3,3)) - 1
synapse2 = 2 * np.random.random((3,1)) - 1


# In[22]:


import numpy as np
def initialise_individu(input_size,hidden_size,output_size):
    Individu = []
    synapse0 = 10 * np.random.random((hidden_size,input_size)) - 10
    synapse1 = 10 * np.random.random((output_size,hidden_size)) - 10
    Individu.append(synapse0)
    Individu.append(synapse1)
    return Individu

def initialiser_population(population_size, input_size, hidden_size, output_size):
    population = []
    for i in range(population_size):
        individu = initialise_individu(input_size,hidden_size,output_size)
        population.append(individu)
    return population 

def fitness(individu, X, y):
    couche_cachee1 = []
    for i in range(6):
        couche_cachee1[i] = np.dot(individu[0][i],X)
    couche_cachee1= activation_function(couche_cachee1,function_type='sigmoid1')        
    for i in range(16):
        couche_sortie = np.dot(individu[1][i],couche_cachee1)
    
    couche_sortie = activation_function(couche_sortie,function_type='sigmoid2')
    
    for i in range(4):
        erreur_couche_sortie = erreur_couche_sortie + (y[i] - couche_sortie[i])**2
    erreur_couche_sortie = (erreur_couche_sortie)**0.5
    return erreur_couche_sortie

def Petite_mutation(individu):
    individu_sortie = np.copy(individu)
    i = random.choice([0, 1])
    for j in range(len(individu[i])):
        for k in range(len(individu[i][j])):
            individu_sortie[i][j][k] = -10
            while ((individu_sortie[i][j][k] > 10) or (individu_sortie[i][j][k] < -10)):
                r = np.random.uniform(-0.5, 0.5)
                individu_sortie[i][j][k] = individu[i][j][k] * (1 + r)          
    return individu_sortie

def Grande_mutation(individu):
    individu = 10 * np.random.random((6,16)) - 10
    individu = 10 * np.random.random((1,6)) - 10
    return individu
            

def crossover(individu_meilleur,individu_mauvais):
    i = random.choice([0, 1])
    individu_mauvais[i] = individu_meilleur[i]
    return individu_mauvais

def activation_function(x, function_type='sigmoid1', deriv=False):
    if function_type == 'sigmoid1':
        if deriv == True:
            return 4*x*(1-x)
        return 1/(1+np.exp(-4*x))
    elif function_type == 'sigmoid2':
        if deriv == True:
            return 14*x*(1-x)
        return 7/(1+np.exp(-2*x))
           
X = np.array([[0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1],[0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1],
              [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0],[1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1]])
y = np.array([[0], [1], [2],[3]])
# On utilise seed pour rendre les calculs déterministes.
np.random.seed(1)

population_size = 20
nb_iterations = 500000
# Initialisation aléatoire de la population
population = initialiser_population(population_size, 16, 6, 1)

# Algorithme
err = []
for j in range(population_size):
    err.append(fitness(population[j], X, y))

i_bon = err.index(min(err)) # Meilleure stratégie parmi les joueurs 1
Err_bonne = min(err)
L=[]
itera=[]
for k in range(nb_iterations):
    itera.append(k)
    L1 = []
    L2 = []
    L3 = []
    L.append(np.mean(np.abs(err)))
    for i in range(10):
        L1.append(i_bon)
        err.remove(err[i_bon])
        i_bon = err.index(min(err))
    
    for i in range(5):
        L2.append(i_bon)
        err.remove(err[i_bon])
        i_bon = err.index(min(err))
        
    for i in range(5):
        L3.append(i_bon)
        err.remove(err[i_bon])
        i_bon = err.index(min(err))    
        
    for i in range(9):
        population[L1[i+1]]=crossover(population[L1[0]],population[L1[i+1]])
        
    for i in range(5):
        population[L2[i]]=Petite_mutation(population[L1[0]])
        
    for i in range(5):
        population[L3[i]]=Grande_mutation(population[L3[i]])    
        
    # réinitialiser la population sauf pour la meilleure stratégie
    for j in range(population_size):
        err.append(fitness(population[j], X, y))

    i_bon = err.index(min(err)) # Meilleure stratégie 
    Err_bonne = min(err)
    
    
plt.plot(itera, L)
plt.show()


# In[ ]:




