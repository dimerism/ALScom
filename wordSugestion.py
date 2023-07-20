import numpy as np
import string



def wordSugestion(partial_word,alphabet):



   
    alphabet = alphabet.tolist()
    lst_alpha = np.load('lst_alpha.npy')
    alpha_vals = np.load('alpha_vals.npy')
    pos_alph = np.load('pos_alph.npy')
    #compares strings to give suggestions
    p = partial_word.upper()
    lenp = len(p)

    pi = p[0]
    pos = alphabet.index(pi)

    posI = pos_alph[pos,0]
    posEnd = pos_alph[pos,1]

    possiveis = lst_alpha[int(posI):int(posEnd)]
    possiveis_vals = alpha_vals[int(posI):int(posEnd)]
    N = np.size(possiveis)
    score = np.zeros(N)

    for n in range(N):

        temp = possiveis[n].upper()

        tempscore = 0
        if len(p)<= len(temp):

            
            for k in range(len(p)):

                if p[k] == temp[k]:
                    tempscore +=1
                else:
                    tempscore = 0
                    break

        else:

         
            for k in range(len(temp)):

                if p[k] == temp[k]:
                    tempscore +=1
                tempscore = 0
                    
        score[n] = tempscore*(possiveis_vals[n]**(1/3)) #+ (possiveis_vals[n]**(1/2))

    rank = np.argsort(-score)
    score[rank]
    return possiveis[rank]
