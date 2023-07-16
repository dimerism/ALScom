
import numpy as np



def merge_alpha_words(alphabet, words):


    alphabet = alphabet.tolist()

    Na = len(alphabet)
    Nw = len(words)


    #N = Na+Nw
    N = Na + Na//3 +Na%3 # a cada 3 simbolos adiciona uma palavra ou espaço em branco
    alphabetPlusWords = [None]*N

    ca = 0
    cw = 0
    #n = 0
    #print([Na,Nw])
    #while ca<Na or cw<Nw:
      #  print([ca,cw])    
    for n in range(N):

        if (n+1)%4 == 0:
            
            if cw>Nw-1:
              
                alphabetPlusWords[n] = ' '
                                   
            else:
                
                alphabetPlusWords[n] = words[cw]
                cw +=1
            
        else:
            
            if ca>Na-1:

                alphabetPlusWords[n] = ' '

            else:
                
                alphabetPlusWords[n] = alphabet[ca]
                ca +=1
        #n +=1
    
    return alphabetPlusWords

