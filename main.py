import cv2
import numpy as np
from datetime import datetime
import time
from screeninfo import get_monitors
import winsound
from constroi_teclado2 import *
from quad_alphabet import *
import matplotlib.pyplot as plt
from adiciona_texto import *
from sklearn.ensemble import RandomForestClassifier
import string
from wordSugestion import *
from merge_alpha_words import *
import pickle
import dlib
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

monitores = get_monitors()
monitor = monitores[0]


#####Land marks######

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def extract_eyes(face,landmarks):
    Oesquerdo = np.zeros((6,2))
    Odireito = np.zeros((6,2))
    count_e = 0
    count_d = 0
    for i in range(36, 48):
        x = landmarks.part(i).x
        y = landmarks.part(i).y
        
       
        if i<42:
            
            Oesquerdo[count_e][0] = x
            Oesquerdo[count_e][1] = y
            count_e+=1
            
        else:

            Odireito[count_d][0] = x
            Odireito[count_d][1] = y
            count_d+=1
            
    Oesquerdo = Oesquerdo.astype(int)
    Odireito = Odireito.astype(int)  
    Exmin,Eymin = np.min(Oesquerdo,axis=0)
    Exmax,Eymax = np.max(Oesquerdo,axis=0)            
    Dxmin,Dymin = np.min(Odireito,axis=0)
    Dxmax,Dymax = np.max(Odireito,axis=0)
    
    OlhoEsquerdo = face[Eymin-3:Eymax+3,Exmin-3:Exmax+3]
    OlhoDireito = face[Dymin-3:Dymax+3,Dxmin-3:Dxmax+3]
    return OlhoEsquerdo,OlhoDireito
        
def create_alphabet(texto):

    alphabet = list(string.ascii_uppercase)
    alphabet = np.array(alphabet)
    alphabet = np.append(alphabet,['_','<'])
    
    if len(texto) == 1:
    
        words = wordSugestion('a',alphabet)
    else:
        t = ''.join(texto)
        w = t.split()
        w = w[-1]
        words = wordSugestion(w,alphabet)
    
    
    if len(words)>4:
        words = words[0:4]
    

    
    alphabet = merge_alpha_words(alphabet, words)
    alphabet = np.array(alphabet)
    
    return alphabet

def extract_feat(threshold_img):

    maxDiffX, maxDiffY = dist_clus(threshold_img)
    Nlinhas,Ncolunas = np.shape(threshold_img)
    
    result = np.where(threshold_img==255)
    if np.size(result)==0:
        
        mean_linha = 0
        mean_coluna = 0
        std_linha = 0
        std_coluna = 0
        frequency = 0
    else:
        mean_linha = np.mean(result[0][:])/Nlinhas
        mean_coluna = np.mean(result[1][:])/Ncolunas
        std_linha = np.std(result[0][:])/Nlinhas
        std_coluna = np.std(result[1][:])/Ncolunas
        frequency = np.count_nonzero(threshold_img == 255)/np.size(threshold_img)
    eccent = Ncolunas/Nlinhas

    temp_amostras = np.array([mean_linha,mean_coluna,std_linha,std_coluna,frequency,eccent,maxDiffX,maxDiffY])
    return temp_amostras

    
def dist_clus(threshold_img):
    p=np.where(threshold_img==255)
    
    if np.size(p)>3: #minimum number of pixels with value 255
        x= p[0][:]
        y= p[1][:]
        
        cx = np.mean(x)
        cy = np.mean(y)
        dx = x-cx   #distancia do centro de massa
        dy = y-cy

        sortedX = np.sort(dx)  #ordena distancia
        sortedY = np.sort(dy)
        maxDiffX = np.max(np.diff(sortedX)) #maior vão distancia entre "clusters"
        maxDiffY = np.max(np.diff(sortedY))
        #print([np.max(np.diff(sortedX)),np.max(np.diff(sortedY))])

        return maxDiffX, maxDiffY
    else:
        return 0,0

    
def normaliza(imagem):
 
    imagem = (imagem-np.mean(imagem))
    imagem = imagem/np.std(imagem)
    imagem = (imagem - np.min(imagem))
    imagem = imagem/np.max(imagem)
    imagem = imagem*255

    return imagem.astype(np.uint8)

cap = cv2.VideoCapture(0)



#face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')





Klimiar = 0.5
limiardata = np.zeros((1,2))
bestK = 0
maxDiff = 0
maxDiffY = 0
maxDiffX = 0


ret = False
while not ret:
    
    ret, frame = cap.read()
    if not ret:
        print("Não foi possível acionar a câmera.")

while True:
    
   
    ret, frame = cap.read() #return either true or false depending of avaiability of the camera
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   
    cv2.imshow('Teste',gray_frame)
    widthC = int(cap.get(3))
    heightC = int(cap.get(4))
    #widthC,heightC,depth = np.shape(frame)
    
    faces = detector(gray_frame)
    for face in faces:
        landmarks = predictor(gray_frame, face)
        
        OlhoEsquerdo,OlhoDireito = extract_eyes(gray_frame,landmarks)       
        
        #Divide análise para olho esquerdo e direito
        #Olho esquerdo

     
        limiar, _ = cv2.threshold(OlhoEsquerdo,0,255,cv2.THRESH_OTSU)
        ret1,threshold_esquerdo = cv2.threshold(OlhoEsquerdo,int(limiar*Klimiar),255,cv2.THRESH_BINARY_INV)


        
        p=np.where(threshold_esquerdo==255)
        if np.size(p)>3:
            x= p[0][:]
            y= p[1][:]
            
            cx = np.mean(x)
            cy = np.mean(y)
            dx = x-cx
            dy = y-cy

            sortedX = np.sort(dx)
            sortedY = np.sort(dy)
            maxDiffX = np.max(np.diff(sortedX))
            maxDiffY = np.max(np.diff(sortedY))
            #print([np.max(np.diff(sortedX)),np.max(np.diff(sortedY))])
            if maxDiff<np.max(np.diff(sortedY)):
                bestK = Klimiar
                maxDiff = np.max(np.diff(sortedY))
                
            Dx = np.max(x)-np.min(x)
            Dy = np.max(y)-np.min(y)

            ratio = Dx/Dy
        else:
            ratio = 0
        
        result = np.where(threshold_esquerdo==255)
        frequency = np.count_nonzero(threshold_esquerdo == 255)/np.count_nonzero(threshold_esquerdo >= 0)

        #limiardata = np.append(limiardata,[[Klimiar,frequency]] , axis=0)
        print([Klimiar*100,frequency*100,ratio*100, maxDiffY])
        Klimiar = Klimiar + 0.01

        break
    if Klimiar<=0.01:
        cv2.waitKey(0)
    else:
        cv2.waitKey(30)
    if Klimiar>0.7:
        break
    if cv2.waitKey(30) == 32:
        break
    
    

print(bestK)
Klimiar = (bestK*0.95)
#Klimiar = 0.6

width = int(monitor.width)
height = int(monitor.height*0.90)

img = cv2.imread('icarohd.jpg')
img = cv2.resize(img,(width,height))
#height,width,depth = img.shape
font = cv2.FONT_HERSHEY_SIMPLEX

#cv2.imshow('Icaro1', img)

def extrai_caracteristicas(imagem):
    print("")

def classifica(food):
    print("")

img1 = img.copy()
imgCount = cv2.putText(img1,"Aperte barra de espaco para comecar.", (width//4,height-10),font,1,(0,255,0),2,cv2.LINE_AA)

cv2.imshow('IcaroCount', imgCount)

cv2.waitKey(0)
img1 = img.copy()
imgCount = cv2.putText(img1,"Feche os olhos quando ouvir o primeiro beep", (width//3,height//2),font,1,(0,255,0),2,cv2.LINE_AA)
imgCount = cv2.putText(img1,"E reabra quando ouvir o segundo beep (beep, nao boop).", (width//3,height//2 + 50),font,1,(0,255,0),2,cv2.LINE_AA)

cv2.namedWindow('IcaroCount', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('IcaroCount',cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
cv2.imshow('IcaroCount', imgCount)

cv2.waitKey(3000)

freq = 2000
duration = 700
freq_select = 400


#calibra
Ny = 2
Nx = 2
#Klimiar = 0.7
N = Ny*Nx

R = 20
py = np.linspace(R/2, height-R/2, num=Ny, endpoint=True)
px = np.linspace(R/2, width-R/2, num=Nx, endpoint=True)
Tempo_reflexo = 0.5 #atraso no início da captura ao trocar ponto de calibração
Tempo_captura = 3 #periodo de amostragem para cada ponto da tela
deltaT_captura = 0.05
amostras = np.zeros((1,16))
nova_amostra = np.zeros((1,16))
posicao = np.zeros((1,2))

#Captura dados olho fechado
then = time.time()
now = then

winsound.Beep(freq,duration)
time.sleep(Tempo_reflexo)
Namostras = 20 #número de amostras olho fechado
Ncapturadas = 0
while Namostras-Ncapturadas>0:
 
  
            
    ret, frame = cap.read() #return either true or false depending of avaiability of the camera
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   
    cv2.imshow('Teste',gray_frame)
    widthC = int(cap.get(3))
    heightC = int(cap.get(4))
    #widthC,heightC,depth = np.shape(frame)
    
    faces = detector(gray_frame)
    for face in faces:
        landmarks = predictor(gray_frame, face)
        
        OlhoEsquerdo,OlhoDireito = extract_eyes(gray_frame,landmarks)       
        OlhoEsquerdo = normaliza(OlhoEsquerdo)       
        OlhoDireito = normaliza(OlhoDireito)
     
        limiar, _ = cv2.threshold(OlhoEsquerdo,0,255,cv2.THRESH_OTSU)
        ret1,threshold_esquerdo = cv2.threshold(OlhoEsquerdo,int(limiar*Klimiar),255,cv2.THRESH_BINARY_INV)
        ret1,threshold_direito = cv2.threshold(OlhoDireito,int(limiar*Klimiar),255,cv2.THRESH_BINARY_INV)
    

        temp_amostras = extract_feat(threshold_esquerdo)         
        nova_amostra[0][:8] = np.copy(temp_amostras)
        
        temp_amostras = extract_feat(threshold_direito)     
        nova_amostra[0][8:] = np.copy(temp_amostras)
        
        amostras = np.append(amostras, nova_amostra,axis = 0)
        
        temp_posicao = np.array([[int(-1),int(-1)]])
        posicao = np.append(posicao,temp_posicao,axis=0)

        Ncapturadas = Ncapturadas + 1
        cv2.imshow('Olho_esquerdo',OlhoEsquerdo)
        cv2.waitKey(50)
        cv2.imshow('Olho_binary_esquerdo',OlhoDireito)
        cv2.waitKey(50)                  
            
        break
            


     


winsound.Beep(freq,duration)    
    #imgCount = cv2.putText(img1,str(int(np.ceil(4 - (now-then)))), (width//2,height//2),font,2,(0,255,0),4,cv2.LINE_AA)
    #cv2.imshow('IcaroCount', imgCount)
 
img1 = img.copy()
imgCount = cv2.putText(img1,"Quando estiver pronto pressione barra de espaco para comecar a calibracao.", (10,height//2),font,1,(0,255,0),2,cv2.LINE_AA)
cv2.namedWindow('IcaroCount', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('IcaroCount',cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
cv2.imshow('IcaroCount', imgCount)

cv2.waitKey(0)

P = np.array([[0,0]])
for n1 in range(Ny):
    for n2 in range(Nx):
        P = np.append(P,[[px[n2],py[n1]]],axis=0)
    px = np.flip(px)
P = np.delete(P,0,axis=0)
#calibra olho aberto
Namostras = 5
d = 10
D = 50
V = np.array((80,0))
p = 0
I = np.array((d,d))
F = np.array((width-d-20,height-d-20))
Px = 10
Py = 10
while p<4:
    time.sleep(0.05)
    #draws new circle for calibration
    img1 = img.copy()
    imgCircle = cv2.circle(img1, (int(Px),int(Py)), 10, (0,0,255), 8)
    cv2.namedWindow('IcaroCount', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('IcaroCount',cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
    cv2.imshow('IcaroCount', img1)

    
            
    ret, frame = cap.read() #return either true or false depending of avaiability of the camera
    #ret = True
    #frame = cv2.imread('rosto.jpg')
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  
    if ret==False:
        print("Camera não encontrada!")
    
    widthC = int(cap.get(3))
    heightC = int(cap.get(4))
    #widthC,heightC,depth = np.shape(frame)
    ################
    faces = detector(gray_frame)
    for face in faces:
        landmarks = predictor(gray_frame, face)
        
        OlhoEsquerdo,OlhoDireito = extract_eyes(gray_frame,landmarks)       
        OlhoEsquerdo = normaliza(OlhoEsquerdo)       
        OlhoDireito = normaliza(OlhoDireito)        
     
        limiar, _ = cv2.threshold(OlhoEsquerdo,0,255,cv2.THRESH_OTSU)
        ret1,threshold_esquerdo = cv2.threshold(OlhoEsquerdo,int(limiar*Klimiar),255,cv2.THRESH_BINARY_INV)
        ret1,threshold_direito = cv2.threshold(OlhoDireito,int(limiar*Klimiar),255,cv2.THRESH_BINARY_INV)
    

        temp_amostras = extract_feat(threshold_esquerdo)         
        nova_amostra[0][:8] = np.copy(temp_amostras)
        
        temp_amostras = extract_feat(threshold_direito)     
        nova_amostra[0][8:] = np.copy(temp_amostras)
        
        amostras = np.append(amostras, nova_amostra,axis = 0)
        
        temp_posicao = np.array([[int(Px),int(Py)]])
        posicao = np.append(posicao,temp_posicao,axis=0)
        
      
        Px = Px + V[0]
        Py = Py + V[1]
        if Px>F[0]:
            #print('px maior que f0')
            Px = F[0]-1
            V = np.roll(V,1)
        if Py>F[1]:
            Py = F[1]-1
            V = -np.roll(V,1)
        if Px<I[0]:
            Px = I[0]+1
            V = np.roll(V,1)
        if Py<I[1]:
            
            V = -np.roll(V,1)
            p = p + 1
            F[0] = F[0] - D
            F[1] = F[1] - D
            I[0] = I[0] + D
            I[1] = I[1] + D
            Py = I[1]+1
            Px = I[0]+1
        cv2.imshow('Olho_direito',OlhoEsquerdo)
        cv2.waitKey(30)
        cv2.imshow('Olho_binary_direito',threshold_esquerdo)
        cv2.waitKey(30)                            
        #k = k + 1
        break   
    if I[0]>=F[0] or I[1]>=F[1]:
        break
            


            
            

posicao = np.delete(posicao,0,axis=0)
amostras = np.delete(amostras,0,axis=0)

#remove nan values
nan_mask = np.isnan(amostras)
amostras[nan_mask]=0

media_amostras = amostras.mean(axis=0)
amostras =(amostras-media_amostras)
std_amostras = np.std(amostras,axis=0)


pos_zero = np.where(std_amostras==0)


if np.size(pos_zero)>0:

    std_amostras[pos_zero[:]] = 1   

amostras = amostras/std_amostras

#amostras =(amostras-media_amostras)/std_amostras
texto = np.empty((1,1),dtype=str)
################################# Aquisição e Classificação ################################
count_closed_it = 0

alphabet = create_alphabet(texto)

series = np.zeros(10)
seriesCount = np.zeros(4)
pos_est = np.array([[0,0]])
maxpos = 0
pos_est_previous = np.array([[0,0]])
#clf = RandomForestClassifier(random_state=0,n_estimators=5)
#clf.fit(amostras, posicao)
#X_train, X_test, y_train, y_test = train_test_split(amostras, posicao, test_size=0.2, random_state=42)
knn = KNeighborsRegressor(n_neighbors=3)

# Train the regressor using the training data
knn.fit(amostras, posicao)


while True:
    time.sleep(0.1)
    nova_amostra = np.zeros((1,16))
    ret, frame = cap.read() #return either true or false depending of avaiability of the camera
    #ret = True
    #frame = cv2.imread('rosto.jpg')
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if ret==False:
        print("Camera não encontrada!")

    widthC = int(cap.get(3))
    heightC = int(cap.get(4))
    #widthC,heightC,depth = np.shape(frame)
    ################

    faces = detector(gray_frame)
    for face in faces:
        landmarks = predictor(gray_frame, face)
        
        OlhoEsquerdo,OlhoDireito = extract_eyes(gray_frame,landmarks)       
        OlhoEsquerdo = normaliza(OlhoEsquerdo)       
        OlhoDireito = normaliza(OlhoDireito)
     
        limiar, _ = cv2.threshold(OlhoEsquerdo,0,255,cv2.THRESH_OTSU)
        ret1,threshold_esquerdo = cv2.threshold(OlhoEsquerdo,int(limiar*Klimiar),255,cv2.THRESH_BINARY_INV)
        ret1,threshold_direito = cv2.threshold(OlhoDireito,int(limiar*Klimiar),255,cv2.THRESH_BINARY_INV)
    

        temp_amostras = extract_feat(threshold_esquerdo)         
        nova_amostra[0][:8] = np.copy(temp_amostras)
        
        temp_amostras = extract_feat(threshold_direito)     
        nova_amostra[0][8:] = np.copy(temp_amostras)
        
        #amostras = np.append(amostras, nova_amostra,axis = 0)
        
        temp_posicao = np.array([[int(Px),int(Py)]])
        #posicao = np.append(posicao,temp_posicao,axis=0)



    
    #print('Nova amostra:')
    #print(nova_amostra)
    nova_amostra_backup = nova_amostra
    nova_amostra = (nova_amostra-media_amostras)/std_amostras

    #print('Nova amostra norm:')
    #print(nova_amostra)



    N1,N2 = np.shape(amostras)
    count_closed_eyes = 0
    cor = (0,0,255)
              
    pos_est = knn.predict(nova_amostra)
    if pos_est[0][0]<0:
        cor = (255,0,0)
        pos_est = pos_est_previous
        count_closed_it = count_closed_it + 1
    else:
        count_closed_it = count_closed_it - 1
    
   
    
   


    pos_est = np.floor(pos_est)

    if pos_est[0][0]<0:#incorrect
        #eyes closed
        f = 1
        
    elif pos_est_previous[0][0]>0: #contabiliza apenas se o anterior nao foi olho fechado
        
        if pos_est[0][0]<=width//2:
            if pos_est[0][1]<=height//2:
                # 0
                series[0] = 0 
                series = np.roll(series, 1)
            else:
                # 2
                series[0] = 2 
                series = np.roll(series, 1)
        else:
            if pos_est[0][1]<=height//2:
                # 1
                series[0] = 1 
                series = np.roll(series, 1)
            else:
                # 3
                series[0] = 3 
                series = np.roll(series, 1)
                
    pos_est_previous = pos_est
    
    seriesCount[0] = np.count_nonzero(series == 0)
    seriesCount[1] = np.count_nonzero(series == 1)
    seriesCount[2] = np.count_nonzero(series == 2)
    seriesCount[3] = np.count_nonzero(series == 3)
    maxpos = np.argmax(seriesCount)

    
    if count_closed_it >= 2:
        
        winsound.Beep(freq_select,duration)

        
        Q = quad_alphabet(alphabet)
        f = np.where(Q[:,0]==maxpos)
        
        temp_alphabet = alphabet[f[0][:]]
        non_blank = np.where(temp_alphabet != ' ')
        
        
        if np.size(non_blank)>1:
            
            alphabet = alphabet[f[0][:]]
            
        elif np.size(non_blank)==0:

            alphabet = create_alphabet(texto)
            
        elif np.size(non_blank)==1:
            
            if alphabet[f[0][0]]=='<':
                
                texto = texto[:-1]
                alphabet = create_alphabet(texto)
                
            elif alphabet[f[0][0]]=='_':
                
                texto = np.append(texto,' ')
                alphabet = create_alphabet(texto)
                
            else:
                if len(alphabet[f[0][0]])==1:
                    texto = np.append(texto,alphabet[f[0][0]])
                    print(texto)
                    alphabet = create_alphabet(texto)
                else:
                    pos_ultimo_espaco = np.where(texto== ' ')
                    if np.size(pos_ultimo_espaco)>0:
                        texto = texto[:pos_ultimo_espaco[0][-1]+1]
                        texto = np.append(texto,alphabet[f[0][0]])
                        texto = np.append(texto,' ')
                        print(texto)
                        alphabet = create_alphabet(texto)
                    else:
                        texto = texto[0]
                        texto = np.append(texto,alphabet[f[0][0]])
                        texto = np.append(texto,' ')
                        print(texto)
                        alphabet = create_alphabet(texto)
            
        print("Quadrante" + str(maxpos))
        cor = (0,255,0)
        count_closed_it = 0
        print("Olhos fechados")
        #run function to update alphabet
    if count_closed_it<0:
        count_closed_it = 0
        
    


    
        

    
    
    imagem_teclado = cria_teclado(alphabet,monitor.width,monitor.height)
    
    Dcolor = 20

    #retangulos design antigo
    #ret_cords = np.array([[(0,0),(width//2,height//2)],[(width//2,0),(width,height//2)],[(0,height//2),(width//2,height)],[(width//2,height//2),(width,height)]])
    #retangulos design atual
    ret_cords = np.array([[(0,0),(width//2,int(height*0.45))],[(width//2,0),(width,int(height*0.45))],[(0,int(height*0.55)),(width//2,height)],[(width//2,int(height*0.55)),(width,height)]])
    
    
    image0 = np.copy(imagem_teclado)
    image0 = cv2.rectangle(image0,ret_cords[0][0] ,ret_cords[0][1] , (Dcolor*seriesCount[0],Dcolor*seriesCount[0],Dcolor*seriesCount[0]), 3)
    
    image1 = np.copy(imagem_teclado)
    image1 = cv2.rectangle(image1,ret_cords[1][0] ,ret_cords[1][1] , (Dcolor*seriesCount[1],Dcolor*seriesCount[1],Dcolor*seriesCount[1]), 3)
    
    image2 = np.copy(imagem_teclado)
    image2 = cv2.rectangle(image2,ret_cords[2][0] ,ret_cords[2][1] , (Dcolor*seriesCount[2],Dcolor*seriesCount[2],Dcolor*seriesCount[2]), 3)
    
    image3 = np.copy(imagem_teclado)
    image3 = cv2.rectangle(image3,ret_cords[3][0] ,ret_cords[3][1] , (Dcolor*seriesCount[3],Dcolor*seriesCount[3],Dcolor*seriesCount[3]), 3)
    
    image = np.maximum(image0,image1)
    image = np.maximum(image,image2)
    image = np.maximum(image,image3)
    
    imagem_teclado = np.maximum(imagem_teclado,image)
    imagem_teclado = cv2.rectangle(imagem_teclado,ret_cords[maxpos][0] ,ret_cords[maxpos][1] , (0,Dcolor*seriesCount[maxpos],0), int(seriesCount[maxpos]))
    
    img1 = imagem_teclado.copy()
    img1 = adiciona_texto(img1,texto,monitor.width,monitor.height)
    
    imgCircle = cv2.circle(img1, (int(pos_est[0][0]),int(pos_est[0][1])), R*2, cor, 2)
    cv2.namedWindow('IcaroCount', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('IcaroCount',cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
    cv2.imshow('IcaroCount', img1)
    cv2.waitKey(30) 
    






cap.release()









