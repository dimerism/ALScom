from PIL import ImageFont, ImageDraw, Image
import cv2
import numpy as np
from screeninfo import get_monitors
import string
from quad_alphabet import *

def cria_teclado(alphabet,Width,Height):
    

    #monitores = get_monitors()
    #monitor = monitores[0]
    N = np.size(alphabet)
    width = int(Width)
    height = int(Height*0.95)
    #W = [[width//20, width//2 - 10],[width//2 + width//20,width-10],[width//20, width//2 - 10],[width//2 + width//20,width-10]]
    W = [[width//25, width//2 - 10],[width//2 + width//25,width-10],[width//25, width//2 - 10],[width//2 + width//25,width-10]]
    #H = [[height//4,height//2 - 10],[height//4,height//2 - 10],[height//2 + height//4, height-10],[height//2 + height//4, height-10]]
    H = [[height//4,height//2 - 10],[height//4,height//2 - 10],[height//2 + height//4, height-10],[height//2 + height//8, height-10]]
    dw = 60
    dh = 40

    #alphabet = list(string.ascii_uppercase)
   


    #Quad code
    
    Q = quad_alphabet(alphabet);
    
    # Create black mask using Numpy and convert from BGR (OpenCV) to RGB (PIL)
    # image = cv2.imread('1.png') # If you were using an actual image
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image)
    
    # Draw non-ascii text onto image
    font = ImageFont.truetype("C:\Windows\Fonts\\arial.ttf", 35)
    draw = ImageDraw.Draw(pil_image)
    #draw.text((30, 30), "ԱԲԳԴԵԶԷԸԹԺԻԼԽԾԿՀՁՂՃՄՅՆՇՈՉՊՋՌՍՎՏՐՑՓՔՖ", font=font)
    for n in range(N):
        if Q[n][0] < 3:
            draw.text((W[Q[n][0]][0]+dw*Q[n][1]+4*dw*Q[n][2],H[Q[n][0]][0] ), alphabet[n], font=font)
        else:
            draw.text((W[Q[n][0]][0],H[Q[n][0]][0]+dh*Q[n][1]+4*dh*Q[n][2] ), alphabet[n], font=font)
            
    # Convert back to Numpy array and switch back from RGB to BGR
    image = np.asarray(pil_image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.line(image, (width//2,0), (width//2,int(0.45*height)), (255,255,255), 1)
    cv2.line(image, (width//2,int(0.55*height)), (width//2,height), (255,255,255), 1)
    cv2.line(image, (0,int(0.45*height)), (width,int(0.45*height)), (255,255,255), 1)
    cv2.line(image, (0,int(0.55*height)), (width,int(0.55*height)), (255,255,255), 1)
    return image;
    #cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
    #cv2.setWindowProperty("window",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
    #cv2.imshow("window", image)
    #cv2.imshow('image', image)
    #cv2.waitKey(0)
