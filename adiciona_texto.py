from PIL import ImageFont, ImageDraw, Image
import cv2
import numpy as np
from screeninfo import get_monitors
import string

def adiciona_texto(image,texto,Width,Height):
    

    #monitores = get_monitors()
    #monitor = monitores[0]
    
    width = int(Width)
    height = int(Height)
   

   
    pos1 = (0,int(height*0.45))
    pos2 = (int(width),int(height*0.6))
    #image = cv2.rectangle(image, pos1, pos2, (0,0,0), -1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image)

    # Draw non-ascii text onto image
    font = ImageFont.truetype("C:\Windows\Fonts\\arial.ttf", 35)
    draw = ImageDraw.Draw(pil_image)
    #draw.text((30, 30), "ԱԲԳԴԵԶԷԸԹԺԻԼԽԾԿՀՁՂՃՄՅՆՇՈՉՊՋՌՍՎՏՐՑՓՔՖ", font=font)

    if np.size(texto)>1:
        textoString = ''
        for n in texto:
            textoString = textoString + n
        draw.text(pos1, textoString, font=font,fill=(155, 155, 155))

    # Convert back to Numpy array and switch back from RGB to BGR
    image = np.asarray(pil_image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    return image;
    #cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
    #cv2.setWindowProperty("window",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
    #cv2.imshow("window", image)
    #cv2.imshow('image', image)
    #cv2.waitKey(0)
