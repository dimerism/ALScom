from PIL import ImageFont, ImageDraw, Image
import cv2
import numpy as np
from screeninfo import get_monitors
import string

def quad_alphabet(alphabet):
    
    N = np.size(alphabet)
    alphaQuad = np.zeros((N,3))
    Q = np.array([[0,0,0]])
    for n in range(N):
        
        q2 = n//(4**2)
        r2 = n - q2*(4**2)

        q1 = r2//4
        r1 = r2 - q1*4

        q0 = r1//1
        Q = np.append(Q,np.array([[q0,q1,q2]]),axis=0)
        
    Q = np.delete(Q,0,0)
    return Q
