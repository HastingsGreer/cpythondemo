import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("/d/python_extension/")
import spam

import mpmath

minx, miny, maxx, maxy = map(mpmath.mpf, (-2, -2, 2, 2))
mpmath.mp.prec = 890


it = 8000
import spam
import cv2
import numpy as np
import random

def p_mandel(minx, miny, maxx, maxy, it, updateR=True):
    

    #minx, miny, maxx, maxy = (-0.8447610481188477, -0.13546062282738092, -0.844761048118734, -0.13546062282726723)



    global bestcx
    global bestcy
    global bestz
    if updateR:
        bestdepth = 0
        
        bestz = []
        for __ in range(1):
            print(".", end="")
            if __ == 0:
                c1 =.5
                c2 = .5
            else:
                c1 = random.random()
                c2 = random.random()
            centerx, centery = (c1 * maxx + (1 - c1) * minx), (c2 * maxy + (1 - c2) * miny)

            z_real_array = [0]
            z_imag_array = [0]
            z_real = 0
            z_complex = 0
            i = 0
            while i < it and z_real**2 + z_complex**2 < it:
                z_real, z_complex = z_real**2 - z_complex**2 + centerx, 2 * z_real * z_complex + centery
                z_real_array.append(float(z_real))
                z_imag_array.append(float(z_complex))
                i += 1
            if i > bestdepth:
                bestdepth = i
                bestz = z_real_array, z_imag_array
                bestcx, bestcy = centerx, centery
            if i == it:
                break
        print("found c")
        import sys
        sys.stdout.flush()
    print("span", bestcy, maxy - bestcy)
    y, x = np.mgrid[float(miny - bestcy):float(maxy - bestcy):512j, float(minx-bestcx):float(maxx-bestcx):512j]
    print(y)
    print(x.dtype)
    z = y.astype(np.int32) * 0
    ref_real_array = np.array(bestz[0])
    ref_imag_array = np.array(bestz[1])
    
    print(repr((minx, miny, maxx, maxy)))

    spam.perturb_mandel(ref_real_array, ref_imag_array, x, y, z, it)

    return z
import matplotlib






import numpy as np
import matplotlib.pyplot as plt
import itertools

class Settings:
    def __init__(self, depth, scale, dim, center):
        self.depth = depth
        self.scale = scale
        self.dim = dim
        self.center = center
        
    def changemethod(self):
        self.method = self.methods.next()

def main():
    #initial settings

    centerx = mpmath.mpf(" -1.756882996047415507177544263880044205560240816681939504337701122205524205232602173448341712703733871053600384797384909496400768301564935457122130644767462662664858651590631630737324402579324531143018225431629992886257332217126295")
    centery = mpmath.mpf("0.012178698364084925222456470314532744312959953422670766200219117572544897165197364251729676017200851973935584329711981645307984466766116936909884995575928407038962826724678263924301445881502867259057819318295270283982300902903242")

    xmin = mpmath.mpf(-2) + centerx
    xmax = mpmath.mpf(2) + centerx
    ymin = mpmath.mpf(-2) + centery
    ymax = mpmath.mpf(2) + centery
     
    depth = mpmath.mpf(60)   
    scale = mpmath.mpf(2)
    dim = 600
    settings = Settings(depth, scale, dim, (centerx, centery))  
                                            # the settings object is used to keep track of
                                            # rendering settings as the click-generated 
                                            # callbacks change the view window
        
    counts = p_mandel(xmin, ymin, xmax, ymax, settings.depth)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    picture = ax.imshow(counts, extent = [-1, 1, -1, 1], interpolation = 'nearest')
    colorbar = fig.colorbar(picture, ax = ax)
    cid = fig.canvas.mpl_connect('button_press_event', lambda e: onclick(e, ax, colorbar, settings))
    cid = fig.canvas.mpl_connect('key_press_event', lambda e: onkey(e, ax, colorbar, settings))
    plt.show()
    
    
def onclick(event, ax, colorbar, settings):
                     #zoom in by a factor of two with every click
    settings.center = (event.xdata * settings.scale + settings.center[0], -event.ydata * settings.scale + settings.center[1])
    settings.scale *= .1 
    render(ax, colorbar, settings)
    
    
def onkey(event, ax, colorbar, settings):
    key = event.key
    
    if key == "1":
        settings.depth = int(settings.depth * 2)
    elif key == "2":
        settings.depth = int(settings.depth / 2)
    elif key == "e":
        settings.dim = int(settings.dim * 2)
    elif key == "d":
        settings.dim = int(settings.dim / 2)
    elif key == "w":
        settings.scale = settings.scale * 2
        render(ax, colorbar, settings, updateR=False)
    elif key == "x":
        settings.scale = settings.scale / 15
        render(ax, colorbar, settings, updateR=False)

    elif key == "r":
        settings.changemethod()
        print(settings.method)
    else:
        return
    render(ax, colorbar, settings)

def render(ax, colorbar, settings, updateR = True):
    print(str(settings.dim), str(settings.depth))
    ax.clear()
    xmin = settings.center[0] - settings.scale
    xmax = settings.center[0] + settings.scale
    ymin = settings.center[1] - settings.scale
    ymax = settings.center[1] + settings.scale
    counts = p_mandel(xmin, ymin, xmax, ymax, settings.depth, updateR)
    
    cax = ax.imshow(counts, extent = [-1, 1, -1, 1])
    colorbar.on_mappable_changed(cax)
    ax.figure.canvas.draw()
    
main()