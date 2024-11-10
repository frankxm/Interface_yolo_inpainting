


def update(im, gradientX, gradientY, confiance, source, masque, dOmega, point, list, index, taillecadre,srcimg,left=None,up=None):
    p = dOmega[index]
    patch = Patch(im, taillecadre, p)
    x1, y1 = patch[0]
    px, py = point
    for (i, j) in list:
        if left and up is not None:
            srcimg[(up+y1 + i, left+x1 + j)] = im[(py + i, px + j)]
        im[y1+i, x1+j] = im[py+i, px+j]
        confiance[y1+i, x1+j] = confiance[py, px]
        source[y1+i, x1+j] = 1
        masque[y1+i, x1+j] = 0
    return(im, gradientX, gradientY, confiance, source, masque,srcimg)

def Patch(im, taillecadre, point):
    """
    Permet de calculer les deux points extreme du patch
    Voici le patch avec les 4 points
        1 _________ 2
          |        |
          |        |
         3|________|4
    """
    px, py = point
    xsize, ysize, c = im.shape
    x3 = max(px - taillecadre, 0)
    y3 = max(py - taillecadre, 0)
    x2 = min(px + taillecadre, ysize - 1)
    y2 = min(py + taillecadre, xsize - 1)
    return((x3, y3),(x2, y2))
