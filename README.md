#Détection et suppression automatique d’avions dans les images satellites




#Programme exécutable pyqt5 packagé sous Windows
https://drive.google.com/drive/folders/1ddQhAtdIlmadTS-HKMH3VyulmMJ1E9gg?usp=sharing
#Programme exécutable pyqt5 packagé sous Linux
https://drive.google.com/drive/folders/1pnjM3ykEtG59gQHAUvd3X5KOT0qcq727?usp=sharing



#mainapp.py
Démarrage de l'interface pyqt5

#detect.py
Inférence à l'aide du modèle entrainé sur les images satellites (best.pt)

#Dossier
##models/
implique principalement la structure, la définition et l’exportation de modèles. C’est la partie centrale de YOLOv5 et est responsable de la construction et de l’exécution des modèles de réseaux neuronaux.
##utils/ 
contient des fonctions auxiliaires impliquant le traitement des données, l'évaluation, la visualisation, les fonctions d'outils, etc., pour aider à optimiser et à simplifier le processus de formation et d'inférence.
Elles sont essentielles pendant l'utilisation de detect.py

##
##inference/
###inference/output
Stocker l'image apres la prediction
###inference/output/labels
Stocker le ficher sortie de la prediction
#####exemple:
	aughv_ELLX__1__8640___3780 0.51 489.0 548.0 481 611 548 571 496 484 429 524 102.0 78.0 59.0 D:\python_pycharm\rotation-yolov5-master\mydatas\images\test\aughv_ELLX__1__8640___3780.png inference\output\aughv_ELLX__1__8640___3780.png 640 640
	
	Nom d'image+la confiance du rectangle prédit + le point central du rectangle prédit + les quatre sommets du rectangle prédit + la largeur et la hauteur du rectangle prédit + les informations d'angle du rectangle prédit + le chemin de l'image d'origine + le chemin prédit + la largeur et la hauteur de l'image
###inference/output/labels
Image du masque de sortie

##
##big_images/split_images
Si la taille de l'image à prédire est trop grande, on le coupe en les petites images et les enregistre ici

##
##icons/
Enregistrer les icones dont l'interface a besoin

##
##inpainting/
###inpainting/images
Images satellites prédites qui doivent être restaurées (si l'image d'entrée de prédiction est grande, plusieurs images après recadrage et prédiction sont stockées ici. Si l'image prédite est de taille appropriée, une seule image prédite est stockée ici.)
###inpainting/mask
Masques d’images satellites
###inpainting/result
Images satellites restaurée
###inpainting/combinediinpainting
Grande image après la fusion de plusieurs petites images satellite restaurées
###inpainting/combinedimages
Grande image après la fusion de plusieurs petites images satellites prédites
###inpainting/combinedmasques
Masque de grande image après la fusion de plusieurs petites images satellite prédites

