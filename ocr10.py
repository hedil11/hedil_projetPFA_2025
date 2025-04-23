#Ce module est utilisé pour interagir avec le système de fichiers, 
# notamment pour vérifier l'existence de dossiers et créer des dossiers.
import os  

#Ce module fait partie de la bibliothèque OpenCV et est utilisé pour manipuler des images, 
# les lire, les afficher et les traiter.
import cv2  

#Ce module permet de lire et d'écrire des fichiers CSV
import csv  

import numpy as np  

#Ce module permet de générer des fichiers XML
import xml.etree.ElementTree as ET

# Créer les dossiers si ils n'existent pas
if not os.path.exists('annotated_images2'): # Vérifie si un dossier existe déjà.
    os.makedirs('annotated_images2') #Crée les dossiers spécifiés si ils n'existent pas.
if not os.path.exists('bounding_boxes'):
    os.makedirs('bounding_boxes')
if not os.path.exists('annotations_xml'):
    os.makedirs('annotations_xml')  # Crée un dossier pour stocker les fichiers XML

# Fonction pour redimensionner l'image pour un affichage optimal
#Cette fonction redimensionne une image pour qu'elle soit affichée 
# sans dépasser une largeur ou une hauteur maximales.
def resize_image_for_display(img, max_width=800, max_height=600):
    # Obtenir les dimensions actuelles de l'image
    h, w = img.shape[:2] #Donne les dimensions de l'image, ici h (hauteur) et w (largeur).
    # Calculer les nouveaux rapports de redimensionnement     
    aspect_ratio = w / h 
    if w > max_width: # Cette condition vérifie si
        # la largeur de l'image dépasse la largeur maximale autorisée (max_width, ici 800)        
        w = max_width #La largeur de l'image est réduite à la valeur maximale autorisée     
        h = int(w / aspect_ratio) # aspect_ratio =w/h => h =w/aspect_ratio
    if h > max_height: #Cette condition vérifie si 
        #la hauteur de l'image dépasse la hauteur maximale autorisée (max_height, ici 600)        
        h = max_height
        w = int(h * aspect_ratio) #aspect_ratio =w/h => w= aspect_ratio*h 
    # Redimensionner l'image
    resized_img = cv2.resize(img, (w, h))
    return resized_img

# Fonction pour dessiner plusieurs bounding boxes et enregistrer les annotations
def annotate_image(image_path, output_image_path, boxes_file):
    img = cv2.imread(image_path) # Charge l'image depuis le chemin spécifié.     
    if img is None:
        print(f"Erreur de lecture de l'image {image_path}. Elle pourrait être corrompue ou dans un format non pris en charge.")
        return

    clone = img.copy() #Crée une copie de l'image originale pour éviter de modifier l'original. 
    resized_img = resize_image_for_display(img) # Redimensionner l'image pour affichage

    # Liste pour stocker toutes les boîtes sélectionnées
    rois = []

    while True:
        # Sélection d'une ROI (Region Of Interest)
        r = cv2.selectROI("Image", resized_img, fromCenter=False, showCrosshair=True)
        cv2.destroyAllWindows()

        if r == (0, 0, 0, 0):
            # Si aucune boîte n'est sélectionnée, on arrête
            break

        # Ajouter la ROI à la liste
        rois.append(r)

        # Dessiner la boîte sur la copie de l'image
        x1, y1, w, h = r
        x2, y2 = x1 + w, y1 + h
        cv2.rectangle(clone, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Afficher l'image annotée avec les boîtes jusqu'à maintenant
        preview = resize_image_for_display(clone)
        cv2.imshow("Image", preview)
        print("Appuie sur une touche pour sélectionner une autre boîte ou fermer la fenêtre si terminé.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if not rois:
        print(f"Aucune ROI sélectionnée pour {image_path}.")
        return

    # Sauvegarder l'image annotée
    cv2.imwrite(output_image_path, clone)

    base_name = os.path.splitext(os.path.basename(image_path))[0]

    # Enregistrement des coordonnées de toutes les boîtes dans un fichier
    box_file_path = os.path.join('bounding_boxes', f'{base_name}_boxes.txt')
    with open(box_file_path, 'w') as box_file:
        for (x1, y1, w, h) in rois:
            x2, y2 = x1 + w, y1 + h
            box_file.write(f'{x1},{y1},{x2},{y2}\n')

    print(f"{len(rois)} annotation(s) sauvegardée(s) pour {image_path}.")

# Fonction pour traiter toutes les images dans un dossier
def process_images(input_folder):
    # Vérifier si le dossier existe et contient des images
    if not os.path.exists(input_folder):     
        print(f"Le dossier spécifié {input_folder} n'existe pas.")
        return

    image_files = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.png'))]

    if not image_files:
        print(f"Aucune image trouvée dans le dossier {input_folder}.")
        return

    # Traiter chaque image du dossier
    for img_file in image_files:
        img_path = os.path.join(input_folder, img_file)         
        print(f"Traitement de l'image {img_path}...")
        output_image_path = os.path.join('annotated_images2', img_file)
        annotate_image(img_path, output_image_path, 'bounding_boxes')         
        #Appelle la fonction pour annoter chaque image.

# Fonction pour générer un fichier XML à partir des fichiers de bounding boxes
def generate_xml_from_boxes(boxes_folder):
    for filename in os.listdir(boxes_folder):
        if filename.endswith('_boxes.txt'):
            base_name = filename.replace('_boxes.txt', '')
            image_filename = base_name + '.jpg'  # ou .png si nécessaire
            image_path = os.path.join('annotated_images2', image_filename)

            try:
                img = cv2.imread(image_path)
                h, w = img.shape[:2]
            except:
                h, w = 0, 0

            annotation = ET.Element('annotation')
            ET.SubElement(annotation, 'filename').text = image_filename
            size = ET.SubElement(annotation, 'size')
            ET.SubElement(size, 'width').text = str(w)
            ET.SubElement(size, 'height').text = str(h)
            ET.SubElement(size, 'depth').text = '3'

            # Lire toutes les boîtes dans le fichier
            with open(os.path.join(boxes_folder, filename), 'r') as f:
                lines = f.readlines()
                for line in lines:
                    x1, y1, x2, y2 = map(int, line.strip().split(','))
                    obj = ET.SubElement(annotation, 'object')
                    ET.SubElement(obj, 'name').text = 'license_plate'
                    bndbox = ET.SubElement(obj, 'bndbox')
                    ET.SubElement(bndbox, 'xmin').text = str(x1)
                    ET.SubElement(bndbox, 'ymin').text = str(y1)
                    ET.SubElement(bndbox, 'xmax').text = str(x2)
                    ET.SubElement(bndbox, 'ymax').text = str(y2)

            tree = ET.ElementTree(annotation)
            tree.write(os.path.join('annotations_xml', base_name + '.xml'))

# Utiliser le programme avec un dossier d'images
input_folder = r'C:\Users\hedil\OneDrive\Bureau\expl'  
process_images(input_folder)

# Générer les fichiers XML après les annotations
generate_xml_from_boxes('bounding_boxes')
