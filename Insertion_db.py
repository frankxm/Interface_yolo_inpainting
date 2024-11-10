import sqlite3

def existe_splitimages(imgname,bigimg):
    conn = sqlite3.connect(
        r"./BDavion.db")
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM ImageAvant WHERE Etat = ? AND emplacement LIKE ?", (1, '%' + imgname + '%.png'))

    result = cursor.fetchone()
    if bigimg and result[0]>1:
        return True
    elif not bigimg and result[0]==1:
        return True
    else:
        return False


def update_image_apres(imgname,path,img,flag):
    conn = sqlite3.connect(
        r"./BDavion.db")
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM ImageApres WHERE nom_fichier = ?", (imgname,))
    result = cursor.fetchone()
    if result[0]==1:
        cursor.execute('''SELECT id_ImageApres, emplacement FROM ImageApres WHERE nom_fichier = ?''', (imgname,))
        records_to_update = cursor.fetchall()
        if records_to_update is not None:
            cursor.execute('''UPDATE ImageApres SET emplacement = ? , Etat = ? WHERE nom_fichier = ?''',
                           (path, 1, imgname))
            if flag=='inpainting':
                cursor.execute('''UPDATE Avion SET Etat = ? WHERE nom_fichier = ?''', (0, imgname))
    elif result[0]==0:
        cursor.execute("SELECT id_ImageApres FROM ImageApres ORDER BY id_ImageApres DESC LIMIT 1")
        last_imgapres = cursor.fetchone()
        if last_imgapres is not None:
            last_imgapres = last_imgapres[0]
        else:
            last_imgapres = 0
            print("imageapres结果集为空")
        cursor.execute(
            '''INSERT INTO ImageApres (id_ImageApres,nom_fichier, emplacement,Etat,longueur, largeur)  VALUES (?, ?, ?, ?, ?,?)''',
            (last_imgapres + 1, imgname, path, 1, img.shape[1], img.shape[0]))

    conn.commit()
    cursor.close()
    conn.close()
def update_etat(parent_path):
    conn = sqlite3.connect(
        r"./BDavion.db")
    cursor = conn.cursor()

    cursor.execute('''
        UPDATE ImageAvant
        SET Etat = 0
        WHERE emplacement LIKE ?
    ''', ('%' + parent_path + '%',))

    cursor.execute('''
            UPDATE ImageApres
            SET Etat = 0
            WHERE emplacement LIKE ?
        ''', ('%' + parent_path + '%',))


    conn.commit()
    cursor.close()
    conn.close()
def insertion_database(outpath, subimg, subimgname):
    conn = sqlite3.connect(
        r"./BDavion.db")
    cursor = conn.cursor()

    cursor.execute("SELECT id_ImageAvant FROM ImageAvant ORDER BY id_ImageAvant DESC LIMIT 1")
    last_imgavant = cursor.fetchone()
    if last_imgavant is not None:
        last_imgavant = last_imgavant[0]
    else:
        last_imgavant = 0
        print("imageavant结果集为空")

    cursor.execute("SELECT id_ImageApres FROM ImageApres ORDER BY id_ImageApres DESC LIMIT 1")
    last_imgapres = cursor.fetchone()
    if last_imgapres is not None:
        last_imgapres = last_imgapres[0]
    else:
        last_imgapres = 0
        print("imageapres结果集为空")

    cursor.execute("SELECT COUNT(*) FROM ImageAvant WHERE nom_fichier = ?", (subimgname,))
    result = cursor.fetchone()
    if result[0] == 1:
        cursor.execute(
            ''' UPDATE ImageAvant SET emplacement = ?, Etat = ?, longueur = ?, largeur = ? WHERE nom_fichier = ? ''',
            (outpath, 1, subimg.shape[1], subimg.shape[0], subimgname))
    elif result[0] == 0:
        cursor.execute(
            '''INSERT INTO ImageAvant (id_ImageAvant,nom_fichier, emplacement,Etat,longueur, largeur) VALUES (?, ?, ?, ?, ?,?)''',
            (last_imgavant + 1, subimgname, outpath, 1, subimg.shape[1], subimg.shape[0]))

    cursor.execute("SELECT COUNT(*) FROM ImageApres WHERE nom_fichier = ?", (subimgname,))
    result = cursor.fetchone()
    if result[0] == 0:
        cursor.execute(
            '''INSERT INTO ImageApres (id_ImageApres,nom_fichier, emplacement,Etat,longueur, largeur)  VALUES (?, ?, ?, ?, ?,?)''',
            (last_imgapres + 1, subimgname, '', 0, subimg.shape[1], subimg.shape[0]))
    elif result[0]==1:
        cursor.execute(
            ''' UPDATE ImageApres SET emplacement = ?, Etat = ?, longueur = ?, largeur = ? WHERE nom_fichier = ? ''',
            ('', 0, subimg.shape[1], subimg.shape[0], subimgname))

    conn.commit()
    cursor.close()
    conn.close()