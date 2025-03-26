import laspy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# --- 1. Lecture des données LiDAR ---
# Remplacer le nom de fichier par votre fichier .laz
filename = "C:/Users/UsagerLocal/Desktop/GMQ580/TD/TD-2/Data/XEOS_LID_Victoriaville_2024_Classifiees_190-5102_25pts_L1_F07_QC.laz"
las = laspy.read(filename)

# Affichage du nombre total de points
total_points = len(las.x)
print(f"Nombre total de points lidar : {total_points}")

# Calcul de l'étendue spatiale
x_min, x_max = las.x.min(), las.x.max()
y_min, y_max = las.y.min(), las.y.max()
z_min, z_max = las.z.min(), las.z.max()
print(f"Étendue spatiale en NAD83 MTM 7 CSRS (EPSG : 2949):")
print(f"  X : [{x_min:.2f}, {x_max:.2f}]")
print(f"  Y : [{y_min:.2f}, {y_max:.2f}]")
print(f"  Z : [{z_min:.2f}, {z_max:.2f}]")

# --- 2. Filtrage des points de sol ---
# On considère que la classification du sol est codée par la valeur 2
ground_mask = (las.classification == 2)
x_ground = las.x[ground_mask]
y_ground = las.y[ground_mask]
z_ground = las.z[ground_mask]
print(f"Nombre de points de sol : {len(x_ground)}")

# --- 3. Regroupement des points en une grille 2D ---
# Définir la taille d'une cellule de grille (en unités des coordonnées, ex. mètres)
cell_size = 5.0

# Création des intervalles (bins) pour les axes X et Y
x_bins = np.arange(x_min, x_max + cell_size, cell_size)
y_bins = np.arange(y_min, y_max + cell_size, cell_size)

# Attribution de chaque point à une cellule
x_indices = np.digitize(x_ground, bins=x_bins) - 1  # indices 0-based
y_indices = np.digitize(y_ground, bins=y_bins) - 1

# Définir la taille de la grille
grid_shape = (len(x_bins)-1, len(y_bins)-1)
# Matrice booléenne pour marquer les cellules identifiées comme routes
road_grid = np.zeros(grid_shape, dtype=bool)
# Matrice pour compter le nombre de points par cellule
count_grid = np.zeros(grid_shape, dtype=int)
# Matrice pour stocker l'écart-type de z par cellule
std_grid = np.full(grid_shape, np.nan)

# Seuils pour l'heuristique de détection des routes
min_points = 5      # nombre minimum de points par cellule
std_threshold = 0.3 # écart-type maximum pour considérer la surface comme plate

# Itération sur les cellules de la grille
for i in range(grid_shape[0]):
    for j in range(grid_shape[1]):
        # Sélection des points dans la cellule (i, j)
        in_cell = (x_indices == i) & (y_indices == j)
        count = np.sum(in_cell)
        count_grid[i, j] = count
        if count > 0:
            # Calcul de l'écart-type en Z pour la cellule
            std_z = np.std(z_ground[in_cell])
            std_grid[i, j] = std_z
            # Si la cellule présente suffisamment de points et est plate, on la considère comme route
            if std_z < std_threshold and count >= min_points:
                road_grid[i, j] = True

# --- 4. Visualisation ---
# Préparation de l'image : chaque cellule sera colorée en rouge si route, sinon en gris.
color_image = np.ones((road_grid.shape[0], road_grid.shape[1], 3)) * 0.7  # gris clair
color_image[road_grid] = [1, 0, 0]  # rouge pour les routes

plt.figure(figsize=(10, 8))
# Utilisation de imshow pour afficher l'image. L'argument 'extent' permet de remettre l'échelle spatiale.
plt.imshow(color_image, extent=[x_min, x_max, y_min, y_max], origin='lower')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Détection des routes (cellules identifiées en rouge)')

# Création d'une légende simple
road_patch = mpatches.Patch(color='red', label='Route')
nonroad_patch = mpatches.Patch(color='gray', label='Non-route')
plt.legend(handles=[road_patch, nonroad_patch], loc='upper right')

# Sauvegarde de l'image
plt.savefig("routes.png", dpi=300)
plt.show()

# --- 5. Limitations ---

# Le code est fonctionnel mais la méthode de détection des routes est basée sur des statistiques.
# Elle ne prend pas en compte la topologie des routes, ni des intersections.
# La méthode est sensible à la densité de points et à la résolution de la grille.
# Alors, pour affiner la méthode il faudrait jouer avec les paramètres de la grille et
# les seuils de détection puisque présentement les résultats sont grossiers.
