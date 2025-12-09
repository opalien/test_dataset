import numpy as np

# Deux vecteurs orthogonaux (leur produit scalaire = 0)
v1 = np.array([1, 0, 0])
v2 = np.array([0, 1, 0])

# Vérifier qu'ils sont orthogonaux
dot_product = np.dot(v1, v2)
print(f'Vecteur 1: {v1}')
print(f'Vecteur 2: {v2}')
print(f'Produit scalaire (doit être 0 pour des vecteurs orthogonaux): {dot_product}')

# Addition des deux vecteurs
result = v1 + v2
print(f'Addition v1 + v2: {result}')
