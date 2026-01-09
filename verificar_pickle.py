import pickle

with open('data/processed/features.pkl', 'rb') as f:
    datos = pickle.load(f)

print(f"Shape de features: {datos['features'].shape}")
print(f"Labels únicos: {set(datos['labels'])}")
print(f"Número de features: {datos['n_features']}")
print(f"K usado: {datos['k']}")
print(f"Timestamp: {datos['timestamp']}")
print(f"\nPrimeros 5 nombres de features:")
for i, nombre in enumerate(datos['feature_names'][:5]):
    print(f"  {i}: {nombre}")