import pandas as pd
import matplotlib.pyplot as plt

# Configurar el estilo
plt.style.use('default')

# Leer el archivo CSV
df = pd.read_csv('data/synthetic_output_my_gmm_experiment_01.csv')

# Crear la figura y los ejes
fig, ax = plt.subplots(figsize=(12, 6))

# Graficar las tres series temporales
ax.plot(df['col1'], label='Serie 1', linewidth=1)
ax.plot(df['col2'], label='Serie 2', linewidth=1)
ax.plot(df['col3'], label='Serie 3', linewidth=1)

# Personalizar la gráfica
ax.set_title('Series Temporales Sintéticas', fontsize=14, pad=15)
ax.set_xlabel('Tiempo', fontsize=12)
ax.set_ylabel('Valor', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Ajustar los márgenes
plt.tight_layout()

# Guardar la figura
plt.savefig('series_temporales.png', dpi=300, bbox_inches='tight')
print("✅ Gráfica guardada como 'series_temporales.png'")
