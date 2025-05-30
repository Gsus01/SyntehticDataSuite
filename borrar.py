import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('default')

# Cargar datos originales y sintéticos
df_real = pd.read_csv('data/train_FD001.csv')
df_synth = pd.read_csv('data/synthetic_output_my_hmm_test.csv')

# Seleccionar solo las columnas numéricas compartidas
cols_real = [col for col in df_real.columns if col.startswith('col')]
cols_synth = [col for col in df_synth.columns if col.startswith('col')]
cols_common = [col for col in cols_real if col in cols_synth]

if not cols_common:
    raise ValueError("No hay columnas comunes para comparar.")

# Graficar cada columna en una figura separada
for col in cols_common:
    plt.figure(figsize=(12, 5))
    # Recortar para igualar longitud si es necesario
    n = min(len(df_real[col]), len(df_synth[col]))
    plt.plot(range(n), df_real[col].values[:n], label='Original', linewidth=1.5)
    plt.plot(range(n), df_synth[col].values[:n], label='Sintético', linewidth=1.5, linestyle='--')
    plt.title(f'Comparación de la serie temporal: {col}', fontsize=14, pad=15)
    plt.xlabel('Índice de tiempo', fontsize=12)
    plt.ylabel('Valor', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    outname = f'comparacion_{col}.png'
    plt.savefig(outname, dpi=300, bbox_inches='tight')
    print(f"✅ Gráfica guardada como '{outname}'")
    plt.close()
