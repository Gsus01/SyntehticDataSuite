#!/usr/bin/env python3
"""
Test directo del modelo HMM sin flujos de Prefect
"""

import pandas as pd
from src.models.hmm_model import HMMWrapper

def test_hmm_direct():
    """
    Prueba el modelo HMM directamente con train_FD001.csv
    """
    
    print("🧪 Testing HMM model directly...")
    
    # Cargar datos
    data_path = "data/train_FD001.csv"
    print(f"📁 Loading data from: {data_path}")
    
    df = pd.read_csv(data_path)
    print(f"📊 Data shape: {df.shape}")
    print(f"📋 Columns: {list(df.columns)}")
    print(f"🔢 Data types:\n{df.dtypes}")
    
    # Configuración del modelo HMM
    config = {
        "model_type": "hmm",
        "hmm_columns_to_use": ["col3", "col4", "col5", "col6", "col7"],
        "hmm_n_states": 3,
        "hmm_covariance_type": "diag",
        "hmm_n_iter": 100,
        "hmm_random_state": 42,
    }
    
    print(f"🔧 HMM config: {config}")
    
    # Crear y entrenar modelo
    model_path = "test_hmm_model.pkl"
    hmm_model = HMMWrapper(
        model_name="test_hmm",
        model_path=model_path,
        config=config
    )
    
    try:
        print("\n🏋️ Training HMM model...")
        hmm_model.train(df)
        print("✅ Training completed successfully!")
        
        # Guardar modelo
        saved_path = hmm_model.save_model()
        print(f"📦 Model saved to: {saved_path}")
        
        # Generar datos sintéticos
        print("\n🧠 Generating synthetic data...")
        synthetic_data = hmm_model.sample(100)
        print(f"✅ Generated {len(synthetic_data)} synthetic samples")
        print(f"📊 Synthetic data columns: {list(synthetic_data.columns)}")
        print("\n📈 First 5 synthetic samples:")
        print(synthetic_data.head())
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_hmm_direct()
    if success:
        print("\n🎉 HMM test completed successfully!")
    else:
        print("\n�� HMM test failed!") 