import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

def extract_color_features(image_path):
    """
    Extrai características de cor (média de H, S, V) da região de interesse (ROI) central da imagem.
    
    Args:
        image_path (str): Caminho para o arquivo de imagem
        
    Returns:
        np.array: Array com as médias de H, S, V da ROI ou None se a imagem não puder ser carregada
    """
    # Carrega a imagem
    img = cv2.imread(image_path)
    if img is None:
        print(f"Erro: Não foi possível carregar a imagem {image_path}")
        return None
    
    # Converte para o espaço de cores HSV
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Define a ROI (Região de Interesse) com margens específicas
    height, width = hsv_img.shape[:2]
    start_h, end_h = int(height * 0.38), int(height * 0.69) # 38% do topo, 31% da base (100-31=69)
    start_w, end_w = int(width * 0.41), int(width * 0.59)  # 41% da esquerda, 41% da direita (100-41=59)
    
    # Extrai a ROI
    roi = hsv_img[start_h:end_h, start_w:end_w]
    
    # Calcula a média dos canais H, S, V
    mean_h = np.mean(roi[:, :, 0])  # Matiz (Hue)
    mean_s = np.mean(roi[:, :, 1])  # Saturação (Saturation)
    mean_v = np.mean(roi[:, :, 2])  # Valor (Value)
    
    return np.array([mean_h, mean_s, mean_v])

def load_dataset():
    """
    Carrega o dataset de imagens e extrai características de cor.
    
    Returns:
        tuple: (X, y) onde X são as características e y são os rótulos
    """
    X = []
    y = []
    
    # Processa imagens aprovadas (rótulo 1)
    aprovado_dir = os.path.join('dataset', 'aprovado')
    if os.path.exists(aprovado_dir):
        for filename in os.listdir(aprovado_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(aprovado_dir, filename)
                features = extract_color_features(img_path)
                if features is not None:
                    X.append(features)
                    y.append(1)  # Rótulo para aprovado
    
    # Processa imagens reprovadas (rótulo 0)
    reprovado_dir = os.path.join('dataset', 'reprovado')
    if os.path.exists(reprovado_dir):
        for filename in os.listdir(reprovado_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(reprovado_dir, filename)
                features = extract_color_features(img_path)
                if features is not None:
                    X.append(features)
                    y.append(0)  # Rótulo para reprovado
    
    # Converte para arrays numpy
    X = np.array(X)
    y = np.array(y)
    
    return X, y

def train_model():
    """
    Função principal para treinar o modelo de classificação.
    """
    print("Iniciando treinamento do modelo...")
    
    # Carrega o dataset
    print("Carregando e processando imagens...")
    X, y = load_dataset()
    
    # Verifica se há dados suficientes
    if len(X) == 0 or len(y) == 0:
        raise ValueError("Nenhuma imagem válida encontrada para treinamento. "
                         "Certifique-se de adicionar imagens nas pastas 'aprovado' e 'reprovado'.")
    
    print(f"Total de amostras carregadas: {len(X)}")
    print(f"Características por amostra: {X.shape[1]}")
    
    # Divide os dados em conjuntos de treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nDivisão dos dados:")
    print(f"- Treino: {len(X_train)} amostras")
    print(f"- Teste: {len(X_test)} amostras")
    
    # Cria e treina o modelo
    print("\nTreinando o modelo RandomForest...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Avalia o modelo
    print("\nAvaliando o modelo...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("\nMétricas de desempenho:")
    print(f"Acurácia: {accuracy:.4f}")
    print("\nRelatório de classificação:")
    print(classification_report(y_test, y_pred, target_names=['Reprovado', 'Aprovado']))
    
    # Salva o modelo treinado
    model_path = 'liquid_classifier_model.pkl'
    joblib.dump(model, model_path)
    print(f"\nModelo salvo em: {os.path.abspath(model_path)}")
    
    return model

if __name__ == "__main__":
    train_model()
