from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import cv2
import numpy as np
import joblib
import os
from io import BytesIO

# Inicializa o aplicativo FastAPI
app = FastAPI(
    title="Liquid Analyzer API",
    description="API para análise de líquidos usando visão computacional",
    version="1.0.0"
)

# Configuração do CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modelo para a resposta da API
class PredictionResult(BaseModel):
    status: str
    message: str
    prediction: Optional[str] = None
    confidence: Optional[float] = None

# Função para extrair características de cor (deve ser idêntica à usada no treinamento)
def extract_color_features(image):
    """
    Extrai características de cor (média de H, S, V) da região de interesse (ROI) central da imagem.
    
    Args:
        image: Imagem no formato OpenCV (BGR)
        
    Returns:
        np.array: Array com as médias de H, S, V da ROI
    """
    # Converte para o espaço de cores HSV
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
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

def extract_color_features_from_bytes(image_bytes: bytes) -> np.ndarray:
    """
    Extrai características de cor a partir dos bytes de uma imagem.
    
    Args:
        image_bytes: Bytes da imagem
        
    Returns:
        np.array: Array com as características de cor [H, S, V]
        
    Raises:
        ValueError: Se a imagem não puder ser processada
    """
    try:
        # Converte bytes para array NumPy
        nparr = np.frombuffer(image_bytes, np.uint8)
        # Decodifica a imagem
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("Não foi possível decodificar a imagem")
            
        # Extrai as características de cor
        features = extract_color_features(img)
        return features
        
    except Exception as e:
        raise ValueError(f"Erro ao processar a imagem: {str(e)}")

# Carrega o modelo ao iniciar a aplicação
model = None
try:
    model = joblib.load('liquid_classifier_model.pkl')
    print("Modelo carregado com sucesso!")
except FileNotFoundError:
    print("AVISO: Modelo não encontrado. Por favor, execute train_model.py primeiro.")
    model = None
except Exception as e:
    print(f"Erro ao carregar o modelo: {e}")
    model = None

# Rota raiz
@app.get("/", response_model=dict)
async def root():
    """
    Rota de teste para verificar se a API está funcionando.
    """
    return {
        "status": "success",
        "message": "API de Análise de Líquidos funcionando!"
    }

# Rota para análise de imagem
@app.post("/analyze_liquid/", response_model=PredictionResult)
async def analyze_liquid(file: UploadFile = File(...)):
    """
    Analisa uma imagem de líquido e retorna se foi aprovado ou reprovado.
    
    Args:
        file: Arquivo de imagem a ser analisado
        
    Returns:
        PredictionResult: Resultado da análise
    """
    # Verifica se o modelo foi carregado
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modelo não disponível. Por favor, treine o modelo primeiro."
        )
    
    # Verifica se o arquivo é uma imagem
    if not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="O arquivo enviado não é uma imagem."
        )
    
    try:
        # Lê os bytes da imagem
        image_bytes = await file.read()
        
        # Extrai as características da imagem
        features = extract_color_features_from_bytes(image_bytes)
        
        # Faz a predição e obtém as probabilidades
        probabilities = model.predict_proba([features])[0]
        prediction = np.argmax(probabilities)
        confidence = np.max(probabilities)
        
        # Converte a predição para texto
        prediction_text = "Aprovado" if prediction == 1 else "Reprovado"
        
        return {
            "status": "success",
            "message": "Análise concluída com sucesso",
            "prediction": prediction_text,
            "confidence": round(confidence, 4) # Arredonda para 4 casas decimais
        }
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Erro ao processar a imagem: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro interno ao processar a requisição: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
