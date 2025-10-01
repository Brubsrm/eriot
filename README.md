# Liquid Analyzer API

API para análise de líquidos utilizando visão computacional e aprendizado de máquina.

## Estrutura do Projeto

```
liquid_analyzer_api/
├── dataset/
│   ├── aprovado/     # Imagens de líquidos aprovados
│   └── reprovado/    # Imagens de líquidos reprovados
├── main.py           # Código principal da API
├── train_model.py    # Script para treinar o modelo
└── README.md         # Este arquivo
```

## Pré-requisitos

- Python 3.8 ou superior
- pip (gerenciador de pacotes do Python)

## Configuração do Ambiente

1. **Crie e ative um ambiente virtual** (recomendado):

   ```bash
   # No Windows
   python -m venv venv
   .\venv\Scripts\activate
   ```

2. **Instale as dependências** necessárias:

   ```bash
   pip install fastapi uvicorn opencv-python scikit-learn joblib pillow
   ```

## Como Usar

### 1. Preparação dos Dados

- Adicione imagens de amostras de líquidos nas pastas correspondentes:
  - `dataset/aprovado/`: Imagens de líquidos aprovados
  - `dataset/reprovado/`: Imagens de líquidos reprovados

### 2. Treinamento do Modelo

Execute o script de treinamento:

```bash
python train_model.py
```

Isso irá:
- Processar as imagens do dataset
- Treinar um modelo de classificação
- Salvar o modelo treinado como `liquid_classifier_model.pkl`

### 3. Iniciar a API

Para iniciar o servidor da API, execute:

```bash
uvicorn main:app --reload
```

A API estará disponível em `http://127.0.0.1:8000`

### 4. Documentação da API

- **Documentação interativa (Swagger UI):** `http://127.0.0.1:8000/docs`
- **Documentação alternativa (ReDoc):** `http://127.0.0.1:8000/redoc`

## Endpoints

### Analisar Imagem

- **Método:** POST
- **URL:** `/analyze_liquid/`
- **Content-Type:** `multipart/form-data`
- **Parâmetro:** `file` (arquivo de imagem)

**Exemplo de resposta:**

```json
{
  "status": "success",
  "message": "Análise concluída com sucesso",
  "prediction": "Aprovado",
  "confidence": 0.9999
}
```

## Próximos Passos

- [ ] Coletar mais imagens para melhorar a precisão do modelo
- [ ] Implementar validação de entrada mais robusta
- [ ] Adicionar autenticação à API
- [ ] Implementar logs detalhados
- [ ] Criar testes automatizados

## Solução de Problemas

- **Erro ao carregar o modelo:** Verifique se o arquivo `liquid_classifier_model.pkl` existe e está no diretório correto.
- **Baixa acurácia:** Adicione mais imagens de treinamento e tente ajustar os parâmetros do modelo.
- **Erros de dependência:** Certifique-se de que todas as dependências foram instaladas corretamente.

## Licença

Este projeto está licenciado sob a licença MIT.
