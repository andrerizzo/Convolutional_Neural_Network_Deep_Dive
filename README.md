> 🇧🇷 Este README está em português.  
> 🇺🇸 [Click here for the English version.](README_EN.md)

# 🧠 Estudo Profundo sobre Redes Neurais Convolucionais (CNN)

## 🔍 Visão Geral
Este projeto explora em profundidade os principais conceitos, camadas e operações envolvidas em **Redes Neurais Convolucionais (CNNs)**. A partir de um conjunto de imagens real (CIFAR-100), são implementadas redes neurais personalizadas com foco didático e prático.

É ideal para demonstrar conhecimento em visão computacional, construção de modelos do zero e boas práticas de treinamento.

---

## 🎯 Objetivos
- Entender a fundo as **camadas fundamentais de CNNs** (conv2d, pooling, batch norm, dropout, etc.)
- Implementar uma arquitetura **customizada** usando Keras
- Aplicar **data augmentation** e técnicas de regularização
- Avaliar o modelo com métricas, curvas e matriz de confusão

---

## 🧠 Conjunto de Dados
- 📚 **Base:** [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html)
- 🔢 **Formato:** 60.000 imagens 32x32 (50.000 treino + 10.000 teste)
- 🔍 **Tarefa:** Classificação de 100 categorias de objetos

---

## 🏗️ Arquitetura da Rede
A rede convolucional foi construída **do zero** com:
- Camadas `Conv2D` com diferentes tamanhos de filtro e strides
- Camadas `MaxPooling2D` e `Dropout` para controle de overfitting
- Normalização com `BatchNormalization`
- Camadas densas para decisão final
- Função de ativação `ReLU` + `Softmax` na saída

---

## 🧪 Treinamento e Avaliação
- **Função de perda:** `SparseCategoricalCrossentropy`
- **Otimizador:** `Adam`
- **Épocas:** 20
- **Técnicas adicionais:** Data Augmentation, EarlyStopping

### 🔍 Métricas de Avaliação
- Acurácia (treino e validação)
- Curva de aprendizado
- Matriz de confusão
- Acurácia por classe (top-1)

---

## 📦 Bibliotecas e Ferramentas
- TensorFlow / Keras
- Matplotlib / Seaborn
- NumPy / Pandas
- Google Colab (ambiente utilizado)

---

## 🔁 Possibilidades de Expansão
- Aplicação de **Transfer Learning** com redes como ResNet, EfficientNet
- Visualizações com **Grad-CAM**
- Exportação do modelo para uso em produção (API ou Mobile)
- Treinamento com outras bases (ex: FashionMNIST, Tiny ImageNet)

---

### 👨‍💻 Sobre o Autor

**André Rizzo**  
📊 Cientista de Dados Sênior | Estatístico | MBA em IA e Big Data (USP)  
🧠 Especialista em Deep Learning, Visão Computacional e Modelagem Preditiva  
📍 Rio de Janeiro, Brasil  

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Perfil-0077B5?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/andrerizzo1)
[![GitHub](https://img.shields.io/badge/GitHub-Portfólio-181717?logo=github&logoColor=white)](https://github.com/andrerizzo)
[![Email](https://img.shields.io/badge/Email-andrerizzo@hotmail.com-D14836?logo=gmail&logoColor=white)](mailto:andrerizzo@hotmail.com)

---

*Última atualização: 30/03/2025*
