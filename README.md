# ALScom
Teclado controlado por Rastreamento Ocular usando imagens capturadas por Webcam: Um sistema que permite a digitação sem o uso das mãos utilizando os movimentos oculares capturados por uma webcam.

- [Guia de Instalação para o Projeto](#guia-de-instalação-para-o-projeto)
- [Guia de Uso do Sistema](#guia-de-uso-do-sistema)
- [Licença](#licença)

## Guia de Instalação para o Projeto:
1. Instalar Python:
- Faça o download da versão mais recente do Python 3.x no site oficial: 
  https://www.python.org/downloads/
- Execute o instalador e siga as instruções para instalar o Python.
    Certifique-se de marcar a opção "Adicionar Python 3.x ao PATH" durante a instalação.
   
2. Instalar Visual Studio com C++:
- Acesse o site do Visual Studio em: 
    https://visualstudio.microsoft.com/pt-br/vs/features/cplusplus/
- Baixe e instale o Visual Studio com suporte ao desenvolvimento em C++.

3. Instalar Cmake:
- Acesse o site do CMake em: https://cmake.org/download/
- Baixe e instale a versão mais recente do CMake compatível com o Windows.

4. Baixar arquivos do projeto ALScom:
- Acesse o repositório do projeto ALScom no GitHub: 
    https://github.com/dimerism/ALScom
- Clique no botão "Code" e selecione "Download ZIP" para baixar o arquivo 
    compactado do projeto.
- Extraia o conteúdo do arquivo ZIP para uma pasta de sua escolha no seu computador.

5. Baixar outros arquivos necessários:
- Acesse o link para a pasta no Google Drive contendo os arquivos adicionais: 
    https://drive.google.com/drive/folders/1CwLl5zjqQekUZnWPppVCD6DDtDbXzvrk
- Na pasta, faça o download dos seguintes arquivos:
  - dlib-19.24.2.dist-info
  - dlib
  - _dlib_pybind11.cp311-win_amd64.pyd
- Salve esses arquivos na seguinte pasta: \Python\Python311\Lib\site-package
- Faça o download do arquivo shape_predictor_68_face_landmarks.dat usando o link 
    fornecido.
- Salve o arquivo na pasta do projeto ALScom (mesma pasta onde você extraiu os arquivos do
    repositório do GitHub).
   
6. Instalação das demais dependências:
- Abra o menu Iniciar do Windows e pesquise por "cmd".
- Abra o prompt de comando (CMD).
- Navegue para a pasta do projeto ALScom usando o comando "cd 
    caminho_da_pasta_do_projeto".
- Execute o seguinte comando para instalar as dependências listadas no arquivo 
    "requirements.txt":
    py -m pip install -r requirements.txt
   
Pronto! Agora você instalou todas as dependências necessárias e configurou o ambiente para utilizar
o projeto ALScom. Lembre-se de seguir as instruções específicas do projeto para executá-lo 

## Guia de Uso do Sistema
Recomendações:
1. Certifique-se de que a distância entre o rosto e a câmera/tela seja de aproximadamente 60 
  cm.
2. Garanta que haja apenas um rosto na imagem capturada.
3. Posicione-se de frente para a fonte de luz de forma que ilumine adequadamente o rosto.
   
  Ideal:

  ![Ideal](https://i.imgur.com/vRNb2bK.png)
  
  Não Ideal:
  
  ![Não Ideal](https://i.imgur.com/CwjAkSq.png)
  
4. Mantenha a face sem rotação e voltada diretamente para a câmera.
5. Durante o uso do sistema, mantenha a posição da face estável.
   
Fase de Calibração:
1. Captura de imagens de olhos fechados:
- Inicie o sistema e aguarde a tela de calibração aparecer.
- Siga as orientações de posicionamento do rosto para garantir uma calibração precisa.
- Na tela de calibração, siga as instruções para fechar os olhos quando solicitado.
- Um sinal sonoro será emitido ao iniciar e ao finalizar a captura de imagens.

2. Captura da Imagem dos Olhos Direcionada a Pontos da Tela:
- Olhe para os pontos vermelhos impressos na tela conforme indicado pelo sistema.
- O sistema capturará imagens dos olhos direcionados a esses pontos para a calibração.

Fase de Uso:
1. Selecionando o Símbolo Desejado:
- Após a calibração, a tela de uso será aberta automaticamente.
- Olhe para a região que contém o símbolo desejado até que o perímetro desta região esteja 
    em um tom verde intenso.
- Feche os olhos até ouvir o sinal sonoro.
    Inicialmente, essa região pode conter outros símbolos além do desejado. Continue repetindo o 
    processo de olhar para a região e fechar os olhos até que reste apenas o símbolo desejado. O 
    símbolo selecionado será então impresso no campo de digitação.
 2. Correção de Seleção Incorreta:
    Caso uma região tenha sido selecionada incorretamente e o símbolo desejado não tenha sido 
    impresso no campo de digitação, selecione uma região que não contenha nenhum símbolo para 
    retornar à tela inicial.
 3. Caracteres adicionais:
- Se um símbolo indesejado tiver sido selecionado, selecione o símbolo " < " para 
    apagá-lo e retornar à seleção de símbolos.
- Para digitar um espaço, selecione o símbolo " _ ". 
    Lembre-se de manter a calma e praticar o uso do sistema para obter o melhor desempenho e 
    precisão. O sistema foi projetado para auxiliar na comunicação e facilitar a interação, tornando-o 
    mais eficiente à medida que você se familiariza com seu funcionamento

## Licença
  Este projeto é distribuído sob a Licença Pública Geral GNU v2.0.
