# Turtle Draw - Visão Computacional com ROS 2 e TurtleSim

## Descrição

Este projeto foi desenvolvido para a ponderada de Robótica e Visão Computacional.

O objetivo é criar uma pipeline de visão computacional capaz de ler uma imagem, processar seus contornos e fazer a tartaruga do TurtleSim desenhar uma representação da imagem na tela.

A imagem utilizada no projeto é um cachorro preto sobre fundo claro. A partir dela, o código realiza etapas de processamento, extrai pontos do contorno e converte esses pontos para coordenadas do TurtleSim.

---

## Tecnologias utilizadas

- Python
- ROS 2
- TurtleSim
- NumPy
- OpenCV
- Matplotlib
- Micromamba

Observação: o OpenCV foi utilizado apenas para carregar a imagem, respeitando a regra da atividade. O restante do processamento foi implementado manualmente usando NumPy.

---

## Estrutura do projeto

```text
ROS 2 Teste/
├── src/
│   └── turtle_draw/
│       ├── images/
│       │   └── dog.jpeg
│       ├── turtle_draw/
│       │   ├── __init__.py
│       │   └── turtle_controller.py
│       ├── package.xml
│       ├── setup.py
│       └── resource/
│           └── turtle_draw
├── build/
├── install/
├── log/
└── README.md
```

## Pipeline de visão computacional

A pipeline implementada segue estas etapas:

```text
imagem original
→ conversão para tons de cinza
→ filtro gaussiano
→ operador de Sobel
→ limiarização
→ operações morfológicas
→ extração de pontos
→ mapeamento para TurtleSim
→ desenho no TurtleSim
```

## Etapas principais

- 1. Conversão para tons de cinza: A imagem colorida é convertida para escala de cinza para reduzir a complexidade do processamento. Assim, em vez de trabalhar com três canais de cor, o código trabalha com apenas uma matriz de intensidade.

- 2. Filtro gaussiano: O filtro gaussiano é usado para suavizar a imagem e reduzir ruídos antes da detecção de bordas.

- 3. Sobel: O operador de Sobel é usado para identificar regiões de grande variação de intensidade, ou seja, bordas.

- 4. Limiarização: A limiarização transforma a imagem em preto e branco. No caso deste projeto, ela também foi adaptada para selecionar regiões escuras, já que o cachorro é preto e o fundo é claro.

- 5. Operações morfológicas: Foram implementadas operações como erosão, dilatação e fechamento para limpar a máscara e melhorar a continuidade do contorno.

- 6. Extração de pontos: Os pontos do contorno são extraídos e organizados para formar um caminho que a tartaruga possa seguir.

- 7. Mapeamento para TurtleSim: As coordenadas da imagem são convertidas para o espaço do TurtleSim, que trabalha aproximadamente em uma área de 0 a 11 nos eixos X e Y.

## Pré-requisitos

Antes de executar, é necessário ter o ambiente ROS 2 funcionando dentro do micromamba.

Neste projeto, o ambiente utilizado foi:

```bash
/home/bebaran/micromamba/envs/ros_env
```

Para ativá-lo:

```bash
micromamba activate /home/bebaran/micromamba/envs/ros_env
```

Também é necessário que o pacote turtlesim esteja instalado no ambiente.

Para verificar:

```bash
ros2 pkg list | grep turtlesim
```
O esperado é aparecer algo como turtlesim turtlesim_msgs

## Como executar o projeto????

### - 1. Entrar na pasta do projeto

```bash
cd "/home/bebaran/Área de trabalho/programação/Módulo 6 - Eng. Comp/ROS 2 Teste"
```

### 2. Ativar o ambiente ROS 2

```bash
micromamba activate /home/bebaran/micromamba/envs/ros_env
```

### 3. Compilar o pacote

Sempre que modificar o código, rode:

```bash
rm -rf build install log
colcon build
source install/setup.bash
```
Se não tiver modificado o código, basta rodar:

```bash
source install/setup.bash
```

## Execução com dois terminais

Para executar o projeto, é necessário usar dois terminais.

### Terminal 1: iniciar o TurtleSim

No primeiro terminal, rode:

```bash
cd "/home/bebaran/Área de trabalho/programação/Módulo 6 - Eng. Comp/ROS 2 Teste"

micromamba activate /home/bebaran/micromamba/envs/ros_env

source install/setup.bash

ros2 run turtlesim turtlesim_node
```

Esse comando abre a janela do TurtleSim.

Deixe essa janela aberta.

### Terminal 2: rodar o controlador da tartaruga

No segundo terminal, rode:

```bash
cd "/home/bebaran/Área de trabalho/programação/Módulo 6 - Eng. Comp/ROS 2 Teste"

micromamba activate /home/bebaran/micromamba/envs/ros_env

source install/setup.bash

ros2 run turtle_draw turtle_controller
```

A tartaruga começará a desenhar o contorno processado a partir da imagem.

### Caso tenha alterado o código

Se alguma alteração foi feita no código, use este fluxo no Terminal 2:

```bash
cd "/home/bebaran/Área de trabalho/programação/Módulo 6 - Eng. Comp/ROS 2 Teste"

micromamba activate /home/bebaran/micromamba/envs/ros_env

rm -rf build install log

colcon build

source install/setup.bash

ros2 run turtle_draw turtle_controller
```
### Arquivo principal

O arquivo principal do projeto é src/turtle_draw/turtle_draw/turtle_controller.py. Nele está tudo que precisa para rodar o projeto. 

## Dificuldades encontradas

Durante o desenvolvimento, uma das primeiras dificuldades foi configurar o ambiente ROS 2 no Ubuntu 25.10. Por isso, foi utilizado um ambiente com micromamba, o que permitiu rodar o ROS 2 sem depender da instalação tradicional via apt.

Também houve dificuldades com a estrutura do pacote ROS 2. Inicialmente, o pacote não estava dentro da pasta src, o que impediu o ROS 2 de reconhecê-lo corretamente. A estrutura foi corrigida para seguir o padrão de workspace do ROS 2.

Outra dificuldade foi a importação dos serviços do TurtleSim. No ambiente utilizado, os serviços estavam disponíveis em turtlesim_msgs.srv, e não em turtlesim.srv.

A maior dificuldade técnica foi transformar os pontos da imagem em um caminho desenhável. Quando os pontos eram extraídos diretamente da imagem binária, a tartaruga ligava pontos fora de ordem, gerando muitos riscos. A solução foi ajustar a extração dos pontos e o mapeamento para o espaço do TurtleSim, usando os pontos da silhueta do cachorro e o retângulo ocupado pelo objeto, em vez da imagem inteira.

## Vídeo

Caso queira me ver explicando, [Clique Aqui!](https://www.youtube.com/watch?v=Kk37toVna2E)