# Passo a Passo do meu projeto!

## Vídeo para a ponderada!

[Clique Aqui!](https://www.youtube.com/watch?v=Kk37toVna2E)

## Como foi fazer a ponderada???

Primeiro passo do nosso projeto, devemos fazer o ambiente ROS 2 no nosso repositório. Como meu Ubuntu é o 25.10, não estava conseguindo instalar. Assim, preparei ele em uma MicroMamba. Para roda-lo, deve usar esse código:

```bash
micromamba activate /home/bebaran/micromamba/envs/ros_env
```

Depois disso, começamos mesmo a fazer a preparação do ambiente para fazer a ponderada.

## Instalar e Configurar o TurtleSim e o Ambiente como Todo!

Após instalar e configurar os ROS 2 dentro do MicroMamba, vamos testar se o TurtleSim está. Para isso, rode:

```bash
ros2 run turtlesim turtlesim_node
```

E em outro terminal, rode:

```bash
micromamba activate /home/bebaran/micromamba/envs/ros_env
ros2 run turtlesim turtle_teleop_key
```

Isso vai fazer os seguintes passos: Primeiro vai "abrir" o TurtleSim em uma tela e no segundo Terminal vamos poder movimentar/girar nossa tartaruguinha para os lados com as teclas (g, t, r, e, d, c, v, b).

## Compilar o WorkSpace 

Com o tudo isso feito, feche tudo e depois ative novamente o MicroMamba para usar e ROS. Com o ROS ativado, rode o seguinte código para criar o WorkSpace para podermos trabalhar:

```bash
colcon build
source install/setup.bash
```

Você vai perceber que foi criado algumas pastas dentro de turtlesim_draw e algumas na raiz do projeto. Se apareceu essas pastas, vamos para o próximo passo!

## Começar a Processar a Imagem!

Antes de qualquer coisa, devemos colocar a imagem no projeto, por isso coloquei ela na pasta "image" dentro de "src", para deixar tudo bem organizado!

Depois, a imagem foi carregada usando cv2.imread(). Essa foi a única utilização do OpenCV na parte de visão computacional, respeitando a restrição do enunciado. 

A primeira transformação feita foi converter a imagem colorida para tons de cinza. Essa escolha foi feita porque a detecção de bordas depende principalmente de mudanças de intensidade entre pixels vizinhos, e não necessariamente das cores.

Em vez de analisar três canais de cor, a imagem passou a ser representada por uma única matriz de intensidade, com valores entre 0 e 255.

A fórmula utilizada foi "cinza = 0.299R + 0.587G + 0.114B" 

Depois da conversão para cinza, foi aplicado um filtro gaussiano manualmente.

A suavização foi necessária porque a imagem possui pequenas variações, textura no pelo do cachorro, sombras e ruídos no fundo. Se o detector de bordas fosse aplicado diretamente, muitos detalhes pequenos seriam identificados como bordas, dificultando o desenho no TurtleSim.

Inicialmente foi considerado o uso de um filtro de média, mas foi feito pelo filtro gaussiano porque ele preserva melhor as formas principais da imagem. Enquanto o filtro de média trata todos os vizinhos com o mesmo peso, o filtro gaussiano dá maior peso aos pixels próximos do centro e menor peso aos pixels mais distantes.

Isso gera uma suavização mais gradual e evita perder totalmente o contorno principal do cachorro.

Para detectar bordas, foi implementado manualmente o operador de Sobel.

O Sobel calcula a variação de intensidade em duas direções: Sobel X → detecta variações horizontais e Sobel Y → detecta variações verticais. Depois, os dois resultados são combinados pela magnitude do gradiente: magnitude = sqrt(gx² + gy²). A magnitude indica a força da borda em cada pixel. Quanto maior o valor, maior a mudança de intensidade naquela região.

O Sobel foi escolhido por ser um algoritmo clássico, simples de explicar e eficiente para identificar bordas em imagens reais. Além disso, sua implementação manual com matrizes é compatível com a proposta da atividade.

Após o Sobel, foi aplicada uma limiarização para transformar a imagem em uma representação binária.

A limiarização transforma os pixels em apenas dois valores: 0   → fundo / região ignorada e 255 → borda / região relevante.

No começo, usamos a limiarização diretamente sobre o resultado do Sobel. Porém, isso gerava muitos detalhes internos, principalmente por causa do pelo, da coleira e de pequenas sombras.

Como a imagem escolhida tem um cachorro escuro sobre fundo claro, ajustamos a função de limiarização para também funcionar no modo "escuro". Assim, pixels abaixo de determinado limiar são considerados parte do objeto principal.

Essa alteração permitiu gerar uma máscara mais limpa do cachorro, separando melhor o animal do fundo.

Também foram implementadas operações morfológicas, como erosão, dilatação e fechamento.

O fechamento morfológico foi usado para conectar pequenas falhas na máscara do cachorro. Ele é composto por: dilatação → erosão 

A dilatação ajuda a unir regiões brancas próximas, enquanto a erosão reduz o excesso gerado pela dilatação. Essa combinação melhora a continuidade da silhueta e ajuda na extração de pontos mais estáveis.

Depois, para extrair apenas o contorno externo, foi utilizada a ideia: borda = máscara - máscara erodida.

Com isso, em vez de desenhar toda a área preenchida do cachorro, foi possível extrair apenas a borda da silhueta.

Uma das principais dificuldades do projeto foi transformar a imagem binária em um caminho que a tartaruga pudesse desenhar.

A primeira abordagem foi extrair todos os pixels brancos usando np.where(). Porém, isso gerava muitos pontos fora de ordem. Como a tartaruga simplesmente conecta um ponto ao próximo, o desenho ficava cheio de riscos atravessando a imagem.

Depois, foi testada uma abordagem que pegava o ponto mais à esquerda e o mais à direita de cada linha. Essa estratégia melhorou a organização, mas deixou o desenho muito vertical e simplificado, parecendo uma forma estreita, e não o cachorro.

A solução final foi extrair os pontos da borda considerando a posição deles em relação ao centro da forma. A ideia foi:

- 1. encontrar todos os pixels brancos da borda;
- 2. calcular o centro desses pontos;
- 3. dividir a volta do objeto em ângulos;
- 4. pegar pontos representativos ao redor da silhueta.

Essa abordagem gerou um caminho mais parecido com o contorno real do cachorro.

Os pontos extraídos da imagem estavam em coordenadas de pixel. Porém, o TurtleSim trabalha em um espaço aproximado de 0 a 11 nos eixos X e Y.

Além disso, existe uma diferença importante entre os sistemas de coordenadas: 

Imagem:
x cresce para a direita
y cresce para baixo

TurtleSim:
x cresce para a direita
y cresce para cima

Por isso, foi necessário inverter o eixo Y durante o mapeamento.

Outra dificuldade foi que, inicialmente, o mapeamento usava a largura e altura totais da imagem. Como o cachorro ocupava apenas uma parte da imagem, o desenho ficava comprimido no eixo X.

Para resolver isso, o mapeamento final passou a usar o retângulo ocupado pelos próprios pontos do objeto, e não a imagem inteira. Isso melhorou a escala e deixou o desenho mais proporcional dentro da tela do TurtleSim.

A primeira dificuldade foi o ambiente de desenvolvimento. Como o sistema utilizado era Ubuntu 25.10, houve problemas para instalar ROS 2 pelos métodos tradicionais. A solução foi utilizar um ambiente com micromamba, onde o ROS 2 já estava configurado.

Outra dificuldade foi entender a estrutura correta de um pacote ROS 2. Inicialmente, o pacote estava fora da pasta src, o que fez o ROS 2 não reconhecer corretamente o pacote. A estrutura foi corrigida para:

```bash
workspace/
├── src/
│   └── turtle_draw/
├── build/
├── install/
└── log/
```

Também ocorreram problemas de importação no Python, principalmente com os serviços do TurtleSim. No ambiente usado, os serviços estavam disponíveis em:
```bash 
from turtlesim_msgs.srv import SetPen, TeleportAbsolute
```

e não em:

```bash
from turtlesim.srv import SetPen, TeleportAbsolute
```

Além disso, houve problemas ao separar o código em múltiplos arquivos. Para simplificar e garantir o funcionamento, a versão final concentrou a pipeline de visão computacional e o controle da tartaruga em um único arquivo principal.

Por fim, a maior dificuldade técnica foi fazer o desenho parecer com a imagem original. A tartaruga conecta pontos em sequência, então a ordem dos pontos é essencial. Quando os pontos eram extraídos sem uma lógica de contorno, o desenho ficava cheio de linhas cruzadas. A solução foi melhorar a extração e o mapeamento dos pontos para representar melhor a silhueta do cachorro.

## Conclusão?

O projeto conseguiu implementar uma pipeline completa de visão computacional e integrá-la ao ROS 2 com TurtleSim. A imagem foi processada manualmente usando NumPy, passando por conversão para tons de cinza, suavização gaussiana, Sobel, limiarização, morfologia, extração de pontos e mapeamento para o espaço do simulador.

Apesar das dificuldades com ambiente, importações e organização dos pontos, a versão final conseguiu transformar a imagem em um caminho desenhável pela tartaruga. O projeto mostrou na prática como uma imagem pode ser convertida em informação geométrica e depois em comandos para um robô simulado.