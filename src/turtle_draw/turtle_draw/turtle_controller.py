import cv2 # Mostrar a imagem
import numpy as np # Responsável por fazer as contas
import matplotlib.pyplot as plt # Plotar imagens e gráficos

import rclpy
from rclpy.node import Node

from turtlesim_msgs.srv import SetPen, TeleportAbsolute


# Caminho da imagem original que vamos operar em cima
IMAGE_PATH = "/home/bebaran/Área de trabalho/programação/Módulo 6 - Eng. Comp/ROS 2 Teste/src/turtle_draw/images/dog.jpeg"

# Função responsável por carregar a imagem
def carregar_imagem(path):
    imagem = cv2.imread(path)

    # Mensagem de erro só para conseguir debugar depois
    if imagem is None:
        raise FileNotFoundError(f"Não foi possível carregar a imagem em: {path}")
    
    return imagem

# Transformar a imagem na escala de cinza usando a grayscale.
# Devemos fazer isso porque a detecção de borda funciona observando a diferença brusca de intensidade de luz nos pixels
# E usamos a grayscale para, ao invés de ter 3 canais para analisar isso, teremos apenas um, tendo valores de 0 ou 1.
def rgb_para_cinza(imagem):
    blue = imagem[:, :, 0]
    green = imagem[:, :, 1]
    red = imagem[:, :, 2]

    grayscale = 0.299 * red + 0.587 * green + 0.114 * blue

    # Convertendo para uint8 porque imagens normalmente usam valores de 0 a 255.
    grayscale = grayscale.astype(np.uint8)

    return grayscale


# Essa função vai suavizar a iamgem, usando um a técnica de suavização por gaussiano, que faz a seguinte coisa;
# O kernel gaussiano é uma matriz de pesos. Pixels mais próximos do centro recebem pesos maiores. 
# Pixels mais distantes recebem pesos menores. Em um grande resumo é isso que ele faz. E porque não escolher outro?
# O por média simples dos viszinhos, não da importância para nenhum ponto da imagem, fazendo com que perca
# Um pouco de nitidez depois de fazer a média, já o gaussiano da importância para o que está no centro, deixando
# Mais nítido o que está no centro. Já que a imagem do nosso queriado diguinho está no centro, nada mais justo que
# Usar o que nos favorece mais.

# Como temos que criar o kernel e tudo do zero, primeiro criamos o kernel responsável por fazer as continhas para nós
def criar_kernel_gaussiano(kernel_size=5, sigma=1.0):

    centro = kernel_size // 2

    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)

    for y in range(kernel_size):
        for x in range(kernel_size):
            distancia_x = x - centro
            distancia_y = y - centro

            expoente = -((distancia_x ** 2 + distancia_y ** 2) / (2 * sigma ** 2))

            kernel[y, x] = np.exp(expoente) / (2 * np.pi * sigma ** 2)

    # Normaliza para a soma dos pesos ser 1
    kernel = kernel / np.sum(kernel)

    return kernel

# Depois, aplicamos esse kernel criado manualmente na imagem que criamos também manualmente usando o padding, para
# Deixar a borda mais nítida.
def aplicar_filtro_gaussiano(grayscale, kernel_size=5, sigma=1.0):

    kernel = criar_kernel_gaussiano(kernel_size, sigma)

    padding = kernel_size // 2

    imagem_com_borda = np.pad(grayscale, padding, mode="edge")

    altura, largura = grayscale.shape

    suavizada = np.zeros_like(grayscale, dtype=np.float32)

    for y in range(altura):
        for x in range(largura):
            regiao = imagem_com_borda[y:y + kernel_size, x:x + kernel_size]

            # Multiplica cada pixel da região pelo peso correspondente do kernel
            valor_filtrado = np.sum(regiao * kernel)

            suavizada[y, x] = valor_filtrado

    return suavizada.astype(np.uint8)

# O filtro de Sobel vai ser o resçponsável por fazer toda a parte de detecção de borda e deixar bem alinhado para 
# Nossa parte de fazer nossa tartarugunha andar sub a borda do nosso doguinho.
# Bom, primeiro de tudo, ela vai passar dois kernels, um horizontal e outro vertical para detectar uma diferença
# Grande de um pixel para o seu vizinho. Depois disso, se calcula a magnitude para saber a força de cada borda e assim
# Descobrimos a borda das imagens. O código ficou um pouco grande porque precisamos configurar onde que esse dois kernels
# Iriam passar.

def aplicar_sobel(imagem_suavizada):

    sobel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=np.float32)

    sobel_y = np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ], dtype=np.float32)

    altura, largura = imagem_suavizada.shape

    # Como o kernel Sobel é 3x3, precisamos de 1 pixel de borda.
    padding = 1

    # Adicionamos uma borda artificial para conseguir calcular os pixels das extremidades.
    imagem_com_borda = np.pad(imagem_suavizada, padding, mode="edge")

    # Matrizes onde vamos guardar os gradientes em X e Y.
    gradiente_x = np.zeros_like(imagem_suavizada, dtype=np.float32)
    gradiente_y = np.zeros_like(imagem_suavizada, dtype=np.float32)

    # Percorremos cada pixel da imagem.
    for y in range(altura):
        for x in range(largura):

            # Pegamos a região 3x3 ao redor do pixel atual.
            regiao = imagem_com_borda[y:y + 3, x:x + 3]

            # Aplicamos o kernel de Sobel X.
            gx = np.sum(regiao * sobel_x)

            # Aplicamos o kernel de Sobel Y.
            gy = np.sum(regiao * sobel_y)

            # Guardamos os resultados.
            gradiente_x[y, x] = gx
            gradiente_y[y, x] = gy

    # Calcula a força total da borda juntando X e Y.
    magnitude = np.sqrt(gradiente_x ** 2 + gradiente_y ** 2)

    # Normaliza para o intervalo 0 a 255 para conseguir visualizar como imagem.
    if np.max(magnitude) != 0:
        magnitude = magnitude / np.max(magnitude) * 255

    bordas = magnitude.astype(np.uint8)

    return bordas, gradiente_x, gradiente_y

# Para melhorar a eficiencia da nossa tartaruguinha, devemos fazer com a imagem fique o mais simples possível, sem muita
# Diferença de pixels e nada mais, deixar o mais simples possível. Por isso usamos a limiarização. Ela vai fazer
# Com que tenha apenas dois valores: ou 225 ou 0, deixando a borda extremamente nítida para nosso amiguinho.
def aplicar_limiarizacao(imagem, limiar=50, modo="claro"):

    imagem_binaria = np.zeros_like(imagem, dtype=np.uint8)

    if modo == "claro":
        imagem_binaria[imagem >= limiar] = 255

    elif modo == "escuro":
        imagem_binaria[imagem < limiar] = 255

    else:
        raise ValueError("O modo deve ser 'claro' ou 'escuro'.")

    return imagem_binaria

# Para deixar a borda ainda mais forte, usamos a dilatação para deixar ainda mais grossa a borda, que é branca 
def aplicar_dilatacao(imagem_binaria, kernel_size=5):

    padding = kernel_size // 2

    imagem_com_borda = np.pad(imagem_binaria, padding, mode="constant", constant_values=0)

    altura, largura = imagem_binaria.shape

    resultado = np.zeros_like(imagem_binaria, dtype=np.uint8)

    for y in range(altura):
        for x in range(largura):
            regiao = imagem_com_borda[y:y + kernel_size, x:x + kernel_size]

            # Se tiver qualquer pixel branco na região, o centro vira branco.
            if np.any(regiao == 255):
                resultado[y, x] = 255
            else:
                resultado[y, x] = 0

    return resultado

# Depois aplicamos a erosão para deixar só o que realmente é branco. Como no anterior meio que aumentamos a borda para
# Deixar mais forte o meio, vamos tirar as bordas das bordas usando a lógica inversa da dilatação: se o pixel tiver 
# Algum vizinho preto ou não branco, coloque ele preto. Assim, deixamos realmente só a bordas reais.
def aplicar_erosao(imagem_binaria, kernel_size=5):

    padding = kernel_size // 2

    imagem_com_borda = np.pad(imagem_binaria, padding, mode="constant", constant_values=0)

    altura, largura = imagem_binaria.shape

    resultado = np.zeros_like(imagem_binaria, dtype=np.uint8)

    for y in range(altura):
        for x in range(largura):
            regiao = imagem_com_borda[y:y + kernel_size, x:x + kernel_size]

            # Só vira branco se toda a região for branca.
            if np.all(regiao == 255):
                resultado[y, x] = 255
            else:
                resultado[y, x] = 0

    return resultado

# Para finalizar de fato, juntamos as duas funções anterior no fechamento
def aplicar_fechamento(imagem_binaria, kernel_size=5):

    dilatada = aplicar_dilatacao(imagem_binaria, kernel_size)
    fechada = aplicar_erosao(dilatada, kernel_size)

    return fechada


# Agora, para fazer nosso mascote andar sobre a borda, vamos pegar esses pontos.
def extrair_pontos_da_borda(borda, quantidade_angulos=1200):

    ys, xs = np.where(borda == 255)

    if len(xs) == 0:
        return []

    centro_x = np.mean(xs)
    centro_y = np.mean(ys)

    angulos = np.arctan2(ys - centro_y, xs - centro_x)
    distancias = np.sqrt((xs - centro_x) ** 2 + (ys - centro_y) ** 2)

    pontos = []

    limites = np.linspace(-np.pi, np.pi, quantidade_angulos)

    for i in range(len(limites) - 1):
        angulo_min = limites[i]
        angulo_max = limites[i + 1]

        indices = np.where((angulos >= angulo_min) & (angulos < angulo_max))[0]

        if len(indices) == 0:
            continue

        # Dentro desse pedaço de ângulo, pega o ponto mais longe do centro
        indice_mais_longe = indices[np.argmax(distancias[indices])]

        x = xs[indice_mais_longe]
        y = ys[indice_mais_longe]

        pontos.append((x, y))

    # Fecha o desenho voltando para o primeiro ponto
    if len(pontos) > 0:
        pontos.append(pontos[0])

    return pontos

# Para mapear tudo e deixar alinhado para deixar a tartaruguinha andando certinho, vamos tranformar os pontos em 
# Coodernadas.
def mapear_pontos_para_turtlesim(pontos, largura_imagem=None, altura_imagem=None):

    if len(pontos) == 0:
        return []

    xs = np.array([p[0] for p in pontos])
    ys = np.array([p[1] for p in pontos])

    min_x = np.min(xs)
    max_x = np.max(xs)
    min_y = np.min(ys)
    max_y = np.max(ys)

    largura_objeto = max_x - min_x
    altura_objeto = max_y - min_y

    centro_x = (min_x + max_x) / 2
    centro_y = (min_y + max_y) / 2

    # Centro da tela do TurtleSim
    turtle_centro_x = 5.5
    turtle_centro_y = 5.5

    # Tamanho máximo que o desenho pode ocupar na tela
    tamanho_turtle = 8.0

    escala_x = tamanho_turtle / largura_objeto
    escala_y = tamanho_turtle / altura_objeto

    # Usa a menor escala para não deformar o cachorro
    escala = min(escala_x, escala_y)

    pontos_turtle = []

    for x_pixel, y_pixel in pontos:
        x_turtle = turtle_centro_x + (x_pixel - centro_x) * escala

        # Inverte o Y, porque na imagem o Y cresce para baixo,
        # e no TurtleSim o Y cresce para cima.
        y_turtle = turtle_centro_y - (y_pixel - centro_y) * escala

        pontos_turtle.append((x_turtle, y_turtle))

    return pontos_turtle
def gerar_pontos_turtlesim():

    imagem = carregar_imagem(IMAGE_PATH)

    grayscale = rgb_para_cinza(imagem)

    suavizada = aplicar_filtro_gaussiano(grayscale, kernel_size=5, sigma=1.0)

    # Mantemos o Sobel na pipeline porque ele faz parte da detecção de bordas.
    bordas, gradiente_x, gradiente_y = aplicar_sobel(suavizada)

    # Aqui usamos a própria limiarização para separar o cachorro escuro do fundo claro.
    mascara_cachorro = aplicar_limiarizacao(suavizada, limiar=50, modo="escuro")

    mascara_limpa = aplicar_fechamento(mascara_cachorro, kernel_size=5)

    mascara_erodida = aplicar_erosao(mascara_limpa, kernel_size=5)

    borda_silhueta = mascara_limpa - mascara_erodida
    borda_silhueta[borda_silhueta > 0] = 255

    pontos = extrair_pontos_da_borda(mascara_limpa, quantidade_angulos=1200)

    pontos_turtle = mapear_pontos_para_turtlesim(
        pontos,
        None,
        None
    )

    return pontos_turtle

class TurtleController(Node):

    def __init__(self):
        super().__init__("turtle_controller")

        # Gera os pontos da borda da imagem já convertidos para o espaço do TurtleSim
        self.pontos = gerar_pontos_turtlesim()

        self.get_logger().info(f"Quantidade de pontos: {len(self.pontos)}")

        # Cria os clientes dos serviços do TurtleSim
        self.pen_client = self.create_client(SetPen, "/turtle1/set_pen")
        self.teleport_client = self.create_client(TeleportAbsolute, "/turtle1/teleport_absolute")

        # Espera os serviços existirem
        self.pen_client.wait_for_service()
        self.teleport_client.wait_for_service()

        # Começa o desenho
        self.desenhar()

    def mudar_caneta(self, ligada):
        """
        Liga ou desliga a caneta da tartaruga.
        """

        pedido = SetPen.Request()

        pedido.r = 0
        pedido.g = 0
        pedido.b = 0
        pedido.width = 2

        # No TurtleSim:
        # off = 0 significa caneta ligada
        # off = 1 significa caneta desligada
        if ligada:
            pedido.off = 0
        else:
            pedido.off = 1

        futuro = self.pen_client.call_async(pedido)
        rclpy.spin_until_future_complete(self, futuro)

    def ir_para_ponto(self, x, y):
        """
        Move a tartaruga direto para uma posição x, y.
        """

        pedido = TeleportAbsolute.Request()

        pedido.x = float(x)
        pedido.y = float(y)
        pedido.theta = 0.0

        futuro = self.teleport_client.call_async(pedido)
        rclpy.spin_until_future_complete(self, futuro)

    def desenhar(self):
        """
        Desenha os pontos encontrados na imagem.
        """

        if len(self.pontos) == 0:
            self.get_logger().info("Nenhum ponto encontrado.")
            return

        # Primeiro ponto da lista
        primeiro_x, primeiro_y = self.pontos[0]

        # Vai até o primeiro ponto sem desenhar
        self.mudar_caneta(False)
        self.ir_para_ponto(primeiro_x, primeiro_y)

        # Liga a caneta para começar o desenho
        self.mudar_caneta(True)

        # Passa por todos os pontos
        for i, ponto in enumerate(self.pontos):
            x, y = ponto
            self.ir_para_ponto(x, y)

            if i % 100 == 0:
                self.get_logger().info(f"Desenhando ponto {i}")

        self.get_logger().info("Desenho finalizado.")


def main(args=None):
    rclpy.init(args=args)

    controlador = TurtleController()

    controlador.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()