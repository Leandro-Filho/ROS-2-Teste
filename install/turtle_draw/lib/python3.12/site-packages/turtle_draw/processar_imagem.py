import cv2 # Mostrar a imagem
import numpy as np # Responsável por fazer as contas
import matplotlib.pyplot as plt # Plotar imagens e gráficos


# Caminho da imagem original que vamos operar em cima
IMAGE_PATH = "../images/dog.jpeg"

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
def aplicar_limiarizacao(imagem, limiar=80, modo="claro"):

    imagem_binaria = np.zeros_like(imagem, dtype=np.uint8)

    if modo == "claro":
        imagem_binaria[imagem >= limiar] = 255

    elif modo == "escuro":
        imagem_binaria[imagem < limiar] = 255

    else:
        raise ValueError("O modo deve ser 'claro' ou 'escuro'.")

    return imagem_binaria

# Para deixar a borda ainda mais forte, usamos a dilatação para deixar ainda mais grossa a borda, que é branca 
def aplicar_dilatacao(imagem_binaria, kernel_size=3):

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
def aplicar_erosao(imagem_binaria, kernel_size=3):

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
def aplicar_fechamento(imagem_binaria, kernel_size=3):

    dilatada = aplicar_dilatacao(imagem_binaria, kernel_size)
    fechada = aplicar_erosao(dilatada, kernel_size)

    return fechada


# Agora, para fazer nosso mascote andar sobre a borda, vamos pegar esses pontos.
def extrair_pontos_da_borda(bordas_limpas, passo_linha=4):

    altura, largura = bordas_limpas.shape

    lado_esquerdo = []
    lado_direito = []

    for y in range(0, altura, passo_linha):
        # Pega todos os x onde existe borda branca nessa linha y
        xs = np.where(bordas_limpas[y, :] == 255)[0]

        # Se essa linha não tem ponto branco, pula
        if len(xs) == 0:
            continue

        x_esquerda = xs[0]
        x_direita = xs[-1]

        lado_esquerdo.append((x_esquerda, y))
        lado_direito.append((x_direita, y))

    # Monta um caminho fechado:
    # desce pelo lado esquerdo e sobe pelo lado direito
    pontos = lado_esquerdo + lado_direito[::-1]

    return pontos

# Depois vamos ver esses pontinhos para ver se ficou bom 
def mostrar_pontos_extraidos(imagem, pontos):

    pontos_x = [ponto[0] for ponto in pontos]
    pontos_y = [ponto[1] for ponto in pontos]

    plt.figure(figsize=(8, 6))
    plt.imshow(imagem)
    plt.scatter(pontos_x, pontos_y, s=1)
    plt.title("Pontos extraídos da borda")
    plt.axis("off")
    plt.show()

# Para mapear tudo e deixar alinhado para deixar a tartaruguinha andando certinho, vamos tranformar os pontos em 
# Coodernadas.
def mapear_pontos_para_turtlesim(pontos, largura_imagem, altura_imagem):

    pontos_turtle = []

    # Margem para a tartaruga não bater exatamente nas bordas da tela.
    turtle_min = 1.0
    turtle_max = 10.0

    turtle_range = turtle_max - turtle_min

    for x_pixel, y_pixel in pontos:
        # Converte x da imagem para x do TurtleSim
        x_turtle = turtle_min + (x_pixel / largura_imagem) * turtle_range

        # Converte y da imagem para y do TurtleSim
        # Aqui fazemos a inversão vertical:
        # y_pixel pequeno, topo da imagem, vira y_turtle grande.
        y_turtle = turtle_min + ((altura_imagem - y_pixel) / altura_imagem) * turtle_range

        pontos_turtle.append((x_turtle, y_turtle))

    return pontos_turtle

def mostrar_pontos_turtlesim(pontos_turtle):

    pontos_x = [ponto[0] for ponto in pontos_turtle]
    pontos_y = [ponto[1] for ponto in pontos_turtle]

    plt.figure(figsize=(6, 6))
    plt.scatter(pontos_x, pontos_y, s=1)
    plt.title("Pontos mapeados para o TurtleSim")
    plt.xlim(0, 11)
    plt.ylim(0, 11)
    plt.xlabel("x TurtleSim")
    plt.ylabel("y TurtleSim")
    plt.grid(True)
    plt.show()

def mostrar_imagens(imagem, grayscale, suavizada, bordas, bordas_binarias, bordas_limpas):

    plt.figure(figsize=(24, 5))

    plt.subplot(1, 6, 1)
    plt.title("Imagem original")
    plt.imshow(imagem)
    plt.axis("off")

    plt.subplot(1, 6, 2)
    plt.title("Tons de cinza")
    plt.imshow(grayscale, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 6, 3)
    plt.title("Filtro gaussiano")
    plt.imshow(suavizada, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 6, 4)
    plt.title("Sobel")
    plt.imshow(bordas, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 6, 5)
    plt.title("Bordas binárias")
    plt.imshow(bordas_binarias, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 6, 6)
    plt.title("Fechamento morfológico")
    plt.imshow(bordas_limpas, cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

# Essa última função faz tudo rodar
def main():

    imagem = carregar_imagem(IMAGE_PATH)

    # 1. Converter para tons de cinza
    grayscale = rgb_para_cinza(imagem)

    # 2. Suavizar com filtro gaussiano
    suavizada = aplicar_filtro_gaussiano(grayscale, kernel_size=5, sigma=1.0)

    # 3. Detectar bordas com Sobel
    bordas, gradiente_x, gradiente_y = aplicar_sobel(suavizada)

    # 4. Transformar bordas em imagem binária
    bordas_binarias = aplicar_limiarizacao(bordas, limiar=60)

    # 5. Limpar/conectar bordas usando fechamento morfológico
    bordas_limpas = aplicar_fechamento(bordas_binarias, kernel_size=3)

    pontos = extrair_pontos_da_borda(bordas_limpas, passo=4)

    altura_imagem, largura_imagem = bordas_limpas.shape

    pontos_turtle = mapear_pontos_para_turtlesim(
        pontos,
        largura_imagem,
        altura_imagem
    )

    print("Imagem carregada com sucesso!")
    print(f"Formato da imagem original: {imagem.shape}")
    print(f"Formato da imagem em cinza: {grayscale.shape}")
    print(f"Formato da imagem suavizada: {suavizada.shape}")
    print(f"Formato da imagem de bordas Sobel: {bordas.shape}")
    print(f"Formato da imagem binária: {bordas_binarias.shape}")
    print(f"Formato da imagem limpa: {bordas_limpas.shape}")
    print(f"Quantidade de pontos extraídos: {len(pontos)}")
    print(f"Quantidade de pontos mapeados para TurtleSim: {len(pontos_turtle)}")
    print("Primeiros 5 pontos no TurtleSim:")
    print(pontos_turtle[:5])

    mostrar_imagens(imagem, grayscale, suavizada, bordas, bordas_binarias, bordas_limpas)
    mostrar_pontos_extraidos(imagem, pontos)
    mostrar_pontos_turtlesim(pontos_turtle)


if __name__ == "__main__":
    main()


#
def gerar_pontos_turtlesim():

    imagem = carregar_imagem(IMAGE_PATH)

    grayscale = rgb_para_cinza(imagem)

    suavizada = aplicar_filtro_gaussiano(grayscale, kernel_size=5, sigma=1.0)

    # Mantemos o Sobel na pipeline porque ele faz parte da detecção de bordas.
    bordas, gradiente_x, gradiente_y = aplicar_sobel(suavizada)

    # Aqui usamos a própria limiarização para separar o cachorro escuro do fundo claro.
    mascara_cachorro = aplicar_limiarizacao(suavizada, limiar=105, modo="escuro")

    # Limpamos/conectamos pequenas falhas na máscara.
    mascara_limpa = aplicar_fechamento(mascara_cachorro, kernel_size=5)

    # Pegamos a borda da máscara usando a ideia:
    # borda = máscara - máscara erodida
    mascara_erodida = aplicar_erosao(mascara_limpa, kernel_size=3)

    borda_silhueta = mascara_limpa - mascara_erodida
    borda_silhueta[borda_silhueta > 0] = 255

    pontos = extrair_pontos_da_borda(borda_silhueta, passo_linha=3)

    altura_imagem, largura_imagem = borda_silhueta.shape

    pontos_turtle = mapear_pontos_para_turtlesim(
        pontos,
        largura_imagem,
        altura_imagem
    )

    return pontos_turtle