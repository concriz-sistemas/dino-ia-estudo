"""
jogo.py — Uma versão simplificada do jogo do dinossauro do Chrome.

O jogo foi escrito para suportar VÁRIOS dinossauros ao mesmo tempo,
todos enfrentando os mesmos obstáculos. Isso é essencial para a
neuroevolução: a população inteira joga a mesma partida, e assim a
comparação entre eles é justa.

Rode este arquivo diretamente para jogar você mesmo:
    python jogo.py
Controles: ESPAÇO / ↑ = pular,  ↓ = abaixar,  R = reiniciar
"""
import random
import pygame

try:
    from dino import sprites
except ImportError:  # rodando "python dino/jogo.py" direto
    import sprites

# ---------------------------------------------------------------------------
# Constantes do mundo
# ---------------------------------------------------------------------------
LARGURA, ALTURA = 800, 300
CHAO_Y = 250              # coordenada y do chão (os pés do dino ficam aqui)
GRAVIDADE = 1.0
FORCA_PULO = -17          # velocidade vertical inicial do pulo (negativo = pra cima)
VEL_INICIAL = 8           # pixels por quadro
VEL_MAX = 20
ACELERACAO = 0.003        # quanto a velocidade aumenta a cada quadro

# Ações possíveis (é isso que a rede neural vai escolher)
NADA, PULAR, ABAIXAR = 0, 1, 2

COR_FUNDO = (247, 247, 247)
COR_CHAO = (83, 83, 83)
COR_OBSTACULO = (60, 60, 60)
COR_NUVEM = (205, 205, 205)


# ---------------------------------------------------------------------------
# Dinossauro
# ---------------------------------------------------------------------------
class Dino:
    LARG_EM_PE, ALT_EM_PE = 40, 50
    LARG_ABAIXADO, ALT_ABAIXADO = 55, 26

    def __init__(self, cor=(83, 83, 83)):
        self.x = 60
        self.pes = CHAO_Y          # y da base do dino
        self.vy = 0.0
        self.abaixado = False
        self.vivo = True
        self.pontos = 0            # quadros sobrevividos (vira o "fitness")
        self.cor = cor

    @property
    def no_chao(self):
        return self.pes >= CHAO_Y

    @property
    def rect(self):
        if self.abaixado and self.no_chao:
            w, h = self.LARG_ABAIXADO, self.ALT_ABAIXADO
        else:
            w, h = self.LARG_EM_PE, self.ALT_EM_PE
        return pygame.Rect(self.x, int(self.pes - h), w, h)

    def atualizar(self, acao):
        if acao == PULAR and self.no_chao:
            self.vy = FORCA_PULO
        self.abaixado = (acao == ABAIXAR)
        # Abaixar no ar faz o dino cair mais rápido (igual ao jogo original)
        if self.abaixado and not self.no_chao:
            self.vy += 2
        self.vy += GRAVIDADE
        self.pes += self.vy
        if self.pes >= CHAO_Y:
            self.pes = CHAO_Y
            self.vy = 0


# ---------------------------------------------------------------------------
# Obstáculos
# ---------------------------------------------------------------------------
# tipo: (largura, altura, distância da base até o chão)
TIPOS_OBSTACULO = {
    "cacto_pequeno": (20, 40, 0),
    "cacto_grande": (30, 55, 0),
    "cactos_grupo": (55, 40, 0),
    "passaro_baixo": (40, 25, 10),   # tem que pular
    "passaro_medio": (40, 25, 35),   # tem que abaixar
    "passaro_alto": (40, 25, 60),    # pode ignorar
}


class Obstaculo:
    def __init__(self, tipo, x):
        self.tipo = tipo
        w, h, elevacao = TIPOS_OBSTACULO[tipo]
        self.rect = pygame.Rect(x, CHAO_Y - elevacao - h, w, h)
        self.x = float(x)

    @property
    def eh_passaro(self):
        return self.tipo.startswith("passaro")

    def mover(self, vel):
        self.x -= vel
        self.rect.x = int(self.x)


# ---------------------------------------------------------------------------
# O jogo em si
# ---------------------------------------------------------------------------
class Jogo:
    def __init__(self, n_dinos=1, semente=None, cores=None):
        self.rng = random.Random(semente)
        if cores is None:
            cores = [self._cor_aleatoria() for _ in range(n_dinos)]
        self.dinos = [Dino(cor) for cor in cores]
        self.obstaculos = []
        self.vel = VEL_INICIAL
        self.quadro = 0
        self.distancia = 0.0       # só para rolar o cenário (chão e nuvens)
        self._proximo_gap = 0

    def _cor_aleatoria(self):
        return tuple(self.rng.randint(40, 200) for _ in range(3))

    # ---- lógica ----------------------------------------------------------
    def algum_vivo(self):
        return any(d.vivo for d in self.dinos)

    def _criar_obstaculo(self):
        # Pássaros só aparecem depois que o jogo acelera um pouco
        tipos = ["cacto_pequeno", "cacto_grande", "cactos_grupo"]
        if self.vel > 10:
            tipos += ["passaro_baixo", "passaro_medio", "passaro_alto"]
        tipo = self.rng.choice(tipos)
        self.obstaculos.append(Obstaculo(tipo, LARGURA + 10))
        # distância até o próximo obstáculo cresce com a velocidade
        self._proximo_gap = self.rng.randint(int(self.vel * 30), int(self.vel * 55))

    def passo(self, acoes):
        """Avança o jogo um quadro. `acoes` tem uma ação por dino."""
        self.quadro += 1

        for dino, acao in zip(self.dinos, acoes):
            if dino.vivo:
                dino.atualizar(acao)

        for obs in self.obstaculos:
            obs.mover(self.vel)
        self.obstaculos = [o for o in self.obstaculos if o.rect.right > 0]

        if not self.obstaculos or (LARGURA - self.obstaculos[-1].rect.right) > self._proximo_gap:
            self._criar_obstaculo()

        # colisões (hitbox levemente menor, para ser justo)
        for dino in self.dinos:
            if not dino.vivo:
                continue
            hit = dino.rect.inflate(-8, -8)
            if any(hit.colliderect(o.rect) for o in self.obstaculos):
                dino.vivo = False
            else:
                dino.pontos += 1

        self.distancia += self.vel
        self.vel = min(VEL_MAX, self.vel + ACELERACAO)

    # ---- o que a IA "enxerga" -------------------------------------------
    def proximo_obstaculo(self, dino):
        for o in self.obstaculos:
            if o.rect.right > dino.x:
                return o
        return None

    def sensores(self, dino):
        """
        Transforma o estado do jogo em 6 números (normalizados ~0..1).
        Estas são as ENTRADAS da rede neural. Escolher boas entradas é
        metade do trabalho — experimente mudar isso depois!
        """
        o = self.proximo_obstaculo(dino)
        if o is None:
            dist, larg, alt, elev = 1.0, 0.0, 0.0, 0.0
        else:
            dist = (o.rect.left - (dino.x + Dino.LARG_EM_PE)) / LARGURA
            larg = o.rect.width / 60
            alt = o.rect.height / 60
            elev = (CHAO_Y - o.rect.bottom) / 60      # 0 = no chão; >0 = pássaro no ar
        altura_dino = (CHAO_Y - dino.pes) / 150       # 0 = no chão
        vel = self.vel / VEL_MAX
        return [dist, larg, alt, elev, altura_dino, vel]

    # ---- desenho ---------------------------------------------------------
    def _desenhar_cenario(self, tela):
        tela.fill(COR_FUNDO)
        # Nuvens andam mais devagar que o chão (parece que estão longe)
        for x0, y in [(100, 50), (420, 80), (700, 40), (950, 70)]:
            x = (x0 - self.distancia * 0.15) % (LARGURA + 150) - 75
            tela.blit(sprites.figura("nuvem", sprites.NUVEM, COR_NUVEM, (60, 15)), (x, y))
        pygame.draw.line(tela, COR_CHAO, (0, CHAO_Y), (LARGURA, CHAO_Y), 2)
        # Pedrinhas no chão, rolando na velocidade do jogo
        for i in range(40):
            x = (i * 53 + i * i * 7 - self.distancia) % (LARGURA + 20) - 10
            y = CHAO_Y + 5 + (i * 37) % 12
            pygame.draw.line(tela, COR_CHAO, (x, y), (x + 2 + i % 4, y), 1)

    def _desenhar_obstaculo(self, tela, o):
        tamanho = o.rect.size
        if o.tipo == "cactos_grupo":
            img = sprites.grupo_de_cactos(COR_OBSTACULO, tamanho)
        elif o.eh_passaro:
            asa = (self.quadro // 10) % 2
            img = sprites.figura(f"passaro{asa}", [sprites.PASSARO_1, sprites.PASSARO_2][asa],
                                 COR_OBSTACULO, tamanho)
        else:
            img = sprites.figura("cacto", sprites.CACTO, COR_OBSTACULO, tamanho)
        tela.blit(img, o.rect)

    def _desenhar_dino(self, tela, d, alfa):
        perna = (self.quadro // 6) % 2
        if not d.vivo:
            nome, mapa = "bateu", sprites.DINO_BATEU
        elif not d.no_chao:
            nome, mapa = "pulando", sprites.DINO_PULANDO
        elif d.abaixado:
            nome, mapa = f"abaixado{perna}", [sprites.DINO_ABAIXADO_1, sprites.DINO_ABAIXADO_2][perna]
        else:
            nome, mapa = f"correndo{perna}", [sprites.DINO_CORRENDO_1, sprites.DINO_CORRENDO_2][perna]
        tela.blit(sprites.figura(nome, mapa, d.cor, d.rect.size, alfa), d.rect)

    def desenhar(self, tela, fonte, linhas_info=(), destacar=None):
        self._desenhar_cenario(tela)

        for o in self.obstaculos:
            self._desenhar_obstaculo(tela, o)

        # Com muitos dinos, eles ficam meio transparentes para dar para ver a multidão.
        # Com um só, ele aparece inteiro, e continua na tela depois de bater.
        sozinho = len(self.dinos) == 1
        for d in self.dinos:
            if d is not destacar and (d.vivo or sozinho):
                self._desenhar_dino(tela, d, 255 if sozinho else 110)
        if destacar is not None and (destacar.vivo or sozinho):
            self._desenhar_dino(tela, destacar, 255)

        # Linha do "sensor": do dino destacado até o obstáculo que ele está olhando
        if destacar is not None and destacar.vivo:
            o = self.proximo_obstaculo(destacar)
            if o:
                pygame.draw.line(tela, (230, 80, 80), destacar.rect.center, o.rect.center, 1)

        y = 8
        for linha in linhas_info:
            tela.blit(fonte.render(linha, True, (40, 40, 40)), (10, y))
            y += 20


# ---------------------------------------------------------------------------
# Modo humano: jogue você mesmo para sentir a dificuldade
# ---------------------------------------------------------------------------
def jogar_manual():
    pygame.init()
    tela = pygame.display.set_mode((LARGURA, ALTURA))
    pygame.display.set_caption("Dino — modo humano")
    fonte = pygame.font.SysFont("monospace", 16)
    relogio = pygame.time.Clock()
    jogo = Jogo(1, cores=[(83, 83, 83)])

    while True:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit()
                return
            if e.type == pygame.KEYDOWN and e.key == pygame.K_r:
                jogo = Jogo(1, cores=[(83, 83, 83)])

        teclas = pygame.key.get_pressed()
        acao = NADA
        if teclas[pygame.K_SPACE] or teclas[pygame.K_UP]:
            acao = PULAR
        elif teclas[pygame.K_DOWN]:
            acao = ABAIXAR

        if jogo.algum_vivo():
            jogo.passo([acao])
        info = [f"Pontos: {jogo.dinos[0].pontos}", f"Velocidade: {jogo.vel:.1f}"]
        if not jogo.algum_vivo():
            info.append("GAME OVER — aperte R")
        # Mostra os 6 números que a IA "enxerga" — os mesmos que você vê com os olhos
        nomes = ["distância", "largura", "altura", "elevação", "alt. dino", "velocidade"]
        valores = jogo.sensores(jogo.dinos[0])
        info.append("")
        info.append("O que a IA enxerga:")
        info += [f"  {n:<10} {v:5.2f}" for n, v in zip(nomes, valores)]
        jogo.desenhar(tela, fonte, info, destacar=jogo.dinos[0])
        pygame.display.flip()
        relogio.tick(60)


if __name__ == "__main__":
    jogar_manual()
