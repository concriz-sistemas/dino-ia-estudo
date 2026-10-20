"""
Lição 2 — Jogue você mesmo e veja, no canto da tela, os 6 números
que a IA vai receber no lugar dos olhos.

Rode a partir da raiz do repositório:
    python licoes/02-o-ambiente/jogar.py

Controles: ESPAÇO / ↑ = pular,  ↓ = abaixar,  R = reiniciar
"""
import os
import sys

# Permite importar o pacote `dino/` que está na raiz do repositório
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from dino.jogo import jogar_manual

if __name__ == "__main__":
    jogar_manual()
