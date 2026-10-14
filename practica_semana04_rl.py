"""
Práctica de Laboratorio - Semana 04: Q-Learning en entorno estocástico
y experimento de Reward Shaping (Gridworld 4x4 tipo FrozenLake slippery).
"""
import random
from typing import List, Tuple

import numpy as np

SIMBOLOS_ACCION = ["<", "v", ">", "^"]   # 0: Izq, 1: Abajo, 2: Der, 3: Arriba


class EntornoGridworldEstocastico:
    """Gridworld 4x4 estocástico (versión reducida de FrozenLake)."""

    def __init__(self, reward_shaping: bool = False):
        self.filas = 4
        self.columnas = 4
        self.n_estados = 16
        self.n_acciones = 4  # 0: Izquierda, 1: Abajo, 2: Derecha, 3: Arriba
        self.agujeros = {5, 7, 11, 12}
        self.meta = 15
        self.estado_actual = 0
        self.reward_shaping = reward_shaping

    def reset(self) -> int:
        self.estado_actual = 0
        return self.estado_actual

    def step(self, accion: int) -> Tuple[int, float, bool]:
        # TAREA 1: estocasticidad del viento (igual que FrozenLake slippery):
        # 1/3 acción deseada, 1/3 desvío a un lado, 1/3 desvío al otro lado.
        # (accion -1) y (accion +1) son las dos direcciones perpendiculares.
        if random.random() < 2.0 / 3.0:
            accion = (accion + random.choice([-1, 1])) % 4

        # Transición determinista tras aplicar el viento
        r, c = divmod(self.estado_actual, 4)
        if accion == 0:
            c = max(0, c - 1)
        elif accion == 1:
            r = min(3, r + 1)
        elif accion == 2:
            c = min(3, c + 1)
        elif accion == 3:
            r = max(0, r - 1)

        nuevo_estado = r * 4 + c
        self.estado_actual = nuevo_estado

        # TAREA 2: recompensas según el esquema activado
        terminado = False
        if nuevo_estado in self.agujeros:
            terminado = True
            recompensa = -1.0 if self.reward_shaping else 0.0
        elif nuevo_estado == self.meta:
            terminado = True
            recompensa = 1.0
        else:
            recompensa = -0.01 if self.reward_shaping else 0.0

        return nuevo_estado, recompensa, terminado


def accion_greedy(q_table: np.ndarray, estado: int) -> int:
    """Acción de mayor valor Q; los empates se rompen al azar
    (con Q=0 al inicio, argmax elegiría siempre la acción 0)."""
    fila = q_table[estado, :]
    mejores = np.flatnonzero(fila == fila.max())
    return int(np.random.choice(mejores))


def entrenar_q_learning(env, episodios=3000, alpha=0.1, gamma=0.95,
                        eps_ini=1.0, eps_min=0.01, eps_decay=0.995,
                        max_pasos=100) -> Tuple[np.ndarray, List[int]]:
    """Entrena con Q-Learning. Devuelve la tabla Q y el historial de éxitos
    (1 si el episodio terminó en la meta, 0 en caso contrario)."""
    q_table = np.zeros((env.n_estados, env.n_acciones))
    epsilon = eps_ini
    historial_exitos: List[int] = []

    for ep in range(episodios):
        state = env.reset()
        done = False
        pasos = 0
        llego_a_meta = False

        while not done and pasos < max_pasos:
            # Política ε-greedy
            if random.uniform(0, 1) < epsilon:
                action = random.randint(0, env.n_acciones - 1)   # explorar
            else:
                action = accion_greedy(q_table, state)           # explotar

            next_state, reward, done = env.step(action)

            # Actualización TD (Bellman sin modelo)
            # En estado terminal no hay valor futuro.
            futuro = 0.0 if done else np.max(q_table[next_state, :])
            td_target = reward + gamma * futuro
            td_error = td_target - q_table[state, action]
            q_table[state, action] += alpha * td_error

            if done and next_state == env.meta:
                llego_a_meta = True
            state = next_state
            pasos += 1

        historial_exitos.append(1 if llego_a_meta else 0)
        # Decaimiento exponencial de ε por episodio
        epsilon = max(eps_min, epsilon * eps_decay)

    return q_table, historial_exitos


def tasa_por_bloques(exitos: List[int], tam_bloque: int = 300) -> List[float]:
    return [float(np.mean(exitos[i:i + tam_bloque]))
            for i in range(0, len(exitos), tam_bloque)]


def evaluar_politica(env, q_table, episodios=2000, max_pasos=100) -> float:
    """Tasa de éxito de la política greedy (sin exploración)."""
    exitos = 0
    for _ in range(episodios):
        s, done, pasos = env.reset(), False, 0
        while not done and pasos < max_pasos:
            s, _, done = env.step(accion_greedy(q_table, s))
            pasos += 1
        exitos += int(done and s == env.meta)
    return exitos / episodios


def mostrar_politica(env, q_table) -> None:
    for r in range(4):
        fila = ""
        for c in range(4):
            s = r * 4 + c
            if s in env.agujeros:
                fila += " H "
            elif s == env.meta:
                fila += " G "
            else:
                fila += f" {SIMBOLOS_ACCION[int(np.argmax(q_table[s, :]))]} "
        print("   " + fila)


if __name__ == "__main__":
    print("=== EVALUACIÓN EJERCICIO 2: Q-LEARNING Y REWARD SHAPING ===")
    EPISODIOS, BLOQUE, SEMILLAS = 3000, 300, 5

    acum = {"Estándar": [], "Reward Shaping": []}
    evals = {"Estándar": [], "Reward Shaping": []}
    q_demo = {}

    for semilla in range(SEMILLAS):
        for nombre, shaping in (("Estándar", False), ("Reward Shaping", True)):
            random.seed(semilla)
            np.random.seed(semilla)
            env = EntornoGridworldEstocastico(reward_shaping=shaping)
            q, exitos = entrenar_q_learning(env, episodios=EPISODIOS)
            acum[nombre].append(tasa_por_bloques(exitos, BLOQUE))
            # La evaluación usa siempre el entorno estándar para comparar de forma justa
            evals[nombre].append(evaluar_politica(EntornoGridworldEstocastico(False), q))
            if semilla == 0:
                q_demo[nombre] = q

    print(f"\nTasa de éxito media por bloques de {BLOQUE} episodios "
          f"(promedio de {SEMILLAS} semillas):")
    print(f"{'Episodios':>14} | {'Estándar':>9} | {'Reward Shaping':>14}")
    n_bloques = EPISODIOS // BLOQUE
    for b in range(n_bloques):
        ini, fin = b * BLOQUE + 1, (b + 1) * BLOQUE
        e = np.mean([x[b] for x in acum["Estándar"]])
        s = np.mean([x[b] for x in acum["Reward Shaping"]])
        print(f"{ini:>6}-{fin:<7} | {e:>9.3f} | {s:>14.3f}")

    print("\nTasa de éxito de la política final (greedy, 2000 episodios, media de semillas):")
    for nombre in evals:
        print(f"  {nombre:<15}: {np.mean(evals[nombre]):.3f}")

    print("\nTabla Q Entorno Estándar (Estado 0):", np.round(q_demo["Estándar"][0, :], 3))
    print("Tabla Q Reward Shaping   (Estado 0):", np.round(q_demo["Reward Shaping"][0, :], 3))

    env_ref = EntornoGridworldEstocastico()
    print("\nPolítica aprendida - Estándar:")
    mostrar_politica(env_ref, q_demo["Estándar"])
    print("\nPolítica aprendida - Reward Shaping:")
    mostrar_politica(env_ref, q_demo["Reward Shaping"])