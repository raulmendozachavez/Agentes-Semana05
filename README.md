# Agentes Inteligentes: Arquitectura BDI y Q-Learning

Dos programas en Python que muestran cómo un agente puede decidir con reglas (agente BDI) y cómo puede aprender por experiencia (Q-Learning con *reward shaping*).a

## Tabla de contenidos

- [Características principales](#características-principales)
- [Tecnologías utilizadas](#tecnologías-utilizadas)
- [Prerrequisitos e instalación](#prerrequisitos-e-instalación)
- [Instrucciones de uso](#instrucciones-de-uso)
- [Estructura del proyecto](#estructura-del-proyecto)
- [Autores y colaboradores](#autores-y-colaboradores)

## Características principales

**Ejercicio 1: Agente BDI (Semana 03)**
- Detecta una fuga de memoria a partir de las últimas 3 lecturas de CPU, antes de llegar al 80 %.
- Atiende primero el problema más importante según su prioridad.
- Cambia a un plan alternativo (*Circuit Breaker*) cuando se agota la cuota de la nube, sin perder su meta.

**Ejercicio 2: Q-Learning (Semana 04)**
- Entrena a un dron en una cuadrícula de 4x4 con viento aleatorio.
- Usa exploración ε-greedy con decaimiento.
- Compara el entorno estándar con el entorno con *reward shaping* y muestra la tasa de éxito por bloques de episodios.

## Tecnologías utilizadas

- Python 3.8 o superior
- NumPy
- Librerías estándar de Python: `dataclasses`, `typing` y `random`

## Prerrequisitos e instalación

1. Instalar [Python 3.8 o superior](https://www.python.org/downloads/).
2. Clonar el repositorio:
```bash
   git clone https://github.com/raulmendozachavez/bdi-rl-ejemplos.git
   cd bdi-rl-ejemplos
```
3. Instalar la librería necesaria (solo la requiere el Ejercicio 2):
```bash
   pip install numpy
```

## Instrucciones de uso

Ejecutar cada programa por separado desde la carpeta del proyecto:

```bash
# Ejercicio 1: Agente BDI
python practica_semana03_bdi.py

# Ejercicio 2: Q-Learning y Reward Shaping
python practica_semana04_rl.py
```

- El primer programa muestra tres escenarios: fuga de memoria, ciberataque con prioridad y cambio de plan por falta de cuota.
- El segundo programa muestra una tabla con la tasa de éxito de ambos entornos y la política aprendida.

## Estructura del proyecto

```
.
├── practica_semana03_bdi.py                  # Ejercicio 1: Agente BDI
├── practica_semana04_rl.py                   # Ejercicio 2: Q-Learning
├── informe_codigos_agentes_inteligentes.docx # Documento explicativo (APA 7)
└── README.md
```

## Autores y colaboradores

- Alvares Paredes, José
- Mendoza Chavez, Raul