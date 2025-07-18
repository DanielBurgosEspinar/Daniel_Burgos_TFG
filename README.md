# Control de Robots Cuadrúpedos Mediante RL
Este repositorio forma parte del trabajo de fin de grado llevado a cabo por Daniel Burgos Espinar. El objetivo principal es mostrar los códigos y modelos de la experimentacion para su análisis. Cabe destacar que durante la investigación de este trabajo se han realizado muchas pruebas, pero por claridad se ha optado por seleccionar solo las que han sido destacadas durante la memoria.  
El repositorio está dividido en las siguinetes carpetas;
1. Entorno Ant: se incluyen su version de caminar en el eje X(ant_movex) como su version de ir a un punto XY (ant_XY)
2. Entorno Go1: se incluyen su version de caminar en el eje X(go1_moveX) como las 2 versiones de ir a un punto XY (go1_XY_v1 y go1_XY_v2)
3. Entornos de los tutoriales realizados para practicar: incluye los códigos de CartPole(CartPole.py) y PandaPickandPlace-v3(PandaPickandPlace.py)

## Instalación del entorno
Para poder ejecutar los códigos se ha dispuesto de un entorno.yml que permite exportar el entorno conda utilizado en un sistema Ubuntu 20.04.6 LTS.
```bash
conda env create -f entorno.yml -n nombre_que_quieras
```
## Ant
Dentro de cada una de las carpetas se encuentra un código de Python junto con el modelo entrenado y los logs correspondientes a los entrenamientos.
### ant_movex
Contiene el mejor modelo entrenado y los logs con diversos algoritmos para su comparacion.
Para evaluar el modelo, desde dentro de la carpeta:
```bash
python ant_movex.py --run test --model_path [copiar el path de "best_model"]
```

### ant_XY
Contiene el mejor modelo entrenado y los logs con diversos algoritmos para su comparacion.
Para evaluar el modelo, desde dentro de la carpeta:
```bash
python ant_XY.py --run test --model_path [copiar el path de "best_model"]
```

## Go1
Dentro de cada una de las carpetas se encuentra un código de Python junto con el modelo entrenado, los logs correspondientes a los entrenamientos, los archivos xml paar el entrono, un codigo train_XX.py para ejecutarlo y el codigo go1_mujoco_env_XX que contiene las recompensas.
### go1_movex
Para evaluar el modelo, desde dentro de la carpeta:
```bash
python train_movex.py --run test --model_path [copiar el path de "best_model"]
```
### go1_XY_v1
Para evaluar el modelo, desde dentro de la carpeta:
```bash
python train_v1.py --run test --model_path [copiar el path de "best_model"]
```
### go1_XY_v2
Para evaluar el modelo, desde dentro de la carpeta:
```bash
python train_final.py --run test --model_path [copiar el path de "best_model"]
```
## Tutoriales
Simplemente contiene los código que con ejecutarlos de manera normal es suficiente

## Gráficas
Para comprobar las gráficas de entrenamiento de los difreentes entrenamientos:
```bash
tensorboard --logdir [copiar el path de la carpeta logs que se quiera visualizar]
```
