# Análisis de sentimiento político en tweets en español

Este repositorio contiene el código incluido en mi trabajo final de máster. El trabajo consiste en la creación de un modelo de clasificación multiclase capaz de diferenciar la orientación política que se denota en tweets escritos en español. Para ello, se emplea el dataset IberLEF 2023. 


## Resultados

Los resultados obtenidos para la clasificación de la variable multiclase (political-ideology) empleando diferentes arquitecturas son los siguientes

| Nombre del modelo | F1 score | Pérdida | Precisión |
| ----------------- | --------:| -------:| ---------:|
| xlm-roberta-base  | 0.122    | 0.694   | 0.122     |
| bert-base         | 0.392    | 0.826   | 0.392     |
| newtral-xlm       | 0.400    | 0.531   | 0.400     |
| gpt3-api-ada      | 0.429    | 0.012   | 0.440     |
| beto-base-uncased | 0.447    | 0.497   | 0.447     |
| beto-base-cased   | 0.483    | 1.452   | 0.483     |
		