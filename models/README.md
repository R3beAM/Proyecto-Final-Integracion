# Modelos versionados con DVC

Esta carpeta está destinada a guardar modelos entrenados, vectorizadores y otros artefactos binarios del sistema de análisis de sentimiento.

Los artefactos reales se excluyen de Git y deben registrarse con DVC, por ejemplo:

```bash
dvc add models/sentiment_model.pkl
git add models/sentiment_model.pkl.dvc models/.gitignore
git commit -m "Track sentiment model with DVC"
```
