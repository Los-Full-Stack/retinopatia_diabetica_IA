# Despliegue en Streamlit Community Cloud

## Archivos necesarios

- `streamlit_app.py`
- `requirements.txt`
- `packages.txt`
- `.streamlit/config.toml`
- `.streamlit/secrets.example.toml`
- `models/best_retina_model.pth` disponible en el entorno

## Modelo

El checkpoint actual pesa aproximadamente 71 MB:

```text
models/best_retina_model.pth
```

Opciones:

1. Usar Git LFS para versionarlo.
2. Alojarlo externamente y descargarlo antes de iniciar la app.
3. Subirlo manualmente al entorno si la plataforma lo permite.

El repositorio actual ignora `models/*.pth`, asi que si usas Git LFS debes ajustar esa regla.

## Secrets

En Streamlit Cloud configura:

```toml
[auth]
username = "admin"
password = "cambia-esta-clave"
```

No subas `.streamlit/secrets.toml`.

## Comando local

```powershell
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

## Limitaciones

- Sin base de datos: pacientes e historial viven en memoria de sesion.
- En CPU la inferencia puede tardar mas.
- Es prototipo de investigacion, no diagnostico autonomo.
