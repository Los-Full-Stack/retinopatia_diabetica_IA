# RetinaAI - Informe de Construccion Actualizado

## 1. Resumen Ejecutivo

RetinaAI es un prototipo de apoyo optometrico para la evaluacion de imagenes retinales orientado a la deteccion de retinopatia diabetica. El sistema permite registrar pacientes, cargar una imagen retinal, ejecutar un modelo de inteligencia artificial entrenado en PyTorch y presentar una sugerencia de severidad junto con una politica de seguridad clinica.

El sistema no reemplaza la evaluacion profesional. Su salida se presenta como apoyo visual y orientativo para el optometra, incluyendo control de calidad de imagen, nivel de confianza, recomendacion de accion y un mapa Grad-CAM que muestra las regiones de la imagen que influyeron en la prediccion del modelo.

## 2. Objetivo del Proyecto

Desarrollar una aplicacion web capaz de asistir al optometra en la evaluacion inicial de imagenes retinales mediante un modelo de clasificacion multiclase de retinopatia diabetica, incorporando mecanismos de seguridad para evitar el uso automatico de predicciones de baja confianza.

## 3. Alcance

El alcance actual incluye:

- Login demo sin base de datos.
- Registro local de pacientes.
- Carga de imagen retinal en formato JPG o PNG.
- Clasificacion en cinco grados de retinopatia diabetica.
- Evaluacion de calidad de imagen.
- Politica de decision `accept`, `review` y `reject`.
- Visualizacion Grad-CAM como mapa de atencion del modelo.
- Historial local en el navegador.
- Descarga de resultado en JSON.
- Despliegue preparado para Render con FastAPI.

No incluye:

- Base de datos clinica real.
- Autenticacion productiva.
- Validacion clinica formal multicentrica.
- Aprobacion como dispositivo medico.
- Diagnostico medico automatico.

## 4. Dataset y Preparacion de Datos

El modelo final protegido se entreno con el dataset original del proyecto, evitando mezclar datos externos en el checkpoint final para reducir el riesgo de ruido por diferencias de dominio.

Durante el desarrollo se evaluaron datasets externos y nuevas fuentes de datos, pero los experimentos no superaron de forma segura al modelo base. Por esta razon, se congelo el checkpoint bueno actual y se priorizo convertirlo en un producto demo confiable con politica de abstencion.

La preparacion de datos incluyo:

- Validacion de imagenes.
- Preprocesamiento retinal.
- Redimensionamiento.
- Normalizacion.
- Aumentacion de datos durante entrenamiento.
- Division en conjuntos de entrenamiento, validacion y prueba.

## 5. Modelo de Inteligencia Artificial

El sistema utiliza un modelo basado en EfficientNet con PyTorch. El checkpoint final se encuentra en:

```text
models/best_retina_model.pth
```

Caracteristicas principales:

- Framework: PyTorch.
- Arquitectura base: EfficientNet.
- Tarea: clasificacion multiclase.
- Clases: 5 grados de retinopatia diabetica.
- Tamano del checkpoint: aproximadamente 71.26 MB.
- Inferencia con TTA activado.
- Calibracion mediante temperatura.

Clases del modelo:

| Grado | Clase |
|---:|---|
| 0 | Sin retinopatia diabetica |
| 1 | Retinopatia diabetica leve |
| 2 | Retinopatia diabetica moderada |
| 3 | Retinopatia diabetica severa |
| 4 | Retinopatia diabetica proliferativa |

## 6. Metricas Actuales del Modelo

Metricas del checkpoint bueno actual:

| Metrica | Valor |
|---|---:|
| Accuracy | 0.7909 |
| Balanced accuracy | 0.6733 |
| F1 macro | 0.6504 |
| F1 weighted | 0.7957 |
| Kappa | 0.8616 |
| ROC AUC multiclass | 0.9201 |
| Error grande >=2 grados | 0.0455 |

Recall por clase:

| Clase | Recall |
|---:|---:|
| 0 | 0.9705 |
| 1 | 0.6429 |
| 2 | 0.6333 |
| 3 | 0.5517 |
| 4 | 0.5682 |

Interpretacion:

El modelo tiene buen rendimiento general y buen Kappa, pero presenta menor sensibilidad en clases 2, 3 y 4. Por ello, no se utiliza como diagnostico automatico. En su lugar, se aplica una politica de seguridad que permite aceptar solo predicciones suficientemente confiables y deriva o rechaza los casos ambiguos.

## 7. Politica de Seguridad Clinica

RetinaAI incorpora una politica de decision para evitar que predicciones debiles sean presentadas como definitivas.

Estados:

| Estado | Significado | Accion |
|---|---|---|
| `accept` | Prediccion confiable | Puede usarse como apoyo automatico supervisado |
| `review` | Prediccion orientativa | Requiere revision profesional |
| `reject` | Prediccion no confiable | Repetir imagen o derivar |

Auditoria sobre test local:

| Estado | Casos | Cobertura | Accuracy si se usa | Error grande |
|---|---:|---:|---:|---:|
| `accept` | 219 | 0.3982 | 0.9772 | 0.0091 |
| `review` | 8 | 0.0145 | 0.8750 | 0.1250 |
| `reject` | 323 | 0.5873 | 0.6563 | 0.1176 |

Conclusion operativa:

La politica sacrifica cobertura para mejorar confiabilidad en los casos aceptados. Esto es adecuado para una demo clinica, porque evita forzar predicciones cuando la imagen es borrosa, la confianza es baja o las clases principales estan demasiado cercanas.

## 8. Grad-CAM

Se implemento Grad-CAM para agregar explicabilidad visual al sistema.

Grad-CAM genera un mapa de calor sobre la imagen retinal que indica las regiones que mas influyeron en la prediccion del modelo. En la interfaz se muestra como `Mapa IA`, junto a la imagen original.

Implementacion tecnica:

- Se engancha un hook sobre la ultima capa convolucional de EfficientNet: `model.features[-1]`.
- Se calcula el gradiente de la clase predicha.
- Se ponderan los mapas de activacion por el promedio de los gradientes.
- Se normaliza el mapa resultante.
- Se aplica un colormap.
- Se superpone el mapa sobre la imagen retinal preprocesada.
- Se devuelve como PNG codificado en base64 dentro de la respuesta de `/predict`.

Advertencia:

Grad-CAM no es una explicacion clinica definitiva. Es una ayuda visual para mostrar zonas de atencion del modelo. La interpretacion final debe ser realizada por el profesional de salud visual.

## 9. Arquitectura del Sistema

Arquitectura actual:

```text
Usuario optometrico
        |
        v
Frontend HTML/CSS/JS
        |
        v
FastAPI
        |
        v
RetinaPredictor
        |
        v
Modelo PyTorch EfficientNet
        |
        v
Prediccion + Politica + Grad-CAM
```

Componentes principales:

| Componente | Funcion |
|---|---|
| `webapp/main.py` | Backend FastAPI |
| `webapp/static/index.html` | Login demo |
| `webapp/static/dashboard.html` | Panel optometrico |
| `webapp/static/app.js` | Logica de interfaz |
| `webapp/static/styles.css` | Estilos visuales |
| `src/inference.py` | Inferencia, politica y Grad-CAM |
| `src/model.py` | Construccion del modelo |
| `models/best_retina_model.pth` | Checkpoint final |

## 10. Interfaz de Usuario

La interfaz fue disenada para un usuario final optometrico, no para un desarrollador.

Por esta razon se retiraron metricas internas como accuracy, F1 o error grande del dashboard. Estas metricas permanecen en la documentacion tecnica, pero no en la interfaz final.

Flujo de uso:

1. Iniciar sesion con credenciales demo.
2. Crear o seleccionar paciente.
3. Cargar imagen retinal.
4. Revisar imagen original o `Mapa IA`.
5. Leer decision del sistema.
6. Guardar resultado en historial local o descargar JSON.

## 11. Despliegue

La aplicacion esta preparada para Render.

Archivo principal de configuracion:

```text
render.yaml
```

Comando de arranque:

```text
uvicorn webapp.main:app --host 0.0.0.0 --port $PORT
```

El modelo se incluye en el repositorio:

```text
models/best_retina_model.pth
```

Como pesa aproximadamente 71.26 MB, puede subirse a GitHub sin Git LFS mientras se mantenga por debajo del limite de 100 MB por archivo.

## 12. Pruebas

Pruebas ejecutadas:

- Compilacion de `src/inference.py`.
- Validacion de sintaxis de `webapp/static/app.js`.
- Pruebas unitarias con `pytest`.
- Prueba HTTP de `/health`.
- Prueba HTTP de `/predict`.
- Validacion de generacion Grad-CAM con checkpoint real.

Resultado actual:

```text
7 passed
```

Validacion Grad-CAM:

```text
method = Grad-CAM
overlay = data:image/png;base64,...
available = True
```

## 13. Limitaciones

Limitaciones tecnicas y clinicas:

- El modelo no debe usarse como diagnostico medico definitivo.
- El rendimiento es menor en clases 2, 3 y 4.
- La calidad de imagen afecta directamente la confiabilidad.
- Grad-CAM puede resaltar regiones no necesariamente equivalentes a lesiones clinicas.
- Se requiere validacion con especialistas antes de cualquier uso real.
- El sistema actual usa almacenamiento local en navegador, no historia clinica real.

## 14. Conclusiones

Se construyo un sistema funcional de apoyo optometrico para la evaluacion retinal asistida por inteligencia artificial. El proyecto integra un modelo PyTorch entrenado, una politica de seguridad clinica, control de calidad de imagen, explicabilidad visual mediante Grad-CAM y una interfaz web preparada para despliegue en Render.

El enfoque final prioriza la confiabilidad sobre la automatizacion completa. En lugar de forzar todas las predicciones, RetinaAI acepta solo casos de mayor seguridad y deriva los casos ambiguos o de baja calidad, lo cual resulta mas adecuado para una aplicacion de apoyo profesional.

## 15. Recomendaciones

- Mantener ocultas las metricas internas al usuario final.
- Presentar Grad-CAM como ayuda visual, no como prueba clinica.
- Recolectar mas datos reales del dominio objetivo.
- Validar con optometras y oftalmologos.
- Agregar base de datos y autenticacion real si el sistema pasa de demo a producto.
- Mantener versionado del modelo y auditoria de predicciones.
- No reemplazar el checkpoint bueno sin comparacion contra el test original.
