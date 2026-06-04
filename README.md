# 🧠 Predicción de Adicción a Redes Sociales en Estudiantes

Este proyecto académico presenta una solución integral de **Ciencia de Datos y Machine Learning** para predecir, clasificar e identificar los factores determinantes del riesgo de adicción a las redes sociales en estudiantes de educación secundaria y superior (de 18 a 24 años). A través del desarrollo de un pipeline completo de ETL, análisis exploratorio de datos (EDA) y tres enfoques de modelado predictivo, el proyecto culmina en un dashboard web interactivo implementado en Flask que reentrena y evalúa los modelos de forma dinámica.

---

## 📁 Estructura del Proyecto

El repositorio está organizado de la siguiente manera:

```
TrabajoFinal/PoryectoFinal/
├── adiccion_etl.ipynb                  # 🧹 ETL: Limpieza, preparación y transformación del dataset.
├── adiccion_eda.ipynb                  # 📊 EDA: Análisis estadístico exploratorio, gráficos y ANOVA.
├── adiccion_regresion_lineal.ipynb     # 📈 Regresión Lineal: Predicción de puntaje continuo de adicción.
├── adiccion_regresion_logistica.ipynb   # 🎯 Regresión Logística: Clasificación de alto riesgo binario.
├── adiccion_arboles_decision.ipynb     # 🌿 Árbol de Decisión: Clasificación multiclase y reglas de negocio.
├── app.py                              # 🐍 Backend de Flask con APIs REST de predicción y estadísticas.
├── index.html                          # 🖥️ Frontend del dashboard web (UI/UX interactiva tipo Single Page).
├── style.css                           # 🎨 Hoja de estilos con diseño oscuro responsive y animaciones.
├── requirements.txt                    # 📦 Lista de dependencias del proyecto.
├── README.md                           # 📖 Este documento de documentación profesional.
└── Students Social Media Addiction.csv # 📊 Dataset crudo original con 705 registros.
```

---

## 📊 Dataset

El archivo original `Students Social Media Addiction.csv` consta de **705 registros** y **13 variables** que describen a la población de estudio.

### Diccionario de Datos

| Columna Original | Alias en Español | Tipo de Dato | Rango / Valores | Descripción |
| :--- | :--- | :--- | :--- | :--- |
| `Student_ID` | *Eliminada* | Entero | `1 - 705` | Identificador de estudiante (excluido en ETL por no ser predictivo). |
| `Age` | `edad` | Entero | `18 - 24` | Edad del estudiante en años. |
| `Gender` | `genero` / `genero_enc` | Categórico | `Female (0)`, `Male (1)` | Género auto-reportado. |
| `Academic_Level` | `nivel_academico` / `enc` | Categórico | `High School (0)`, `Undergraduate (1)`, `Graduate (2)` | Nivel académico actual del alumno. |
| `Country` | `pais` / `pais_enc` | Categórico | 110 países | País de origen del estudiante. |
| `Avg_Daily_Usage_Hours`| `horas_uso_diario` | Decimal | `1.5 - 8.5` | Promedio de horas diarias de uso de redes sociales. |
| `Most_Used_Platform` | `plataforma` / `enc` | Categórico | 12 plataformas | Red social en la que pasa más tiempo. |
| `Affects_Academic_Performance`| `afecta_rendimiento` / `enc`| Categórico | `No (0)`, `Yes (1)` | Reporte sobre si las redes afectan sus estudios. |
| `Sleep_Hours_Per_Night`| `horas_sueno_noche` | Decimal | `3.8 - 9.6` | Promedio de horas de sueño nocturno. |
| `Mental_Health_Score` | `puntaje_salud_mental` | Entero | `4 - 9` | Autoevaluación de salud mental (menor es peor). |
| `Relationship_Status` | `estado_relacion` / `enc` | Categórico | 3 estados | Estado sentimental actual del alumno. |
| `Conflicts_Over_Social_Media`| `conflictos_por_redes`| Entero | `0 - 5` | Frecuencia de discusiones por redes (mayor es peor). |
| `Addicted_Score` | `puntaje_adiccion` | Entero | `2 - 9` | Puntaje objetivo de nivel de adicción. |

---

## 🤖 Modelos de Machine Learning

Se implementaron tres modelos bajo paradigmas diferentes para responder a preguntas de negocio distintas:

| Modelo | Variable Objetivo | Mapeo / Rango | Métricas Principales | Interpretación Clave |
| :--- | :--- | :--- | :--- | :--- |
| **Regresión Lineal** | `puntaje_adiccion` | Continuo (`2.0 - 9.0`) | $R^2 \approx 0.89$, $RMSE \approx 0.52$, $MAE \approx 0.43$ | Estima **cuánto** puntaje de adicción presentará un estudiante en base a sus hábitos cotidianos. |
| **Regresión Logística** | `High_Addiction_Risk` | Binario (`0` o `1`) | $Exactitud \approx 94\%$, $Sensibilidad (Recall) \approx 97\%$, $AUC-ROC \approx 0.99$ | Determina **si está o no** en un nivel crítico de adicción (puntaje $\ge 7$) priorizando reducir los falsos negativos. |
| **Árbol de Decisión** | `nivel_adiccion_enc` | Multiclase (`0`: Baja, `1`: Media, `2`: Alta) | $Exactitud \approx 92\%$, $Macro F1 \approx 0.91$ | Explica **por qué** un estudiante cae en cierta categoría a través de reglas lógicas interpretables. |

---

## 🚀 Instalación y Uso

### A. Google Colab (Cuadernos de Jupyter)

1. Abre [Google Colab](https://colab.research.google.com/).
2. Sube los cuadernos en orden numérico: `adiccion_etl.ipynb`, `adiccion_eda.ipynb`, `adiccion_regresion_lineal.ipynb`, `adiccion_regresion_logistica.ipynb` y `adiccion_arboles_decision.ipynb`.
3. Para cada cuaderno, al iniciar, sube el archivo de datos original (`Students Social Media Addiction.csv`) o el archivo generado por el ETL según corresponda, utilizando la barra lateral de archivos (directorio `/content/`).
4. Haz clic en **Entorno de ejecución** > **Ejecutar todo** (o presiona `Ctrl + F9`).
5. Los cuadernos descargarán automáticamente las gráficas generadas (`.png`) y los CSV procesados para que los uses en el siguiente paso.

### B. Aplicación Web Local (Flask Dashboard)

Asegúrate de contar con Python 3.10 o superior instalado en tu sistema local.

1. Abre tu terminal de comandos en el directorio del proyecto `TrabajoFinal/PoryectoFinal/`.
2. Instala las dependencias y bibliotecas necesarias ejecutando:
   ```bash
   pip install -r requirements.txt
   ```
3. Ejecuta el servidor de Flask:
   ```bash
   python app.py
   ```
4. Abre tu navegador de preferencia e ingresa a la siguiente dirección:
   ```
   http://127.0.0.1:5000/
   ```
5. ¡Interactúa con los sliders, visualiza los gráficos dinámicos de Chart.js y obtén predicciones en tiempo real reentrenando los modelos al instante!

---

## 📈 Resultados Principales (Hallazgos del Análisis)

1. **Horas de Pantalla como Predictores**: El número de horas que un estudiante pasa en redes diariamente (`Avg_Daily_Usage_Hours`) es el factor positivo más correlacionado con la adicción ($r \approx 0.72$). Consumos superiores a 6 horas elevan el riesgo exponencialmente.
2. **Impacto en el Sueño**: Existe una fuerte correlación negativa entre las horas de sueño (`Sleep_Hours_Per_Night`) y la adicción digital ($r \approx -0.74$). Estudiantes con puntaje de adicción crítico duermen en promedio menos de 5.5 horas por noche.
3. **Salud Mental y Conectividad**: Un menor puntaje de salud mental se vincula fuertemente con adicciones graves. Los coeficientes de regresión logística revelan que por cada punto de decremento en bienestar mental, los odds de adicción se elevan en un $1.4\times$.
4. **Frecuencia de Conflictos**: Las disputas o discusiones familiares y de pareja vinculadas al teléfono celular (`Conflicts_Over_Social_Media`) actúan como un fuerte síntoma temprano de adicción digital; es el segundo coeficiente más alto en importancia de árbol.
5. **Diferenciación de Plataformas**: Instagram y TikTok son las plataformas de redes con los promedios de adicción estudiantil más elevados y que concentran una mayor tasa de usuarios calificados en "Alto Riesgo", duplicando a redes profesionales como LinkedIn o VKontakte.

---

## 🛠️ Tecnologías Utilizadas

* ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
* ![Flask](https://img.shields.io/badge/flask-%23000.svg?style=for-the-badge&logo=flask&logoColor=white)
* ![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
* ![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
* ![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
* ![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)
* ![Chart.js](https://img.shields.io/badge/chart.js-F5788D.svg?style=for-the-badge&logo=chart.js&logoColor=white)

---

## ⚠️ Limitaciones

1. **Datos Auto-reportados**: Gran parte del dataset se basa en cuestionarios donde los alumnos estiman subjetivamente sus horas de sueño, uso y salud mental, lo cual puede incorporar sesgo de deseabilidad social.
2. **Estudio Transversal**: El dataset recopila la información en un único punto temporal, por lo que los modelos estadísticos denotan asociaciones y correlaciones fuertes, pero no pueden probar causalidad directa.
3. **Distribución de Edades**: La muestra se concentra estrictamente en universitarios de 18 a 24 años, por lo que los resultados no pueden generalizarse directamente a niños de primaria o adultos mayores.

---

## 📄 Licencia

Este proyecto está bajo la Licencia MIT - consulte el archivo de licencia para obtener más detalles. Desarrollado con fines educativos y de investigación académica.
