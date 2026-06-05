from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os

app = Flask(__name__, template_folder='.', static_folder='.', static_url_path='')
CORS(app)

def cargar_dataset():
    # Intentar cargar el archivo con espacio o con guion bajo
    for nombre in ['Students Social Media Addiction.csv', 'Students_Social_Media_Addiction.csv']:
        if os.path.exists(nombre):
            return pd.read_csv(nombre)
    raise FileNotFoundError("❌ No se encontró 'Students Social Media Addiction.csv' en el directorio local.")

def entrenar_modelos(df):
    df_clean = df.copy()
    
    # Codificación controlada
    df_clean['genero_enc'] = df_clean['Gender'].map({'Female': 0, 'Male': 1, 'Femenino': 0, 'Masculino': 1})
    df_clean['nivel_academico_enc'] = df_clean['Academic_Level'].map({
        'High School': 0, 'Undergraduate': 1, 'Graduate': 2,
        'Bachillerato': 0, 'Pregrado': 1, 'Posgrado': 2
    })
    df_clean['afecta_rendimiento_enc'] = df_clean['Affects_Academic_Performance'].map({
        'No': 0, 'Yes': 1, 'Sí': 1
    })
    
    # LabelEncoder para Most_Used_Platform
    le_platform = LabelEncoder()
    df_clean['plataforma_enc'] = le_platform.fit_transform(df_clean['Most_Used_Platform'])
    
    # Features de entrada suministradas por el formulario
    features = [
        'Avg_Daily_Usage_Hours', 
        'Conflicts_Over_Social_Media', 
        'Sleep_Hours_Per_Night', 
        'Mental_Health_Score', 
        'genero_enc', 
        'nivel_academico_enc', 
        'plataforma_enc', 
        'afecta_rendimiento_enc'
    ]
    
    X = df_clean[features].copy()
    
    # Objetivos
    y_linear = df_clean['Addicted_Score']
    y_logistic = (df_clean['Addicted_Score'] >= 7).astype(int)
    
    # nivel_adiccion_enc (Baja=0, Media=1, Alta=2)
    def get_level(score):
        if score <= 4:
            return 0
        elif score <= 6:
            return 1
        else:
            return 2
    y_tree = df_clean['Addicted_Score'].apply(get_level)
    
    # Escalamiento de numéricas
    num_features = ['Avg_Daily_Usage_Hours', 'Conflicts_Over_Social_Media', 'Sleep_Hours_Per_Night', 'Mental_Health_Score']
    scaler = StandardScaler()
    X[num_features] = scaler.fit_transform(X[num_features])
    
    # Modelos
    model_linear = LinearRegression()
    model_linear.fit(X, y_linear)
    
    model_logistic = LogisticRegression(class_weight='balanced', max_iter=1000)
    model_logistic.fit(X, y_logistic)
    
    model_tree = DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_split=10, min_samples_leaf=5)
    model_tree.fit(X, y_tree)
    
    return model_linear, model_logistic, model_tree, scaler, le_platform, features

def generar_recomendaciones(puntaje, prob, nivel, salud_mental, conflictos):
    recs = []
    if puntaje >= 7.0 or nivel == 'Alta':
        recs.append("Reducir el tiempo diario en redes sociales: Establecer límites estrictos de pantalla y realizar ayunos digitales semanales.")
    if salud_mental < 6:
        recs.append("Buscar apoyo profesional: Se sugiere consultar con servicios de orientación psicológica universitaria para fortalecer el bienestar mental.")
    if conflictos > 3:
        recs.append("Talleres de comunicación y asertividad digital: Participar en sesiones que aborden la resolución de conflictos interpersonales vinculados a la conectividad.")
    
    # Completar hasta tener exactamente 3 recomendaciones
    default_recs = [
        "Fomentar actividades offline: Dedicar tiempo a pasatiempos al aire libre, lectura y socialización cara a cara con amigos y familiares.",
        "Establecer una rutina de sueño libre de dispositivos: Apagar pantallas al menos 30 minutos antes de dormir para mejorar la calidad del descanso.",
        "Monitorear las emociones al usar redes: Identificar qué plataformas causan estrés o ansiedad y silenciar notificaciones no esenciales."
    ]
    for dr in default_recs:
        if len(recs) >= 3:
            break
        if dr not in recs:
            recs.append(dr)
            
    return recs[:3]

@app.route('/')
def home():
    try:
        df = cargar_dataset()
        total_estudiantes = len(df)
        
        # % Alto Riesgo (puntaje >= 7)
        pct_alto_riesgo = round((df['Addicted_Score'] >= 7).mean() * 100, 1)
        
        # Medias
        media_horas_uso = round(df['Avg_Daily_Usage_Hours'].mean(), 1)
        media_salud_mental = round(df['Mental_Health_Score'].mean(), 1)
        
        # Plataforma con más riesgo
        df['High_Risk'] = (df['Addicted_Score'] >= 7).astype(int)
        plat_risk = df.groupby('Most_Used_Platform')['High_Risk'].mean()
        plataforma_mas_riesgo = plat_risk.idxmax()
        
        return render_template('index.html', 
                               total_estudiantes=total_estudiantes,
                               pct_alto_riesgo=pct_alto_riesgo,
                               media_horas_uso=media_horas_uso,
                               media_salud_mental=media_salud_mental,
                               plataforma_mas_riesgo=plataforma_mas_riesgo)
    except Exception as e:
        return f"<h3>Error al cargar el dashboard: {str(e)}</h3>", 500

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Extraer entradas
        horas_uso = float(data.get('horas_uso', 0))
        salud_mental = int(data.get('salud_mental', 5))
        conflictos = int(data.get('conflictos', 0))
        sueno = float(data.get('sueno', 7))
        genero = data.get('genero', 'Female')
        nivel_academico = data.get('nivel_academico', 'Undergraduate')
        plataforma = data.get('plataforma', 'Instagram')
        afecta_rendimiento = data.get('afecta_rendimiento', 'No')
        
        # Cargar datos y entrenar modelos dinámicamente
        df = cargar_dataset()
        model_linear, model_logistic, model_tree, scaler, le_platform, features = entrenar_modelos(df)
        
        # Codificación de la entrada del usuario
        genero_enc = 0 if genero in ['Female', 'Femenino'] else 1
        
        nivel_map = {
            'High School': 0, 'Undergraduate': 1, 'Graduate': 2,
            'Bachillerato': 0, 'Pregrado': 1, 'Posgrado': 2
        }
        nivel_academico_enc = nivel_map.get(nivel_academico, 1)
        
        afecta_rendimiento_enc = 1 if afecta_rendimiento in ['Yes', 'Sí', 'yes', 'si'] else 0
        
        try:
            plataforma_enc = le_platform.transform([plataforma])[0]
        except ValueError:
            # Fallback en caso de que la plataforma no exista en el dataset original
            plataforma_enc = 0
            
        # Escalar variables numéricas de la entrada
        # El orden en el scaler fue: ['Avg_Daily_Usage_Hours', 'Conflicts_Over_Social_Media', 'Sleep_Hours_Per_Night', 'Mental_Health_Score']
        num_vals = np.array([[horas_uso, conflictos, sueno, salud_mental]])
        num_vals_scaled = scaler.transform(num_vals)[0]
        
        # Reconstruir vector de entrada en el orden correcto
        # features = ['Avg_Daily_Usage_Hours', 'Conflicts_Over_Social_Media', 'Sleep_Hours_Per_Night', 'Mental_Health_Score', 'genero_enc', 'nivel_academico_enc', 'plataforma_enc', 'afecta_rendimiento_enc']
        x_input = [
            num_vals_scaled[0],
            num_vals_scaled[1],
            num_vals_scaled[2],
            num_vals_scaled[3],
            genero_enc,
            nivel_academico_enc,
            plataforma_enc,
            afecta_rendimiento_enc
        ]
        
        # Realizar predicciones
        # 1. Regresión Lineal (rango 2-9)
        pred_score = float(model_linear.predict([x_input])[0])
        pred_score = np.clip(pred_score, 2.0, 9.0)
        
        # 2. Regresión Logística (probabilidad de alto riesgo)
        prob_riesgo = float(model_logistic.predict_proba([x_input])[0][1])
        
        # 3. Árbol de Decisión (nivel de adicción)
        pred_tree_idx = int(model_tree.predict([x_input])[0])
        niveles_map = {0: 'Baja', 1: 'Media', 2: 'Alta'}
        nivel_adiccion = niveles_map.get(pred_tree_idx, 'Media')
        
        # Clasificación final de riesgo
        clasificacion_riesgo = 'ALTO' if pred_score >= 7.0 else 'BAJO'
        
        # Factores de riesgo personalizados basados en entradas
        factores_riesgo = []
        if horas_uso >= 5.5:
            factores_riesgo.append(f"Elevadas horas de uso diario de redes ({horas_uso}h/día).")
        if conflictos >= 3:
            factores_riesgo.append(f"Frecuentes conflictos sociales/familiares (nivel {conflictos}/5).")
        if sueno <= 6.0:
            factores_riesgo.append(f"Insuficientes horas de sueño nocturno ({sueno}h).")
        if salud_mental <= 5:
            factores_riesgo.append(f"Puntaje de salud mental vulnerable ({salud_mental}/10).")
        if afecta_rendimiento_enc == 1:
            factores_riesgo.append("Afectación directa en el rendimiento académico estudiantil.")
            
        # Autocompletar factores si hay menos de 3
        globales = [
            "Tiempo excesivo frente a la pantalla digital.",
            "Conflictos interpersonales por hiperconectividad.",
            "Alteración de la higiene del sueño por uso nocturno.",
            "Bajo puntaje de bienestar de salud mental."
        ]
        for g in globales:
            if len(factores_riesgo) >= 3:
                break
            if g not in factores_riesgo:
                factores_riesgo.append(g)
        factores_riesgo = factores_riesgo[:3]
        
        # Generar recomendaciones
        recomendaciones = generar_recomendaciones(pred_score, prob_riesgo, nivel_adiccion, salud_mental, conflictos)
        
        return jsonify({
            'status': 'success',
            'puntaje_predicho': round(pred_score, 2),
            'probabilidad_riesgo': round(prob_riesgo, 4),
            'nivel_adiccion': nivel_adiccion,
            'clasificacion_riesgo': clasificacion_riesgo,
            'factores_riesgo': factores_riesgo,
            'recomendaciones': recomendaciones
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

@app.route('/api/stats')
def stats():
    try:
        df = cargar_dataset()
        
        # 1. Distribución de puntajes (2-9)
        dist = df['Addicted_Score'].value_counts().to_dict()
        distribucion_puntajes = {int(i): int(dist.get(i, 0)) for i in range(2, 10)}
        
        # 2. Riesgo por plataforma
        df['High_Risk'] = (df['Addicted_Score'] >= 7).astype(int)
        riesgo_plat = (df.groupby('Most_Used_Platform')['High_Risk'].mean() * 100).round(1).to_dict()
        
        # Ordenar por porcentaje descendente
        riesgo_por_plataforma = dict(sorted(riesgo_plat.items(), key=lambda x: x[1], reverse=True))
        
        # 3. Riesgo por nivel académico (traducido)
        nivel_map = {
            'High School': 'Bachillerato',
            'Undergraduate': 'Pregrado',
            'Graduate': 'Posgrado'
        }
        df['Nivel_ES'] = df['Academic_Level'].map(nivel_map)
        riesgo_nivel = (df.groupby('Nivel_ES')['High_Risk'].mean() * 100).round(1).to_dict()
        
        # 4. Correlaciones de Pearson con Addicted_Score
        df_corr = df[['Age', 'Avg_Daily_Usage_Hours', 'Sleep_Hours_Per_Night', 'Mental_Health_Score', 'Conflicts_Over_Social_Media', 'Addicted_Score']].copy()
        df_corr.columns = ['Edad', 'Horas Uso Diario', 'Horas Sueño', 'Salud Mental', 'Conflictos Redes', 'Addicted_Score']
        correlaciones = df_corr.corr()['Addicted_Score'].drop('Addicted_Score').round(3).to_dict()
        
        return jsonify({
            'status': 'success',
            'distribucion_puntajes': distribucion_puntajes,
            'riesgo_por_plataforma': riesgo_por_plataforma,
            'riesgo_por_nivel_academico': riesgo_nivel,
            'correlaciones': correlaciones
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    print(f"🚀 Iniciando servidor Flask en puerto {port}")
    app.run(debug=debug, port=port, host='0.0.0.0')
