from flask import Flask, request, jsonify
import cv2
import numpy as np
import mediapipe as mp

app = Flask(__name__)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)

# Detectar puntos
def analizar_postura(imagen):
    imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
    resultado = pose.process(imagen_rgb)
    return resultado

@app.route("/analizar", methods=["POST"])
def analizar():
    try:
        if 'imagen' not in request.files:
            return jsonify({"error": "No se encontró el archivo 'imagen'"}), 400

        altura_ciclista = request.form.get("altura_ciclista")
        if not altura_ciclista:
            return jsonify({"error": "Falta el campo 'altura_ciclista'"}), 400

        try:
            altura_ciclista = float(altura_ciclista)
        except ValueError:
            return jsonify({"error": "El campo 'altura_ciclista' debe ser numérico"}), 400

        archivo = request.files['imagen']
        npimg = np.frombuffer(archivo.read(), np.uint8)
        imagen = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        if imagen is None:
            return jsonify({"error": "No se pudo decodificar la imagen"}), 400

        # 🔧 REDIMENSIONAR imagen si es demasiado grande
        h, w = imagen.shape[:2]
        if max(h, w) > 720:
            factor = 720.0 / max(h, w)
            imagen = cv2.resize(imagen, (int(w * factor), int(h * factor)))

        resultado = analizar_postura(imagen)
        if not resultado or not resultado.pose_landmarks:
            return jsonify({"error": "No se detectaron puntos clave"}), 400

        puntos = resultado.pose_landmarks.landmark

        # Cálculo de alturas
        y_cadera = puntos[mp_pose.PoseLandmark.LEFT_HIP].y
        y_talon = puntos[mp_pose.PoseLandmark.LEFT_HEEL].y
        y_cabeza = puntos[mp_pose.PoseLandmark.NOSE].y

        altura_total_px = abs(y_talon - y_cabeza)
        altura_sillin_px = abs(y_talon - y_cadera)

        px_por_cm = altura_total_px / altura_ciclista
        altura_sillin_cm = altura_sillin_px / px_por_cm

        entrepierna = altura_ciclista * 0.45
        altura_recomendada = entrepierna * 0.883
        diferencia = altura_recomendada - altura_sillin_cm

        if abs(diferencia) < 1:
            recomendacion = "✅ Tu sillín está bien ajustado."
        elif diferencia > 0:
            recomendacion = f"🔼 Sube el sillín aproximadamente {diferencia:.1f} cm."
        else:
            recomendacion = f"🔽 Baja el sillín aproximadamente {abs(diferencia):.1f} cm."

        return jsonify({
            "altura_ciclista_cm": altura_ciclista,
            "altura_sillin_actual_cm": round(altura_sillin_cm, 2),
            "altura_sillin_recomendada_cm": round(altura_recomendada, 2),
            "diferencia_cm": round(diferencia, 2),
            "recomendacion": recomendacion
        }), 200

    except Exception as e:
        return jsonify({"error": f"Error inesperado: {str(e)}"}), 500

@app.route("/", methods=["GET"])
def ping():
    return "✅ API funcionando correctamente", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
