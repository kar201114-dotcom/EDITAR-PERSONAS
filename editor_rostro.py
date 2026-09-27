import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="Estudio de Retoque Localizado", layout="wide")

st.title("🎨 Estudio de Retoque con Selección Interactiva")
st.write("Sube una foto, pinta con el mouse sobre la zona que quieras modificar (rostro, labios, cuerpo) y ajusta los efectos.")

# Panel Lateral
st.sidebar.header("Herramientas")
archivo_subido = st.sidebar.file_uploader("Sube tu imagen", type=["jpg", "png", "jpeg"])

modo_accion = st.sidebar.selectbox("Efecto a aplicar en la zona seleccionada:", ["Aclarar / Iluminar Piel", "Suavizar Imperfecciones", "Aumentar Contraste Local"])
radio_pincel = st.sidebar.slider("Tamaño del Pincel", 5, 50, 15)

if archivo_subido is not None:
    # Cargar imagen
    file_bytes = np.asarray(bytearray(archivo_subido.read()), dtype=np.uint8)
    img_original = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    h, w = img_original.shape[:2]
    
    # Redimensionar para mostrar de forma cómoda en la web si es muy grande
    max_w = 600
    if w > max_w:
        h = int(h * (max_w / w))
        w = max_w
        img_original = cv2.resize(img_original, (w, h))
        
    img_rgb = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)
    
    st.info("💡 **Instrucciones:** Usa el mouse en el cuadro de abajo para **pintar o marcar** encima de la parte que deseas modificar (ej. la piel, los labios o una zona del cuerpo).")

    # Crear el lienzo interactivo web sobre la imagen
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Color del trazo visual
        stroke_width=radio_pincel,
        stroke_color="#FFFFFF",
        background_image=Image.fromarray(img_rgb),
        update_streamlit=True,
        height=h,
        width=w,
        drawing_mode="freedraw",
        key="canvas_retoco",
    )

    img_resultado = img_original.copy()

    # Si el usuario pintó algo en el lienzo, aplicamos el efecto exactamente en esa máscara
    if canvas_result.image_data is not None:
        # Extraer el canal Alfa o la zona dibujada por el usuario como máscara
        canvas_img = canvas_result.image_data.astype(np.uint8)
        # Detectar dónde pintó el usuario (diferencia con la imagen base)
        gris_canvas = cv2.cvtColor(canvas_img, cv2.COLOR_RGBA2GRAY)
        _, mascara = cv2.threshold(gris_canvas, 50, 255, cv2.THRESH_BINARY)
        
        if np.any(mascara > 0):
            # Aplicar el efecto seleccionado únicamente donde hay máscara
            if modo_accion == "Aclarar / Iluminar Piel":
                # Elevar brillo en la zona seleccionada
                zona_editada = cv2.convertScaleAbs(img_original, alpha=1.2, beta=30)
            elif modo_accion == "Suavizar Imperfecciones":
                # Aplicar filtro bilateral suave en la zona
                suave = cv2.bilateralFilter(img_original, 9, 75, 75)
                zona_editada = suave
            else:
                zona_editada = cv2.convertScaleAbs(img_original, alpha=1.4, beta=10)
                
            # Combinar la imagen original con la zona editada usando la máscara del pincel
            mascara_3ch = cv2.cvtColor(mascara, cv2.COLOR_GRAY2BGR) / 255.0
            img_resultado = (img_original * (1 - mascara_3ch) + zona_editada * mascara_3ch).astype(np.uint8)

    # Mostrar el resultado final
    st.subheader("Resultado del Retoque Localizado")
    st.image(cv2.cvtColor(img_resultado, cv2.COLOR_BGR2RGB), use_container_width=True)
    
    # Botón de descarga
    res_pil = Image.fromarray(cv2.cvtColor(img_resultado, cv2.COLOR_BGR2RGB))
    res_pil.save("resultado_local.jpg")
    with open("resultado_local.jpg", "rb") as f:
        st.download_button("📥 Descargar Imagen Editada", f, file_name="editado.jpg", mime="image/jpeg")

else:
    st.info("👈 Sube una imagen en el panel lateral para empezar a pintar y retocar.")
