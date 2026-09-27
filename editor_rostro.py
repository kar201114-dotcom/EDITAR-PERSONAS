import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Configuración de la página web
st.set_page_config(page_title="Estudio de Retoque Físico y Facial", layout="wide")

st.title("🎨 Estudio de Retoque Físico y Facial en la Nube")
st.write("Sube una foto y ajusta la silueta, el rostro, el brillo y los tonos de piel usando las barras del panel lateral.")

# --- Panel Lateral de Controles Web ---
st.sidebar.header("Herramientas de Edición")
archivo_subido = st.sidebar.file_uploader("Sube tu imagen (Rostro/Cuerpo)", type=["jpg", "png", "jpeg"])

st.sidebar.subheader("Deformación Física y Silueta")
# Barra para estilizar / adelgazamiento o volumen global
factor_forma = st.sidebar.slider("Estilizar / Delgadez ↔ Volumen", -25, 25, 0)

st.sidebar.subheader("Ajustes de Color e Iluminación")
contraste = st.sidebar.slider("Contraste", 0.5, 2.0, 1.0, 0.1)
brillo = st.sidebar.slider("Brillo (-100 a 100)", -100, 100, 0)
tono_piel = st.sidebar.slider("Tono de Piel (Matiz)", 0, 180, 0)

if archivo_subido is not None:
    # Leer imagen subida
    file_bytes = np.asarray(bytearray(archivo_subido.read()), dtype=np.uint8)
    img_original = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    h, w = img_original.shape[:2]
    img_trabajo = img_original.copy()
    
    # 1. Aplicación de deformación física inteligente mediante controles web
    if factor_forma != 0:
        src_pts = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        m_val = int(abs(factor_forma) * 1.2)
        
        if factor_forma > 0: 
            # Adelgaza el centro (efecto silueta / rostro más perfilado)
            dst_pts = np.float32([[m_val, 0], [w - m_val, 0], [0, h], [w, h]])
        else: 
            # Da volumen / expande
            dst_pts = np.float32([[-m_val, 0], [w + m_val, 0], [0, h], [w, h]])
            
        matriz = cv2.getPerspectiveTransform(src_pts, dst_pts)
        img_trabajo = cv2.warpPerspective(img_original, matriz, (w, h), borderMode=cv2.BORDER_REPLICATE)

    # 2. Aplicar contraste y brillo
    img_ajustada = cv2.convertScaleAbs(img_trabajo, alpha=contraste, beta=brillo)
    
    # 3. Modificador de tono de piel
    if tono_piel > 0:
        hsv = cv2.cvtColor(img_ajustada, cv2.COLOR_BGR2HSV)
        mask_piel = cv2.inRange(hsv, (0, 20, 70), (25, 250, 255))
        hsv[:, :, 0] = np.where(mask_piel > 0, (hsv[:, :, 0].astype(int) + tono_piel) % 180, hsv[:, :, 0])
        img_ajustada = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
    # Mostrar resultados lado a lado en la web
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Imagen Original")
        st.image(cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB), use_container_width=True)
        
    with col2:
        st.subheader("Imagen Modificada")
        st.image(cv2.cvtColor(img_ajustada, cv2.COLOR_BGR2RGB), use_container_width=True)
        
    # Botón de descarga para guardar la foto editada
    resultado_pil = Image.fromarray(cv2.cvtColor(img_ajustada, cv2.COLOR_BGR2RGB))
    resultado_pil.save("resultado_retocado.jpg")
    
    with open("resultado_retocado.jpg", "rb") as file:
        st.sidebar.download_button(
            label="📥 Descargar Imagen Modificada",
            data=file,
            file_name="imagen_retocada.jpg",
            mime="image/jpeg"
        )
else:
    st.info("👈 Sube una imagen en el panel lateral para comenzar a editar.")
