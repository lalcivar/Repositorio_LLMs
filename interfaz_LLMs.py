import os
import torch
import requests
import customtkinter as ctk
import base64
from tkinter import filedialog, messagebox
from dotenv import load_dotenv
import threading
import time
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import zipfile
import json
from datetime import datetime
import ast
import re
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score
from extractor_ground_truth import procesar_archivo_codigo  # Importante
import warnings
from multiprocessing import Process, Queue

# Cargar la API Key desde el archivo .env
load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY")

# Configurar el aspecto visual en modo oscuro
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

# Diccionario que conecta lo que ve el usuario con los IDs oficiales de OpenRouter

MODELOS_OPENROUTER = {"-- Seleccionar un modelo --":"",
    "Google - Gemini": "google/gemini-2.5-pro",
    "Anthropic - Claude": "anthropic/claude-opus-5",
    "OpenAI - ChatGPT": "openai/o3",
    "Meta - Llama": "meta/muse-spark-1.2:contributor"
}

def _trabajador_bertscore_proceso(referencia, generada, queue):

    """Se ejecuta BERTScore en un proceso aislado en segundo plano."""

    try:
        

        # Configuración de entorno
        HF_TOKEN = os.getenv("HF_TOKEN", "TU_HUGGING_FACE_TOKEN")
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
        os.environ["TRANSFORMERS_VERBOSITY"] = "error"
        warnings.filterwarnings("ignore")
        
        torch.set_num_threads(1)

        # Validar que los textos no estén vacíos
        if not referencia.strip() or not generada.strip():
            queue.put((0.0, 0.0, 0.0, "Texto de referencia o generado vacío."))
            return

        with torch.no_grad():
            # Modelo multilingüe optimizado de alto rendimiento
            P, R, F1 = score(
                [generada], 
                [referencia], 
                model_type="bert-base-multilingual-cased",
                lang="es", 
                verbose=False,
                device="cpu",
                batch_size=1
            )
        
        # Extracción y formateo numérico
        p_val = max(0.0, round(float(P.mean().item()) * 100, 2))
        r_val = max(0.0, round(float(R.mean().item()) * 100, 2))
        f1_val = max(0.0, round(float(F1.mean().item()) * 100, 2))
        
        queue.put((p_val, r_val, f1_val, None))

    except Exception as e:
        queue.put((0.0, 0.0, 0.0, str(e)))

class AplicacionDocumentador(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.ground_truth_extraido = ""

        # Configuración de la ventana principal
        self.title("GENERADOR TECNICO DE DOCUMENTACIÓN CON LLMS")
        self.centrar_ventana(1100, 680)

        # Variables para guardar datos en memoria
        self.contenido_codigo = ""

        # 1. Panel de configuracion - superior
        self.frame_config = ctk.CTkFrame(self)
        self.frame_config.pack(padx=15, pady=(15, 5), fill="x")

        # Título del panel
        self.lbl_config = ctk.CTkLabel(self.frame_config, text="1. PANEL DE CONFIGURACIÓN", font=("Arial", 12, "bold"))
        self.lbl_config.pack(anchor="w", padx=10, pady=(5, 0))

        # Sub-contenedor para controles de configuración
        self.subframe_config = ctk.CTkFrame(self.frame_config, fg_color="transparent")
        self.subframe_config.pack(fill="x", padx=10, pady=5)

        self.lbl_modelo = ctk.CTkLabel(self.subframe_config, text="MODELOS LLMS:")
        self.lbl_modelo.pack(side="left", padx=(0, 5))

        self.combo_modelos = ctk.CTkOptionMenu(
            self.subframe_config, 
            values=list(MODELOS_OPENROUTER.keys()),
            width=280
        )
        self.combo_modelos.pack(side="left", padx=5)

        # Establecer la opción por defecto visible en pantalla
        self.combo_modelos.set("-- Seleccionar un modelo --")

        # Estado de seguridad de la API Key
        self.lbl_sec_status = ctk.CTkLabel(
            self.subframe_config, 
            text="🔒 API Key Cargada (.env)" if API_KEY else "⚠️ Sin API Key",
            text_color="#34d399" if API_KEY else "#f87171"
        )
        self.lbl_sec_status.pack(side="right", padx=10)

        # 2. Panel de archivo y atefacto
        self.frame_archivo = ctk.CTkFrame(self)
        self.frame_archivo.pack(padx=15, pady=5, fill="x")

        self.lbl_panel_archivo = ctk.CTkLabel(self.frame_archivo, text="2. PANEL DE ARCHIVO Y ARTEFACTO", font=("Arial", 12, "bold"))
        self.lbl_panel_archivo.pack(anchor="w", padx=10, pady=(5, 0))

        self.subframe_archivo = ctk.CTkFrame(self.frame_archivo, fg_color="transparent")
        self.subframe_archivo.pack(fill="x", padx=10, pady=5)

        self.btn_cargar = ctk.CTkButton(
            self.subframe_archivo, 
            text="📁 Seleccionar archivo en: (.py / .java)", 
            command=self.cargar_archivo
        )
        self.btn_cargar.pack(side="left", padx=(0, 10))

        self.lbl_archivo_cargado = ctk.CTkLabel(self.subframe_archivo, text="Sin archivo cargado", text_color="#94a3b8")
        self.lbl_archivo_cargado.pack(side="left", padx=5)

        self.combo_artefacto = ctk.CTkOptionMenu(self.subframe_archivo, 
        values=[
        "-- Seleccionar tipo de documento --", 
        "Manual de usuario (Alto nivel)",
         "Docstrings de funciones", 
        "Documentación de APIs"]
)
        # Forzar que el valor por defecto sea la instrucción de selección
        self.combo_artefacto.set("-- Seleccionar tipo de documento --")
        self.combo_artefacto.pack(side="right", padx=5)

        self.lbl_tipo_doc = ctk.CTkLabel(self.subframe_archivo, text="TIPO DE DOCUMENTACIÓN:")
        self.lbl_tipo_doc.pack(side="right", padx=(0, 5))
        
       
        
        # MARCO CONTENEDOR PADRE - Contiene ambas filas de botones
        self.frame_botones_principal = ctk.CTkFrame(self, fg_color="transparent")
        self.frame_botones_principal.pack(padx=15, pady=10, fill="x")

        # BOTONES DE ACCIÓN - GENERAR DOCUMENTACIÓN - AUDITAR ALUCINACIONES Y EVALUAR METRICAS NLP
        self.frame_botones_accion = ctk.CTkFrame(self.frame_botones_principal, fg_color="transparent")
        self.frame_botones_accion.pack(fill="x", pady=(0, 6))

        # Botón - GENERAR DOCUMENTACIÓN
        self.btn_generar = ctk.CTkButton(
            self.frame_botones_accion, 
            text="⚡ GENERAR DOCUMENTACIÓN", 
            font=("Arial", 13, "bold"),
            fg_color="#16a34a", 
            hover_color="#15803d",
            height=40,
            width=120,
            command=self.generar_documentacion
        )
        self.btn_generar.pack(side="left", fill="x", expand=True, padx=(0, 10))

        # Botón para auditar estructura ya sea en Python o Java vs el LLMs
        self.btn_auditar = ctk.CTkButton(
            self.frame_botones_accion, 
            text="🔍 AUDITAR ALUCINACIONES", 
            font=("Arial", 12, "bold"),
            fg_color="#0284c7", 
            hover_color="#0369a1", 
            height=40,
            width=120,
            command=self.auditar_estructura_codigo
        )
        self.btn_auditar.pack(side="left", expand=True, fill="x", padx=3)

        # Botón para evaluación de métricas NLP
        self.btn_evaluar_nlp = ctk.CTkButton(
            self.frame_botones_accion, 
            text="📊 EVALUAR METRICAS NLP", 
            font=("Arial", 12, "bold"),
            fg_color="#0284c7", 
            hover_color="#0369a1", 
            height=40,
            width=120,
            command=self.abrir_ventana_evaluacion_nlp
        )
        self.btn_evaluar_nlp.pack(side="left", expand=True, fill="x", padx=3)

        # BOTONES DE CONTROL - CANCELAR Y SALIR
        self.frame_control = ctk.CTkFrame(self.frame_botones_principal, fg_color="transparent")
        self.frame_control.pack(fill="x", pady=(2, 0))
        
        # Botón CANCELAR - inicialmente deshabilitado
        self.btn_cancelar = ctk.CTkButton(
            self.frame_control, 
            text="❎ CANCELAR", 
            fg_color="#dc2626", 
            hover_color="#991b1b",
            font=("Arial", 13, "bold"),
            height=40,
            width=120,
            command=self.cancelar_proceso,
            state="disabled"
        )
        self.btn_cancelar.pack(side="left", expand=True, fill="x", padx=3)

        # Botón SALIR - ubicado a la derecha del verde
        self.btn_salir = ctk.CTkButton(
            self.frame_control, 
            text="🪟 SALIR", 
            font=("Arial", 13, "bold"),
            fg_color="#dc2626", 
            hover_color="#991b1b", 
            height=40,
            width=120,
            command=self.salir_del_sistema
        )
        self.btn_salir.pack(side="left", expand=True, fill="x", padx=3)
        

        # PANELS DE CÓDIGO Y RESULTADO
        self.frame_editores = ctk.CTkFrame(self, fg_color="transparent")
        self.frame_editores.pack(padx=15, pady=5, fill="both", expand=True)

        # Entrada de Código
        self.frame_izq = ctk.CTkFrame(self.frame_editores)
        self.frame_izq.pack(side="left", fill="both", expand=True, padx=(0, 5))

        self.lbl_izq = ctk.CTkLabel(self.frame_izq, text="3. ENTRADA DE CÓDIGO FUENTE", font=("Arial", 11, "bold"))
        self.lbl_izq.pack(anchor="w", padx=10, pady=5)

        self.txt_codigo = ctk.CTkTextbox(self.frame_izq, font=("Consolas", 12))
        self.txt_codigo.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        # Documentación Generada
        self.frame_der = ctk.CTkFrame(self.frame_editores)
        self.frame_der.pack(side="right", fill="both", expand=True, padx=(5, 0))

        self.lbl_der = ctk.CTkLabel(self.frame_der, text="4. RESULTADO DE DOCUMENTACIÓN GENERADA", font=("Arial", 11, "bold"))
        self.lbl_der.pack(anchor="w", padx=10, pady=5)

        self.txt_resultado = ctk.CTkTextbox(self.frame_der, font=("Arial", 12))
        self.txt_resultado.pack(fill="both", expand=True, padx=10, pady=(0, 10))

       # BANDA INFERIOR DE ESTADO
        self.frame_status = ctk.CTkFrame(self)
        self.frame_status.pack(padx=15, pady=(5, 15), fill="x")

        # Estado, barra y tiempo
        self.frame_status_top = ctk.CTkFrame(self.frame_status, fg_color="transparent")
        self.frame_status_top.pack(fill="x", padx=10, pady=(5, 2))

        # Mensaje de estado
        self.lbl_estado = ctk.CTkLabel(self.frame_status_top, text="Estado: Listo para procesar", text_color="#38bdf8")
        self.lbl_estado.pack(side="left", padx=5)

        # Tiempo transcurrido
        self.lbl_tiempo = ctk.CTkLabel(self.frame_status_top, text="⏱ Tiempo: 0.0s", font=("Arial", 11))
        self.lbl_tiempo.pack(side="right", padx=5)

        # Lado derecho barra de progreso
        self.progress_bar = ctk.CTkProgressBar(self.frame_status_top, width=180)
        self.progress_bar.pack(side="right", padx=15)
        self.progress_bar.set(0)


        # INFERIOR - RECUADRO VERDE DEBAJO: Modelo y botones de exportación
        self.frame_status_bottom = ctk.CTkFrame(self.frame_status, fg_color="transparent")
        self.frame_status_bottom.pack(fill="x", padx=10, pady=(2, 5))

        # Lado izquierdo nombre del modelo activo
        self.lbl_info_modelo = ctk.CTkLabel(self.frame_status_bottom, text="🌐 Modelo:  |  📄 Lenguaje:  ", font=("Arial", 11))
        self.lbl_info_modelo.pack(side="left", padx=5)

        # Lado derecho botón - Paquete de datos (.zip)
        self.btn_export_tesis = ctk.CTkButton(
            self.frame_status_bottom, 
            text="📦 Paquete de datos (.zip)", 
            font=("Arial", 11, "bold"),
            fg_color="#8b5cf6", 
            hover_color="#6d28d9", 
            width=90,
            command=self.exportar_zip_analisis
        )
        self.btn_export_tesis.pack(side="right", padx=4)

        # Lado derecho botón MD
        self.btn_export_md = ctk.CTkButton(
            self.frame_status_bottom, 
            text="📄 MD", 
            width=70, 
            fg_color="#0284c7", 
            command=self.exportar_md
        )
        self.btn_export_md.pack(side="right", padx=4)

        # Lado derecho botón PDF
        self.btn_export_pdf = ctk.CTkButton(
            self.frame_status_bottom, 
            text="📕 PDF", 
            width=70, 
            fg_color="#e11d48", 
            command=self.exportar_pdf
        )
        self.btn_export_pdf.pack(side="right", padx=4)

        # Lado derecho etiqueta Exportar
        self.lbl_exportar = ctk.CTkLabel(self.frame_status_bottom, text="Exportar:", font=("Arial", 11, "bold"))
        self.lbl_exportar.pack(side="right", padx=(10, 2))
  
        # Guardar referencias de ejecución
        self.hilo_generacion = None
        self.proceso_bertscore_ref = None
        self.cancelar_solicitado = False

        
    # --- Seccion de funcionalidades ---
    def cargar_archivo(self):
        """Abre una ventana para elegir un archivo de código y separa el Ground Truth"""
        ruta_archivo = filedialog.askopenfilename(
            filetypes=[ ("Todos los archivos", "*.*"), ("Archivos Java", "*.java"), ("Archivos Python", "*.py")]
        )
        if ruta_archivo:
            nombre_archivo = os.path.basename(ruta_archivo)
            
            try:
                # Extraer código y Ground Truth
                codigo_limpio, ground_truth = procesar_archivo_codigo(ruta_archivo)
                
                # IMPRIMIR EN CONSOLA PARA VERIFICAR QUE SE EXTRAJO
                print("\n--- DIAGNÓSTICO DE EXTRACCIÓN ---")
                print(f"Archivo: {nombre_archivo}")
                print(f"Caracteres de Ground Truth encontrados: {len(ground_truth)}")
                print(f"Contenido Ground Truth:\n{ground_truth}\n---------------------------------\n")

                self.contenido_codigo = codigo_limpio
                self.ground_truth_extraido = ground_truth
                
                # 2. Insertar código en la caja principal
                self.txt_codigo.delete("1.0", "end")
                self.txt_codigo.insert("1.0", self.contenido_codigo)
                
                self.lbl_archivo_cargado.configure(text=f"Cargado: {nombre_archivo}", text_color="#34d399")
                
            except Exception as e:
                print(f"❌ Error durante el procesamiento: {e}")
                with open(ruta_archivo, "r", encoding="utf-8") as f:
                    self.contenido_codigo = f.read()
                self.txt_codigo.delete("1.0", "end")
                self.txt_codigo.insert("1.0", self.contenido_codigo)
                self.ground_truth_extraido = ""

    def generar_documentacion(self):
        try:
           

            # Limpiar interfaz
            self.txt_resultado.configure(state="normal")
            self.txt_resultado.delete("1.0", "end")

            if not API_KEY:
                messagebox.showerror("Error de Configuración", "No se encontró OPENROUTER_API_KEY en .env")
                return

            opcion_elegida = self.combo_modelos.get()
            modelo_id = MODELOS_OPENROUTER.get(opcion_elegida, "")

            if not modelo_id:
                messagebox.showwarning("Modelo no seleccionado", "Por favor, seleccione un modelo de LLM válido y un tipo de documento antes de continuar.")
                return

            tipo_doc = self.combo_artefacto.get().strip()

            if not tipo_doc or tipo_doc in ["-- Seleccionar tipo de documento --", ""]:
                messagebox.showwarning("Tipo de Documentación No Seleccionado", "Por favor, seleccione un tipo de documento válido y un modelo de LLMs antes de continuar.")
                return
            
            codigo = self.txt_codigo.get("1.0", "end-1c").strip()

            if not codigo:
                messagebox.showwarning("Advertencia", "Por favor carga un archivo o escribe código antes de generar.")
                return

            # Detectar lenguaje
            #lenguaje_detectado = detectar_lenguaje_codigo(codigo)
            _, lenguaje_detectado = self._extraer_funciones_codigo(codigo)

            # Restablecer etiquetas relativas a la generación previa
            self.lbl_info_modelo.configure(text=f"🌐 Modelo: {modelo_id}  |  🔤 Lenguaje: {lenguaje_detectado}  |  📋 Documento: {tipo_doc}")
            self.lbl_tiempo.configure(text="⏱ Tiempo: 0.0s")
            self.ultimas_metricas_nlp = None

            # Configurar estado activo para generación
            self.cancelar_solicitado = False
            self.btn_generar.configure(state="disabled", text="⏳ GENERANDO DOCUMENTACIÓN...")
            self.btn_cancelar.configure(state="normal")
            self.btn_salir.configure(state="disabled")
            self.lbl_estado.configure(text="Estado: ⏳ Procesando con OpenRouter...", text_color="#facc15")

            # Configurar la barra de progreso
            self.progress_bar.set(0)
            self.progreso_valor = 0.0
            self.progreso_direccion = 0.05

            # Iniciar temporizador y animación
            self.tiempo_inicio = time.time()
            self.generando = True
            self._animar_interfaz()

            # Lanza el hilo sin bloque finally posterior
            self.hilo_generacion = threading.Thread(
                target=self._solicitar_api_thread, 
                args=(codigo,),
                daemon=True
            )
            self.hilo_generacion.start()

        except Exception as e:
            messagebox.showerror("Error", f"Ocurrió un error inesperado: {str(e)}")
            # En caso de excepción local previa al hilo, re-restablecer estado
            self._restablecer_botones_ui()
            
    def _animar_interfaz(self):
        """Mueve la barra de progreso y actualiza el reloj en tiempo real"""
        if getattr(self, 'generando', False):
            # Mover la barra de izquierda a derecha (rebote)
            self.progreso_valor += self.progreso_direccion
            if self.progreso_valor >= 1.0 or self.progreso_valor <= 0.0:
                self.progreso_direccion *= -1 # Invertir dirección
            
            self.progress_bar.set(self.progreso_valor)

            # Actualizar reloj transcurrido
            tiempo_transcurrido = round(time.time() - self.tiempo_inicio, 1)
            self.lbl_tiempo.configure(text=f"⏱ Tiempo: {tiempo_transcurrido}s")

            # Ejecutar este método cada 80 milisegundos
            self.after(80, self._animar_interfaz)

    def _solicitar_api_thread(self, codigo):
        """Ejecuta la petición HTTP en segundo plano con puntos de cancelación."""
        
        # Si se canceló antes de empezar la red
        if self.cancelar_solicitado:
            return

        modelo_nombre = self.combo_modelos.get()
        modelo_id = MODELOS_OPENROUTER[modelo_nombre]
        tipo_artefacto = self.combo_artefacto.get()

        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": modelo_id,
            "messages": [
                {"role": "system", 
                "content": f"""Actúa como un arquitecto de software y redactor técnico senior.
                Tu tarea es analizar código fuente y generar un/a '{tipo_artefacto}' exhaustivo y preciso en lenguaje natural.

                Estructura Requerida:
                1. Resumen ejecutivo: Propósito general del código en 2 o 3 oraciones.
                2. Especificación de funciones: Tablas con parámetros de entrada (nombre, tipo, requerido), valores de retorno 
                y manejo de errores. 
                - IMPORTANTE: Debes incluir y documentar TODAS las funciones y métodos detectados en el código, sin excepción.
                  Esto incluye métodos 
                principales, métodos de clase, 
                métodos auxiliares y funciones anidadas (locales/internas dentro de otros métodos, callbacks, workers de hilos, etc.).
                3. Ejemplo de uso: Bloque de código con un caso real.

                Restricciones:
                - Documenta al 100% todas las funciones presentes en el código (principales, auxiliares y anidadas/locales).
                - Mantén explícitamente el nombre exacto de cada función/método en el texto o encabezados (ejemplo: `mi_funcion()`).
                - No inventes ni asumas métodos externos que no pertenezcan al código fuente provisto.
                - Usa formato Markdown limpio.
                - Mantén un tono profesional y directo."""},
                { "role": "user", "content": f"Documenta el siguiente código:\n\n```\n{codigo}\n```"}
            ],
            "temperature": 0.2
        }

        try:
            response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
            
            # Si se canceló mientras OpenRouter respondía
            if self.cancelar_solicitado:
                return

            tiempo_final = round(time.time() - self.tiempo_inicio, 1)

            if response.status_code == 200:
                resultado = response.json()['choices'][0]['message']['content']
                
                # Actualizar pantalla usando .after para garantizar estabilidad del hilo
                self.after(0, lambda: self._actualizar_exito(resultado, tiempo_final))
            else:
                msg_err = f"Error {response.status_code}: {response.text}"
                self.after(0, lambda: self._actualizar_error(msg_err))

        except Exception as e:
            # Si el usuario canceló, ignorar cualquier error provocado por el cierre abrupto
            if not self.cancelar_solicitado:
                self.after(0, lambda: self._actualizar_error(str(e)))


    def _actualizar_exito(self, resultado, tiempo_total):
        """Detiene la animación y vuelca los datos finales"""
        self.generando = False

        # Habilitar temporalmente la edición para escribir los datos
        self.txt_resultado.configure(state="normal")
        
        # Asegurar que el resultado sea una cadena de texto válida
        texto_resultado = str(resultado) if resultado is not None else "No se obtuvo respuesta del modelo."

        # Llenar la caja de resultado
        self.txt_resultado.delete("1.0", "end")
        self.txt_resultado.insert("1.0", texto_resultado)

        # Bloquear la caja de texto para que sea de solo de lectura
        self.txt_resultado.configure(state="disabled")
        
        # Llenar métricas inferiores
        self.lbl_estado.configure(text="Estado: ✅ Generación completada", text_color="#34d399")
        self.lbl_tiempo.configure(text=f"⏱ Tiempo: {tiempo_total}s")
        #self.lbl_info_modelo.configure(text=f"🌐 Modelo: {modelo} | 📄 Lenguaje: {lenguaje_detectado}")
        
        # Barra de progreso al 100% y botón normal
        self.progress_bar.set(1.0)
        self.btn_generar.configure(state="normal", text="⚡ GENERAR DOCUMENTACIÓN")

        # Restablecer botones y combos
        self._restablecer_botones_ui()

    def _actualizar_error(self, mensaje_error):

        """Detiene la animación y muestra el mensaje de falla"""
        
        self.generando = False
        self.progress_bar.set(0)
        self.lbl_estado.configure(text="Estado: ❌ Error en la generación", text_color="#f87171")
        self.btn_generar.configure(state="normal", text="⚡ GENERAR DOCUMENTACIÓN")
        messagebox.showerror("Error API", mensaje_error)

        # Restablecer botones y combos
        self._restablecer_botones_ui()

    def exportar_md(self):
        """Guarda la documentación generada en un archivo Markdown (.md)"""
        contenido = self.txt_resultado.get("1.0", "end-1c").strip()
        
        if not contenido:
            messagebox.showwarning("Advertencia", "No hay documentación generada para exportar.")
            return

        # Abre la ventana de diálogo para elegir dónde guardar
        filepath = filedialog.asksaveasfilename(
            defaultextension=".md", 
            filetypes=[("Archivos Markdown", "*.md"), ("Todos los archivos", "*.*")]
        )
        
        if filepath:
            try:
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(contenido)
                messagebox.showinfo("Éxito", "¡Archivo .md guardado correctamente!")
            except Exception as e:
                messagebox.showerror("Error", f"No se pudo guardar el archivo: {str(e)}")

    def exportar_pdf(self):
        """Genera un archivo PDF válido con el contenido de la documentación"""
        contenido = self.txt_resultado.get("1.0", "end-1c").strip()
        
        if not contenido:
            messagebox.showwarning("Advertencia", "No hay documentación generada para exportar.")
            return

        filepath = filedialog.asksaveasfilename(
            defaultextension=".pdf", 
            filetypes=[("Archivos PDF", "*.pdf"), ("Todos los archivos", "*.*")]
        )
        
        if filepath:
            try:
                # Crear el lienzo del PDF
                pdf = canvas.Canvas(filepath, pagesize=letter)
                pdf.setTitle("Documentación Generada")
                
                # Configurar márgenes y coordenadas
                ancho, alto = letter
                x = 50
                y = alto - 50
                line_height = 14

                # Título del PDF
                pdf.setFont("Helvetica-Bold", 14)
                pdf.drawString(x, y, "DOCUMENTACIÓN DE CÓDIGO FUENTE")
                y -= 30

                # Formato del cuerpo de texto
                pdf.setFont("Helvetica", 10)
                
                # Dividir el contenido por líneas y escribir en el documento
                for linea in contenido.split('\n'):
                    # Si el texto llega al final de la página, crear una nueva
                    if y < 50:
                        pdf.showPage()
                        pdf.setFont("Helvetica", 10)
                        y = alto - 50
                    
                    pdf.drawString(x, y, linea)
                    y -= line_height

                # Guardar el PDF real
                pdf.save()
                messagebox.showinfo("Éxito", "¡Archivo PDF generado y guardado con éxito!")

            except Exception as e:
                messagebox.showerror("Error", f"No se pudo crear el PDF: {str(e)}")

    def exportar_zip_analisis(self):
        """Exporta código, documentación y métricas en un ZIP comprimido y sincroniza con GitHub"""

        codigo = self.txt_codigo.get("1.0", "end-1c").strip()
        documentacion = self.txt_resultado.get("1.0", "end-1c").strip()

        if not codigo or not documentacion:
            messagebox.showwarning("Advertencia", "Debe haber código de entrada y documentación generada para exportar.")
            return

        timestamp_default = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # =========================================================================
        # DETERMINAR EL NOMBRE DEL ARCHIVO .ZIP
        # =========================================================================
        # Intenta obtener la ruta del archivo cargado si la guardaste previamente en un atributo
        archivo_cargado = getattr(self, "archivo_cargado_path", None)
        
        if archivo_cargado and os.path.exists(archivo_cargado):
            # Extrae solo el nombre sin la extensión 
            nombre_base = os.path.splitext(os.path.basename(archivo_cargado))[0]
            default_zip_name = f"Empaquetado_{nombre_base}.zip"
        else:
            # Nombre por defecto actual si el código fue pegado manualmente o no hay archivo cargado
            default_zip_name = f"Empaquetado_De_Datos_{timestamp_default}.zip"

        filepath = filedialog.asksaveasfilename(
            initialfile=default_zip_name,
            defaultextension=".zip",
            filetypes=[("Archivo ZIP", "*.zip"), ("Todos los archivos", "*.*")]
        )

        if not filepath:
            return

        try:
            modelo_seleccionado = self.combo_modelos.get()
            tipo_artefacto = self.combo_artefacto.get()
            tiempo_texto = self.lbl_tiempo.cget("text").replace("⏱ Tiempo: ", "").replace("s", "").strip()

            # Obtener las métricas NLP si fueron calculadas previamente
            metricas_nlp = getattr(self, "ultimas_metricas_nlp", None)

            metricas_data = {
                "experimento": {
                    "fecha_hora": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "modelo_llm": modelo_seleccionado,
                    "modelo_id": MODELOS_OPENROUTER.get(modelo_seleccionado, "Desconocido"),
                    "tipo_artefacto_generado": tipo_artefacto
                },
                "metricas_rendimiento": {
                    "Tiempo_respuesta_segundos": float(tiempo_texto) if tiempo_texto.replace('.', '', 1).isdigit() else 0.0,
                    "Caracteres_codigo_entrada": len(codigo),
                    "Lineas_codigo_entrada": len(codigo.splitlines()),
                    "Caracteres_documentacion_salida": len(documentacion),
                    "Estimacion_aproximada_tokens": round((len(codigo) + len(documentacion)) / 4)
                },
                "metricas_nlp_evaluacion": metricas_nlp if metricas_nlp else "No calculadas antes de exportar"
            }

            # Detectar lenguaje para asignar la extensión correcta - .java, .py
            _, lenguaje = self._extraer_funciones_codigo(codigo)
            
            ext = ".java" if lenguaje.lower() == "java" else ".py"
            nombre_archivo_codigo = f"01_codigo_fuente{ext}"

            # Empaquetado Local
            with zipfile.ZipFile(filepath, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                # Archivos principales
                zip_file.writestr(nombre_archivo_codigo, codigo)
                zip_file.writestr("02_documentacion_generada.md", documentacion)
                zip_file.writestr("03_metricas_experimento.json", json.dumps(metricas_data, indent=4, ensure_ascii=False))

                # Reporte legible TXT de métricas NLP en caso de que existan
                if metricas_nlp:
                    txt_nlp = (
                        "==========================================================\n"
                        "        REPORTE DE EVALUACIÓN MÉTRICAS NLP\n"
                        "==========================================================\n\n"
                        f"1. BLEU Score: {metricas_nlp['bleu']}%\n\n"
                        "2. ROUGE Scores (F1):\n"
                        f"   • ROUGE-1: {metricas_nlp['rouge']['rouge_1_f1']}%\n"
                        f"   • ROUGE-2: {metricas_nlp['rouge']['rouge_2_f1']}%\n"
                        f"   • ROUGE-L: {metricas_nlp['rouge']['rouge_l_f1']}%\n\n"
                        "3. BERTScore:\n"
                        f"   • Precisión (P): {metricas_nlp['bertscore']['precision']}%\n"
                        f"   • Recall (R)   : {metricas_nlp['bertscore']['recall']}%\n"
                        f"   • F1-Score     : {metricas_nlp['bertscore']['f1_score']}%\n\n"
                        "----------------------------------------------------------\n"
                        "GROUND TRUTH DE REFERENCIA UTILIZADO:\n"
                        "----------------------------------------------------------\n"
                        f"{metricas_nlp['ground_truth_referencia']}\n"
                    )
                    zip_file.writestr("04_reporte_metricas_nlp.txt", txt_nlp)

            messagebox.showinfo("Éxito Local", "¡Paquete de documentación técnica generada exportado en .zip con éxito!")

            # Sincronización automática con GitHub
            exito_gh, msj_gh = self.subir_zip_a_github(filepath)
            if exito_gh:
                messagebox.showinfo("Sincronización GitHub", msj_gh)
            else:
                messagebox.showwarning("Advertencia GitHub", msj_gh)

        except Exception as e:
            messagebox.showerror("Error de Exportación", f"No se pudo empaquetar el archivo ZIP: {str(e)}")

    def salir_del_sistema(self):

        """Muestra un mensaje de confirmación y cierra la aplicación"""
        respuesta = messagebox.askyesno("Confirmar salida", "¿Estás seguro de que deseas salir del sistema?")
        if respuesta:
            self.destroy() # Destruye la ventana principal y cierra Python de forma limpia

    def centrar_ventana(self, ventana_target=None, ancho=1100, alto=680):

        """Calcula el centro de la pantalla y ubica la ventana allí.
        Si no se le pasa => ventana_target, centrará la ventana principal => self."""
        
        # Si no se envía una ventana específica, se asume que es la principal => self
        if ventana_target is None or isinstance(ventana_target, int):
            # Por si acaso se llamó de la forma antigua: self.centrar_ventana(1100, 680)
            if isinstance(ventana_target, int):
                alto = ancho
                ancho = ventana_target
            win = self
        else:
            win = ventana_target

        win.update_idletasks()

        # Dimensiones de la pantalla
        pantalla_ancho = win.winfo_screenwidth()
        pantalla_alto = win.winfo_screenheight()

        # Posición centrada X e Y
        x = (pantalla_ancho // 2) - (ancho // 2)
        y = (pantalla_alto // 2) - (alto // 2)

        # Aplicar geometría a la ventana correspondiente
        win.geometry(f"{ancho}x{alto}+{x}+{y}")

    def _extraer_funciones_codigo(self, codigo, ruta_archivo=None):
        
        """Combina extensión de archivo + AST Python + Regex Java estricto.
        Funciona tanto al cargar archivos como al copiar y pegar."""
        
        if not codigo or not codigo.strip():
            return set(), "Desconocido/General"

        lenguaje_detectado = None

        # Extensión del archivo, si proviene de caargar un archivo
        # Si la interfaz guarda la ruta o si se la pasa explícitamente
        if ruta_archivo:
            ext = os.path.splitext(ruta_archivo)[1].lower()
            if ext == ".py":
                lenguaje_detectado = "Python"
            elif ext in [".java", ".jav"]:
                lenguaje_detectado = "Java"

        # Análisis por contenido, en caso de que el código es copiado y pegado
        if not lenguaje_detectado:
            # Intentar AST de Python primero que es el estándar más fiable
            try:
                ast.parse(codigo)
                lenguaje_detectado = "Python"
            except SyntaxError:
                pass

            # Si falla AST, se revisa la sintaxis nativa de Python como imports relativos, def, class, etc...
            if not lenguaje_detectado:
                patron_python = r'^\s*(from|import)\s+[\.\w]+|\b(def|class|self|elif)\b'
                if re.search(patron_python, codigo, re.MULTILINE):
                    lenguaje_detectado = "Python"

            # Si no coincide con Python, se evalua la sintaxis estricta de Java
            if not lenguaje_detectado:
                patron_java_estricto = r'\b(public\s+class|private\s+|protected\s+|public\s+static\s+void|package\s+[\w\.]+;)\b'
                if re.search(patron_java_estricto, codigo):
                    lenguaje_detectado = "Java"
                else:
                    lenguaje_detectado = "Desconocido/General"

        # Extracción de funciones según el lenguaje
        funciones = set()

        if lenguaje_detectado == "Python":
            try:
                tree = ast.parse(codigo)
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        funciones.add(node.name)
            except SyntaxError:
                pass

        elif lenguaje_detectado == "Java":
            patron_java = r'(?:public|protected|private|static|final|synchronized|\s)+[\w\<\>\[\]\?]+\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^\)]*\)\s*(?:throws\s+[\w\s,]+)?\s*\{'
            coincidencias = re.findall(patron_java, codigo)
            reservadas = {"if", "for", "while", "switch", "catch", "main", "try", "else", "finally", "do", "return", "new", "throw"}
            funciones = {m for m in coincidencias if m not in reservadas}

        return funciones, lenguaje_detectado

    def auditar_estructura_codigo(self):

        """Compara las funciones del código fuente con la documentación generada"""

        codigo = self.txt_codigo.get("1.0", "end-1c").strip()
        documentacion = self.txt_resultado.get("1.0", "end-1c").strip()

        if not codigo or not documentacion:
            messagebox.showwarning("Advertencia", "Se requiere código de entrada y documentación generada para ejecutar la auditoría.")
            return

        # Extraer funciones reales del código
        funciones_reales, lenguaje = self._extraer_funciones_codigo(codigo)

        if not funciones_reales:
            messagebox.showinfo("Auditoría", "No se detectaron funciones o métodos explícitos en el código cargado.")
            return

        # Detectar cuáles funciones fueron documentadas y cuáles no
        funciones_documentadas = set()
        funciones_olvidadas = set()

        for fn in funciones_reales:
            # Buscar coincidencia de la función dentro del Markdown
            if re.search(rf'\b{re.escape(fn)}\b', documentacion):
                funciones_documentadas.add(fn)
            else:
                funciones_olvidadas.add(fn)

        # Detectar posibles alucinaciones - menciones a funciones inexistentes
        posibles_funciones_doc = set(re.findall(r'`([a-zA-Z_][a-zA-Z0-9_]*)\(\)`', documentacion))
        posibles_alucinaciones = posibles_funciones_doc - funciones_reales

        # Mostrar reporte en ventana modal
        self._mostrar_reporte_auditoria(lenguaje, funciones_reales, funciones_documentadas, funciones_olvidadas, posibles_alucinaciones)

    def _mostrar_reporte_auditoria(self, lenguaje, totales, documentadas, olvidadas, alucinaciones):
        """Abre una ventana emergente estructurada con los resultados del análisis"""
        ventana_auditoria = ctk.CTkToplevel(self)
        ventana_auditoria.title(f"📊 Reporte de auditoría del código en: {lenguaje}")

       # Centrar ventana
        self.centrar_ventana(ventana_auditoria, 600, 550)
        ventana_auditoria.grab_set()

        txt_reporte = ctk.CTkTextbox(ventana_auditoria, font=("Consolas", 12))
        txt_reporte.pack(fill="both", expand=True, padx=15, pady=15)

        reporte = []
        reporte.append("=============================================================")
        reporte.append(f" AUDITORÍA DE ESTRUCTURA Y ALUCINACIONES EN LENGUAJE: {lenguaje.upper()}")
        reporte.append("=============================================================\n")
        
        reporte.append(f"• Funciones detectadas en el código ({len(totales)}):")
        reporte.append(f"  {', '.join(sorted(totales)) if totales else 'Ninguna'}\n")

        reporte.append(f"✅ Funciones documentadas ({len(documentadas)}/{len(totales)}):")
        reporte.append(f"  {', '.join(sorted(documentadas)) if documentadas else 'Ninguna'}\n")

        reporte.append(f"⚠️ Funciones olvidadas / omitidas por el LLMs ({len(olvidadas)}):")
        reporte.append(f"  {', '.join(sorted(olvidadas)) if olvidadas else 'Ninguna (Cobertura 100%)'}\n")

        reporte.append(f"🚨 Posibles alucinaciones detectadas ({len(alucinaciones)}):")
        reporte.append("  (Nombres referenciados en texto como `fn()` que no existen en el código)")
        reporte.append(f"  {', '.join(sorted(alucinaciones)) if alucinaciones else 'Ninguna alucinación detectada'}\n")
        
        reporte.append("--------------------------------------------------")
        cobertura = (len(documentadas) / len(totales) * 100) if totales else 0
        reporte.append(f"📊 Tasa de cobertura estructural: {cobertura:.1f}%")
        reporte.append("--------------------------------------------------")

        txt_reporte.insert("1.0", "\n".join(reporte))
        txt_reporte.configure(state="disabled") # Solo lectura

        # =========================================================================
        # Boton volver a la pantalla principal
        # =========================================================================
        btn_volver_auditoria = ctk.CTkButton(
            ventana_auditoria, 
            text="🔙 VOLVER", 
            fg_color="#dc2626", 
            hover_color="#991b1b",
            command=ventana_auditoria.destroy
        )
        btn_volver_auditoria.pack(padx=15, pady=(0, 15))
 
    def abrir_ventana_evaluacion_nlp(self):
        """Abre una ventana modal para ingresar la documentación esperada y calcular métricas"""
        doc_generada = self.txt_resultado.get("1.0", "end-1c").strip()

        if not doc_generada:
            messagebox.showwarning("Advertencia", "Primero debes generar la documentación con el LLM para poder evaluarla.")
            return

        # Ventana modal de entrada y resultados
        win_eval = ctk.CTkToplevel(self)
        win_eval.title("📊 Evaluación de Métricas NLP => BLEU, ROUGE, BERTScore")

        # Centrar ventana
        self.centrar_ventana(win_eval, 700, 680)
        win_eval.grab_set()

        lbl_instrucciones = ctk.CTkLabel(
            win_eval, 
            text="Documentación esperada => Ground truth o Referencia humana:",
            font=("Arial", 11, "bold")
        )
        lbl_instrucciones.pack(anchor="w", padx=15, pady=(15, 5))

        # Caja de texto para la documentación de referencia
        txt_referencia = ctk.CTkTextbox(win_eval, height=160, font=("Consolas", 11))
        txt_referencia.pack(fill="x", padx=15, pady=5)

        gt_extraido = self._obtener_ground_truth_actual()
        if gt_extraido:
            txt_referencia.insert("1.0", gt_extraido)

        # Frame para los botones de acción
        frame_acciones = ctk.CTkFrame(win_eval, fg_color="transparent")
        frame_acciones.pack(padx=15, pady=10, fill="x")

        # Área de visualización del reporte de salida
        txt_reporte_nlp = ctk.CTkTextbox(win_eval, font=("Consolas", 12))
        txt_reporte_nlp.pack(fill="both", expand=True, padx=15, pady=(5, 5))

        # =========================================================================
        # BARRA DE ESTADO Y TIEMPO INDEPENDIENTE DE LA VENTANA MODAL
        # =========================================================================
        frame_status_modal = ctk.CTkFrame(win_eval, fg_color="transparent", height=30)
        frame_status_modal.pack(fill="x", padx=15, pady=(0, 10))

        lbl_estado_metricas = ctk.CTkLabel(
            frame_status_modal, 
            text="Estado: Listo", 
            font=("Arial", 11, "bold")
        )
        lbl_estado_metricas.pack(side="left")

        lbl_tiempo_metricas = ctk.CTkLabel(
            frame_status_modal, 
            text="⏱ Tiempo: 0.0s", 
            font=("Arial", 11)
        )
        lbl_tiempo_metricas.pack(side="right")

        # Variables de control locales para el cálculo de métricas
        calculando_metricas = False
        tiempo_inicio_metricas = 0.0

        # =========================================================================
        # LÓGICA Y FUNCIONES INTERNAS (USAN SUS PROPIAS VARIABLES DE TIEMPO)
        # =========================================================================

        def _actualizar_cronometro_modal():
            """Actualiza el tiempo transcurrido en el label local de la ventana modal."""
            if calculando_metricas and win_eval.winfo_exists():
                transcurrido = round(time.time() - tiempo_inicio_metricas, 1)
                lbl_tiempo_metricas.configure(text=f"⏱ Tiempo: {transcurrido:.1f}s")
                win_eval.after(100, _actualizar_cronometro_modal)

        def ejecutar_calculo_metricas():
            """Realiza los cálculos pesados en segundo plano."""
            doc_ref = txt_referencia.get("1.0", "end-1c").strip()
            if not doc_ref:
                def mostrar_warn():
                    messagebox.showwarning("Atención", "Por favor ingresa la documentación esperada de referencia.")
                    _finalizar_ui_metricas()
                win_eval.after(0, mostrar_warn)
                return

            def iniciar_texto_ui():
                txt_reporte_nlp.configure(state="normal")
                txt_reporte_nlp.delete("1.0", "end")
                txt_reporte_nlp.insert("1.0", "⏳Calculando métricas NLP => BLEU, ROUGE, BERTScore. Esto puede tomar unos segundos, espere por favor...\n")
                lbl_estado_metricas.configure(text="Estado: ⏳ Calculando...", text_color="#3b82f6")
            win_eval.after(0, iniciar_texto_ui)

            try:
                # BLEU
                bleu = self._calcular_bleu(doc_ref, doc_generada)

                # ROUGE
                rouge_1, rouge_2, rouge_l = self._calcular_rouge(doc_ref, doc_generada)

                # BERTScore
                bert_p, bert_r, bert_f1 = self._calcular_bertscore(doc_ref, doc_generada)

                # Guardar métricas
                self.ultimas_metricas_nlp = {
                    "ground_truth_referencia": doc_ref,
                    "bleu": round(bleu, 2),
                    "rouge": {
                        "rouge_1_f1": round(rouge_1, 2),
                        "rouge_2_f1": round(rouge_2, 2),
                        "rouge_l_f1": round(rouge_l, 2)
                    },
                    "bertscore": {
                        "precision": round(bert_p, 2),
                        "recall": round(bert_r, 2),
                        "f1_score": round(bert_f1, 2)
                    }
                }

                reporte = [
                    "==========================================================",
                    "        REPORTE DE EVALUACIÓN MÉTRICAS NLP",
                    "==========================================================\n",
                    "1. BLEU Score (Precisión Léxica N-Gramas):",
                    f"   • BLEU Score: {bleu:.2f}%\n",
                    "2. ROUGE Scores (Recall / Cobertura de Texto):",
                    f"   • ROUGE-1 (Unigramas)   - F1-Score: {rouge_1:.2f}%",
                    f"   • ROUGE-2 (Bigramas)    - F1-Score: {rouge_2:.2f}%",
                    f"   • ROUGE-L (Sec. Larga) - F1-Score: {rouge_l:.2f}%\n",
                    "3. BERTScore (Similitud Semántica por Embeddings Contextuales):",
                    f"   • Precisión (P) : {bert_p:.2f}%",
                    f"   • Recall (R)    : {bert_r:.2f}%",
                    f"   • F1-Score      : {bert_f1:.2f}%\n",
                    "----------------------------------------------------------",
                    "💡 Nota: Un BERTScore F1 cercano al 100% indica",
                    "   que la IA capturó el mismo significado semántico, aunque",
                    "   usara palabras diferentes a la referencia humana.",
                    "----------------------------------------------------------"
                ]

                def finalizar_exito():
                    if win_eval.winfo_exists():
                        txt_reporte_nlp.delete("1.0", "end")
                        txt_reporte_nlp.insert("1.0", "\n".join(reporte))
                        txt_reporte_nlp.configure(state="disabled")
                        lbl_estado_metricas.configure(text="Estado: ✅ Completado", text_color="#10b981")

                win_eval.after(0, finalizar_exito)

            except Exception as e:
                def finalizar_error(err_msg):
                    if win_eval.winfo_exists():
                        txt_reporte_nlp.delete("1.0", "end")
                        txt_reporte_nlp.insert("1.0", f"❌ Error al calcular métricas:\n{err_msg}")
                        lbl_estado_metricas.configure(text="Estado: ❌ Error", text_color="#ef4444")
                        messagebox.showerror("Error en Métricas NLP", f"Ocurrió un detalle técnico:\n{err_msg}")

                win_eval.after(0, lambda: finalizar_error(str(e)))

        def _finalizar_ui_metricas():
            """Restaura los botones y fija el tiempo final."""
            nonlocal calculando_metricas
            calculando_metricas = False

            if win_eval.winfo_exists():
                btn_procesar.configure(state="normal", text="⚡ CALCULAR MÉTRICAS")
                btn_cancelar_metricas.configure(state="disabled")
                btn_volver.configure(state="normal")
                
                tiempo_total = round(time.time() - tiempo_inicio_metricas, 1)
                if getattr(self, 'cancelar_solicitado', False):
                    lbl_tiempo_metricas.configure(text=f"⏱ Cancelado en {tiempo_total:.1f}s")
                    lbl_estado_metricas.configure(text="Estado: 🚫 Cancelado", text_color="#ef4444")
                else:
                    lbl_tiempo_metricas.configure(text=f"⏱ Tiempo: {tiempo_total:.1f}s")

        def _worker_calcular_metricas():
            """Worker invocado por el hilo secundario."""
            try:
                ejecutar_calculo_metricas()
            except Exception as e:
                print(f"⚠️ Error durante el cálculo: {e}")
            finally:
                if win_eval.winfo_exists():
                    win_eval.after(0, _finalizar_ui_metricas)

        def iniciar_evaluacion_hilo():
            """Inicia la tarea asíncrona y la animación del tiempo."""
            nonlocal calculando_metricas, tiempo_inicio_metricas
            self.cancelar_solicitado = False
            calculando_metricas = True
            tiempo_inicio_metricas = time.time()

            btn_procesar.configure(state="disabled", text="⏳ CALCULANDO...")
            btn_cancelar_metricas.configure(state="normal")
            btn_volver.configure(state="disabled")

            # Iniciar cronómetro de la ventana modal
            _actualizar_cronometro_modal()

            threading.Thread(target=_worker_calcular_metricas, daemon=True).start()

        def cancelar_evaluacion_metricas():
            """Cancela el proceso activo de métricas."""
            self.cancelar_solicitado = True
            if hasattr(self, 'proceso_bertscore_ref') and self.proceso_bertscore_ref and self.proceso_bertscore_ref.is_alive():
                self.proceso_bertscore_ref.terminate()

        # =========================================================================
        # BOTONES DE ACCIÓN
        # =========================================================================

        btn_procesar = ctk.CTkButton(
            frame_acciones, 
            text="⚡ CALCULAR MÉTRICAS", 
            fg_color="#0284c7", 
            hover_color="#0369a1",
            command=iniciar_evaluacion_hilo
        )
        btn_procesar.pack(side="left", padx=5, expand=True, fill="x")

        btn_cancelar_metricas = ctk.CTkButton(
            frame_acciones, 
            text="❎ CANCELAR", 
            fg_color="#dc2626", 
            hover_color="#991b1b",
            state="disabled",
            command=cancelar_evaluacion_metricas
        )
        btn_cancelar_metricas.pack(side="left", padx=5, expand=True, fill="x")

        btn_volver = ctk.CTkButton(
            frame_acciones, 
            text="🔙 VOLVER", 
            fg_color="#dc2626", 
            hover_color="#991b1b",
            command=win_eval.destroy
        )
        btn_volver.pack(side="left", padx=5)
            
            
    def _calcular_bleu(self, referencia, generada):
            
            """Calcula el BLEU Score con suavizado de frecuencia"""

            ref_tokens = [referencia.lower().split()]
            gen_tokens = generada.lower().split()
            smooth = SmoothingFunction().method1
            return sentence_bleu(ref_tokens, gen_tokens, smoothing_function=smooth) * 100

    def _calcular_rouge(self, referencia, generada):
            
            """Calcula ROUGE-1, ROUGE-2 y ROUGE-L"""

            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            scores = scorer.score(referencia, generada)
            return (
                scores['rouge1'].fmeasure * 100,
                scores['rouge2'].fmeasure * 100,
                scores['rougeL'].fmeasure * 100
            )
        

    def _calcular_bertscore(self, referencia, generada):

        """Ejecuta BERTScore de manera segura evitando bloqueos de memoria en Windows."""

        try:
            queue = Queue()
            
            self.proceso_bertscore_ref = Process(
                target=_trabajador_bertscore_proceso, 
                args=(referencia, generada, queue)
            )
            self.proceso_bertscore_ref.start()

            # Esperar hasta 300 segundos o 5 min, para la primera descarga del modelo
            self.proceso_bertscore_ref.join(timeout=300) 
            
            if self.proceso_bertscore_ref.is_alive():
                print("⚠️ Tiempo de espera agotado para BERTScore. Finalizando subproceso...")
                self.proceso_bertscore_ref.terminate()
                self.proceso_bertscore_ref.join()
                return 0.0, 0.0, 0.0

            if not queue.empty():
                p_val, r_val, f1_val, error = queue.get()
                if error:
                    print(f"❌ Error interno dentro de BERTScore: {error}")
                    return 0.0, 0.0, 0.0
                return p_val, r_val, f1_val
            
            print("⚠️ La cola de resultados quedó vacía inesperadamente.")
            return 0.0, 0.0, 0.0

        except Exception as e:
            print(f"⚠️ Error general en gestión de BERTScore: {e}")
            return 0.0, 0.0, 0.0


    def _obtener_ground_truth_actual(self):

        """Extrae el Ground Truth directamente del texto visible en la caja txt_codigo"""

        codigo_pantalla = self.txt_codigo.get("1.0", "end-1c").strip()
        if not codigo_pantalla:
            return ""

        ground_truth_list = []

        # Intentar extracción por AST - Python
        try:
            tree = ast.parse(codigo_pantalla)
            doc_modulo = ast.get_docstring(tree)
            if doc_modulo:
                ground_truth_list.append(doc_modulo)

            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    doc = ast.get_docstring(node)
                    if doc:
                        ground_truth_list.append(f"[{node.name}]: {doc}")
        except Exception:
            pass

        # Intentar extracción por Expresiones Regulares - Javadoc => Java
        if not ground_truth_list:
            patron_javadoc = r'/\*\*([\s\S]*?)\*/'
            coincidencias = re.findall(patron_javadoc, codigo_pantalla)
            for c in coincidencias:
                texto_limpio = re.sub(r'^\s*\* ?', '', c, flags=re.MULTILINE).strip()
                if texto_limpio:
                    ground_truth_list.append(texto_limpio)

        # Respaldo para comentarios simples
        if not ground_truth_list:
            lineas_comentario = [
                l.strip().lstrip("#").lstrip("//").strip()
                for l in codigo_pantalla.splitlines()
                if l.strip().startswith("#") or l.strip().startswith("//")
            ]
            if lineas_comentario:
                ground_truth_list.append("\n".join(lineas_comentario))

        return "\n\n".join(ground_truth_list)

    def subir_zip_a_github(self, filepath_local):

        """Sube automáticamente el paquete ZIP a una carpeta dentro de GitHub"""

        GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "TU_GITHUB_TOKEN")
        REPO_OWNER = "lalcivar"
        REPO_NAME = "DocumentacionTecnicaGenerada"
        CARPETA_DESTINO = "ManualTecnico"
        
        nombre_archivo = os.path.basename(filepath_local)
        path_github = f"{CARPETA_DESTINO}/{nombre_archivo}"
        url_api = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{path_github}"

        try:
            with open(filepath_local, "rb") as file_data:
                contenido_base64 = base64.b64encode(file_data.read()).decode("utf-8")

            # Cabecera user - agent obligatoria para GitHub REST API
            headers = {
                "Authorization": f"Bearer {GITHUB_TOKEN}",
                "Accept": "application/vnd.github+json",
                "User-Agent": "LLMs-DocApp"
            }
            
            data = {
                "message": f"build(artifacts): empaquetado de código fuente, documentación IA y métricas NLP ({nombre_archivo})",
                "content": contenido_base64,
                "branch": "main"
            }

            respuesta = requests.put(url_api, json=data, headers=headers, timeout=30)

            if respuesta.status_code in [200, 201]:
                return True, f"¡Sincronizado con éxito en GitHub!\nRuta: /{path_github}"
            else:
                error_msg = respuesta.json().get("message", "Error de conexión")
                return False, f"No se pudo subir a GitHub (Código {respuesta.status_code}): {error_msg}"

        except Exception as e:
            return False, f"Error en el proceso de sincronización con GitHub: {str(e)}"

    def cancelar_proceso(self):

        """Detiene la ejecución activa y restablece los controles de la interfaz."""

        self.cancelar_solicitado = True
       
        # Notificar al usuario
        self.lbl_estado.configure(text="Proceso cancelado por el usuario.", text_color="orange")
        
        # Restablecer botones, combos y detener animación
        self._restablecer_botones_ui()

    def _restablecer_botones_ui(self):
        """Garantiza la restauración segura de la UI en el hilo principal de Tkinter."""
        def _aplicar():
            self.generando = False
            # Desactivar barra de progreso si aplica
            if hasattr(self, 'progress_bar'):
                self.progress_bar.set(1.0)
                
            # Habilitar 'Generar' y Deshabilitar 'Cancelar'
            self.btn_generar.configure(state="normal", text="⚡ GENERAR DOCUMENTACIÓN")
            self.btn_cancelar.configure(state="disabled")
            self.btn_salir.configure(state="normal")
            
            # Resetear combos
            self.combo_modelos.set("-- Seleccionar un modelo --")
            self.combo_artefacto.set("-- Seleccionar tipo de documento --")

            # Si se invoca desde un hilo secundario, forzar ejecución en MainThread
        self.after(0, _aplicar)


    
# Ejecutar la aplicación
if __name__ == "__main__":
    app = AplicacionDocumentador()
    app.mainloop()