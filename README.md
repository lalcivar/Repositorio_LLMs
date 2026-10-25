# Repositorio_LLMs: Evaluación Experimental de Documentación Técnica con LLMs

Este repositorio reúne los artefactos, archivos de soporte y scripts de ejecución correspondientes a la evaluación experimental de Modelos de Lenguaje de Gran Escala (LLMs) en tareas de generación de documentación técnica.

## Estructura del Proyecto

La estructura del proyecto comprende los siguientes componentes:

*   **`dataSet/`**: Contiene el corpus de código fuente original utilizado en el experimento.
    *   Este corpus se compone de 30 archivos seleccionados (5 por cada uno) de 6 repositorios de código abierto maduros:
        *   **Python:** `flask`, `requests`, `scikit-learn`
        *   **Java:** `commons-lang`, `guava`, `junit5`
*   **`prompts.xlsx`**: Archivo de configuración que detalla las configuraciones metodológicas de los *prompts* parametrizados utilizados en las distintas fases del experimento, incluyendo las versiones finales adaptadas por lenguaje (Python y Java) para la ejecución principal.
*   **`openrouter_execution_logs.csv`**: El registro consolidado de peticiones a la API en OpenRouter, capturando la actividad completa del experimento.
*   **`sistema_evaluacion_LLMs.py`**: El script principal de Python que conforma el entorno de interfaz e interacción programática, y gestiona la lógica de la arquitectura híbrida para la generación y evaluación de la documentación.
*   **`README.md`**: Este archivo, que proporciona una visión general del repositorio y sus contenidos.

## Descripción del Experimento (Contexto Rápido)

Este repositorio soporta un experimento que evalúa 4 modelos comerciales de LLM (Google - Gemini, Anthropic - Claude, OpenAI - ChatGPT, Meta - Llama) bajo dos condiciones (con y sin contexto sintáctico AST/Regex) para generar Docstrings y Javadoc. Se realizaron 240 ejecuciones sobre 30 archivos de código fuente Python y Java. Los resultados detallados de este experimento se encuentran en el repositorio de resultados específico, enlazado más abajo.

## Configuración y Ejecución Local

Con el propósito de cumplir los estándares de seguridad computacional y prevenir la divulgación involuntaria de credenciales de acceso, todas las claves de interfaz de programación de aplicaciones (API keys) vinculadas a los servicios de Hugging Face, GitHub y OpenRouter **fueron removidas del código fuente público.**

Las funciones de invocación en `sistema_evaluacion_LLMs.py` leen las credenciales mediante **variables de entorno del sistema operativo.** Para ejecutar los módulos de forma local, el investigador debe configurar previamente las claves requeridas en sus respectivas variables antes de iniciar el entorno de ejecución.

## Repositorio de Resultados

Los artefactos de documentación generada, junto con todos los informes de métricas detallados del experimento, se encuentran en un repositorio separado para su consulta y auditoría:

[**Acceder a Resultados del Experimento**](https://github.com/lalcivar/DocumentacionTecnicaGenerada)

---
