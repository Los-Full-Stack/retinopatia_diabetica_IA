import json
import os
import uuid
from datetime import datetime
from io import BytesIO
from pathlib import Path

import streamlit as st
from PIL import Image

from config.config import MODELS_DIR
from config.config import CLASS_NAMES
from src.inference import RetinaPredictor


APP_TITLE = "RetinaAI"
APP_SUBTITLE = "Asistente de analisis retinal con politica de seguridad"

DECISION_CONTENT = {
    "accept": {
        "css": "accept",
        "title": "Prediccion aceptada",
        "subtitle": "La imagen y la confianza cumplen la politica de seguridad del prototipo.",
        "action": "Puede registrarse como resultado automatico del prototipo, manteniendo supervision clinica.",
    },
    "review": {
        "css": "review",
        "title": "Revision recomendada",
        "subtitle": "El resultado es informativo, pero no alcanza seguridad suficiente para aceptacion automatica.",
        "action": "Enviar a revision profesional antes de tomar decisiones.",
    },
    "reject": {
        "css": "reject",
        "title": "No usar automaticamente",
        "subtitle": "La imagen, la confianza o la ambiguedad no permiten usar la prediccion como valida.",
        "action": "Repetir captura si es posible o derivar a evaluacion.",
    },
}

GRADE_DESCRIPTIONS = {
    0: "Sin signos detectados de retinopatia diabetica.",
    1: "Hallazgos leves; requiere seguimiento segun criterio clinico.",
    2: "Retinopatia moderada; caso referible para evaluacion.",
    3: "Retinopatia severa; prioridad alta de revision.",
    4: "Retinopatia proliferativa; prioridad alta de derivacion.",
}


def get_secret(name: str, default: str) -> str:
    env_name = f"RETINAAI_{name.upper()}"
    if os.environ.get(env_name):
        return os.environ[env_name]
    try:
        return str(st.secrets["auth"][name])
    except Exception:
        return default


def init_state() -> None:
    st.session_state.setdefault("authenticated", False)
    st.session_state.setdefault("patients", {})
    st.session_state.setdefault("selected_patient_id", None)
    st.session_state.setdefault("diagnoses", {})


@st.cache_resource(show_spinner="Cargando modelo RetinaAI...")
def load_predictor() -> RetinaPredictor:
    return RetinaPredictor()


def model_available() -> bool:
    return (MODELS_DIR / "best_retina_model.pth").exists()


def page_css() -> None:
    st.markdown(
        """
        <style>
        :root {
          --bg: #f5f7fb;
          --ink: #172033;
          --muted: #667085;
          --line: #d9e0ea;
          --blue: #2563eb;
          --green: #15803d;
          --amber: #b45309;
          --red: #b42318;
          --panel: #ffffff;
        }
        .stApp {
          background: linear-gradient(180deg, #f7faff 0%, #f4f6fa 42%, #eef3f8 100%);
        }
        .block-container {
          padding-top: 1.4rem;
          padding-bottom: 2.5rem;
          max-width: 1260px;
        }
        h1, h2, h3 {
          color: var(--ink);
          letter-spacing: 0;
        }
        [data-testid="stMetric"] {
          background: white;
          border: 1px solid var(--line);
          border-radius: 8px;
          padding: 12px 14px;
        }
        .hero {
          display: grid;
          grid-template-columns: 1fr auto;
          gap: 16px;
          align-items: end;
          padding: 18px 0 16px;
          border-bottom: 1px solid var(--line);
          margin-bottom: 18px;
        }
        .hero-title {
          font-size: 34px;
          font-weight: 750;
          color: var(--ink);
          line-height: 1.1;
        }
        .hero-subtitle {
          color: var(--muted);
          font-size: 15px;
          margin-top: 6px;
        }
        .hero-pill {
          display: inline-flex;
          align-items: center;
          gap: 8px;
          background: #102033;
          color: white;
          border-radius: 999px;
          padding: 9px 13px;
          font-size: 13px;
          white-space: nowrap;
        }
        .hero-pill-dot {
          width: 8px;
          height: 8px;
          border-radius: 999px;
          background: #22c55e;
        }
        .notice {
          border: 1px solid var(--line);
          border-left: 5px solid var(--blue);
          background: var(--panel);
          padding: 12px 14px;
          border-radius: 8px;
          color: var(--ink);
        }
        .warning-panel {
          border: 1px solid #fecaca;
          border-left: 5px solid var(--red);
          background: #fff7f7;
          padding: 14px;
          border-radius: 8px;
          color: #7f1d1d;
          margin-bottom: 14px;
        }
        .soft-panel {
          background: var(--panel);
          border: 1px solid var(--line);
          border-radius: 8px;
          padding: 16px;
          margin-bottom: 12px;
        }
        .soft-panel h3 {
          margin: 0 0 6px 0;
          font-size: 18px;
        }
        .soft-panel p {
          margin: 0;
          color: var(--muted);
          line-height: 1.45;
        }
        .step-grid {
          display: grid;
          grid-template-columns: repeat(3, minmax(0, 1fr));
          gap: 10px;
          margin: 10px 0 18px;
        }
        .step {
          background: var(--panel);
          border: 1px solid var(--line);
          border-radius: 8px;
          padding: 12px;
        }
        .step strong {
          display: block;
          color: var(--ink);
          margin-bottom: 4px;
        }
        .step span {
          color: var(--muted);
          font-size: 13px;
        }
        .decision {
          border-radius: 8px;
          padding: 18px;
          color: white;
          margin-bottom: 12px;
          box-shadow: 0 12px 26px rgba(16, 32, 51, .12);
        }
        .decision strong {
          font-size: 24px;
          display: block;
          line-height: 1.15;
        }
        .decision span {
          font-size: 14px;
          opacity: .95;
        }
        .accept { background: #166534; }
        .review { background: #a16207; }
        .reject { background: #b42318; }
        .action-box {
          background: #fff;
          border: 1px solid var(--line);
          border-radius: 8px;
          padding: 12px;
          margin: 10px 0;
        }
        .action-box strong {
          display: block;
          margin-bottom: 4px;
          color: var(--ink);
        }
        .result-summary {
          display: grid;
          grid-template-columns: repeat(3, minmax(0, 1fr));
          gap: 10px;
          margin: 12px 0;
        }
        .result-chip {
          background: #fff;
          border: 1px solid var(--line);
          border-radius: 8px;
          padding: 10px 12px;
        }
        .result-chip span {
          display: block;
          color: var(--muted);
          font-size: 12px;
        }
        .result-chip strong {
          color: var(--ink);
          font-size: 16px;
        }
        .hero-card {
          background: #ffffff;
          border: 1px solid var(--line);
          border-radius: 8px;
          padding: 18px;
          min-height: 170px;
          position: relative;
          overflow: hidden;
        }
        .hero-card:after {
          content: "";
          position: absolute;
          right: -60px;
          top: -70px;
          width: 180px;
          height: 180px;
          border-radius: 50%;
          background: rgba(37, 99, 235, .10);
        }
        .hero-card h3 {
          margin-top: 0;
        }
        .model-strip {
          display: grid;
          grid-template-columns: repeat(4, minmax(0, 1fr));
          gap: 10px;
          margin: 12px 0 18px;
        }
        .model-stat {
          background: #fff;
          border: 1px solid var(--line);
          border-radius: 8px;
          padding: 12px;
        }
        .model-stat span {
          display: block;
          color: var(--muted);
          font-size: 12px;
        }
        .model-stat strong {
          display: block;
          color: var(--ink);
          font-size: 22px;
          line-height: 1.2;
          margin-top: 4px;
        }
        .patient-banner {
          display: grid;
          grid-template-columns: 1.2fr .8fr .8fr;
          gap: 12px;
          background: #102033;
          color: white;
          border-radius: 8px;
          padding: 15px;
          margin-bottom: 14px;
        }
        @media (max-width: 900px) {
          .hero,
          .patient-banner,
          .result-summary,
          .model-strip,
          .step-grid {
            grid-template-columns: 1fr;
          }
          .hero-pill {
            justify-content: center;
          }
        }
        .patient-banner span {
          display: block;
          opacity: .75;
          font-size: 12px;
        }
        .patient-banner strong {
          font-size: 16px;
        }
        .small-muted {
          color: var(--muted);
          font-size: 13px;
        }
        .class-row {
          display: grid;
          grid-template-columns: 96px 1fr 58px;
          gap: 10px;
          align-items: center;
          margin: 8px 0;
          font-size: 14px;
        }
        .bar-bg {
          height: 11px;
          background: #e8edf5;
          border-radius: 999px;
          overflow: hidden;
        }
        .bar-fill {
          height: 11px;
          background: #2563eb;
          border-radius: 999px;
        }
        .legend {
          display: grid;
          gap: 8px;
          margin-top: 8px;
        }
        .legend-item {
          display: grid;
          grid-template-columns: 14px 1fr;
          gap: 8px;
          align-items: start;
          font-size: 13px;
          color: var(--muted);
        }
        .dot {
          width: 12px;
          height: 12px;
          border-radius: 999px;
          margin-top: 3px;
        }
        .dot.accept { background: #166534; }
        .dot.review { background: #a16207; }
        .dot.reject { background: #b42318; }
        .footer-note {
          color: var(--muted);
          font-size: 12px;
          border-top: 1px solid var(--line);
          padding-top: 12px;
          margin-top: 20px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def header() -> None:
    st.markdown(
        f"""
        <div class="hero">
          <div>
            <div class="hero-title">{APP_TITLE}</div>
            <div class="hero-subtitle">{APP_SUBTITLE}</div>
          </div>
          <div class="hero-pill"><span class="hero-pill-dot"></span> Modelo congelado · v1</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def model_missing_view() -> None:
    header()
    st.markdown(
        f"""
        <div class="warning-panel">
          <strong>No se encontro el checkpoint del modelo.</strong><br>
          La app necesita <code>{MODELS_DIR / "best_retina_model.pth"}</code> para ejecutar inferencia.
          En Streamlit Cloud puedes subirlo con Git LFS o descargarlo desde un alojamiento externo antes de iniciar.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.code(str(MODELS_DIR / "best_retina_model.pth"), language="text")


def login_view() -> None:
    header()
    left, right = st.columns([1.1, 0.9])
    with left:
        st.markdown(
            """
            <div class="hero-card">
              <h3>De imagen retinal a decision operativa</h3>
              <p>
                Esta interfaz no intenta vender una certeza falsa: separa cada caso
                en aceptado, revision o rechazo, y muestra por que.
              </p>
            </div>
            <div class="soft-panel">
              <h3>Prototipo visual para tamizaje asistido</h3>
              <p>
                RetinaAI organiza el flujo de paciente, analiza una imagen retinal y
                separa la salida en tres decisiones: aceptar, revisar o rechazar.
                La demo no usa base de datos; todo vive en la sesion del navegador.
              </p>
            </div>
            <div class="step-grid">
              <div class="step"><strong>1. Crear paciente</strong><span>Registra datos basicos para asociar el analisis.</span></div>
              <div class="step"><strong>2. Cargar imagen</strong><span>Evalua calidad, confianza y prediccion del modelo.</span></div>
              <div class="step"><strong>3. Guardar resultado</strong><span>Conserva el historial de la sesion y exporta JSON.</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            """
            <div class="notice">
              Acceso de demostracion. Configura credenciales en
              <code>.streamlit/secrets.toml</code> o variables de entorno.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write("")
        with st.form("login_form"):
            username = st.text_input("Usuario")
            password = st.text_input("Clave", type="password")
            submitted = st.form_submit_button("Entrar", use_container_width=True)
    if submitted:
        expected_user = get_secret("username", "admin")
        expected_password = get_secret("password", "admin123")
        if username == expected_user and password == expected_password:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Credenciales invalidas.")


def patient_form() -> None:
    st.subheader("Nuevo paciente")
    with st.form("patient_form", clear_on_submit=True):
        name = st.text_input("Nombre completo")
        identifier = st.text_input("Identificacion")
        age = st.number_input("Edad", min_value=0, max_value=120, value=45)
        sex = st.selectbox("Sexo", ["No especificado", "Femenino", "Masculino"])
        notes = st.text_area("Notas clinicas", height=90)
        submitted = st.form_submit_button("Crear paciente", use_container_width=True)

    if submitted:
        if not name.strip():
            st.warning("Ingresa el nombre del paciente.")
            return
        patient_id = str(uuid.uuid4())
        patient = {
            "patient_id": patient_id,
            "name": name.strip(),
            "identifier": identifier.strip(),
            "age": int(age),
            "sex": sex,
            "notes": notes.strip(),
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        st.session_state.patients[patient_id] = patient
        st.session_state.diagnoses[patient_id] = []
        st.session_state.selected_patient_id = patient_id
        st.success("Paciente creado.")
        st.rerun()


def patient_selector() -> dict | None:
    patients = st.session_state.patients
    if not patients:
        st.info("Crea un paciente para iniciar un diagnostico.")
        return None

    labels = {
        patient_id: f"{patient['name']} - {patient.get('identifier') or 'sin ID'}"
        for patient_id, patient in patients.items()
    }
    selected = st.selectbox(
        "Paciente activo",
        list(labels.keys()),
        index=list(labels.keys()).index(st.session_state.selected_patient_id)
        if st.session_state.selected_patient_id in labels
        else 0,
        format_func=lambda patient_id: labels[patient_id],
        key="active_patient_selector",
    )
    st.session_state.selected_patient_id = selected
    return patients[selected]


def decision_label(decision: str) -> tuple[str, str, str]:
    content = DECISION_CONTENT.get(decision, DECISION_CONTENT["reject"])
    return content["css"], content["title"], content["subtitle"]


def active_patient_banner(patient: dict) -> None:
    st.markdown(
        f"""
        <div class="patient-banner">
          <div><span>Paciente activo</span><strong>{patient['name']}</strong></div>
          <div><span>Identificacion</span><strong>{patient.get('identifier') or 'No registrada'}</strong></div>
          <div><span>Edad / sexo</span><strong>{patient['age']} - {patient['sex']}</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("### RetinaAI")
        st.caption("Modelo congelado con politica de seguridad.")
        st.markdown("#### Flujo")
        st.write("1. Crear o seleccionar paciente")
        st.write("2. Cargar imagen retinal")
        st.write("3. Revisar decision")
        st.write("4. Guardar o exportar resultado")
        st.markdown("#### Leyenda")
        st.markdown(
            """
            <div class="legend">
              <div class="legend-item"><div class="dot accept"></div><div><strong>Accept</strong><br>Confianza e imagen suficientes.</div></div>
              <div class="legend-item"><div class="dot review"></div><div><strong>Review</strong><br>Resultado orientativo.</div></div>
              <div class="legend-item"><div class="dot reject"></div><div><strong>Reject</strong><br>Repetir captura o derivar.</div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("#### Version")
        st.caption("Checkpoint: original_only_b1")
        st.caption("No es diagnostico autonomo.")


def render_probabilities(probabilities: list[float]) -> None:
    st.markdown("#### Probabilidades por grado")
    for idx, probability in enumerate(probabilities):
        pct = max(0.0, min(1.0, float(probability))) * 100
        st.markdown(
            f"""
            <div class="class-row">
              <div>Grado {idx}</div>
              <div class="bar-bg"><div class="bar-fill" style="width:{pct:.1f}%"></div></div>
              <div>{pct:.1f}%</div>
            </div>
            <div class="small-muted">{CLASS_NAMES[idx]} - {GRADE_DESCRIPTIONS[idx]}</div>
            """,
            unsafe_allow_html=True,
        )


def render_interpretation(result: dict) -> None:
    content = DECISION_CONTENT.get(result["policy_decision"], DECISION_CONTENT["reject"])
    class_id = int(result["class_id"])
    st.markdown(
        f"""
        <div class="result-summary">
          <div class="result-chip"><span>Decision</span><strong>{content["title"]}</strong></div>
          <div class="result-chip"><span>Confianza minima auto</span><strong>{result["required_confidence"]:.2f}</strong></div>
          <div class="result-chip"><span>Accion</span><strong>{result["clinical_action"]}</strong></div>
        </div>
        <div class="action-box">
          <strong>Interpretacion del resultado</strong>
          Grado {class_id}: {GRADE_DESCRIPTIONS[class_id]}
        </div>
        <div class="action-box">
          <strong>Siguiente accion sugerida</strong>
          {content["action"]}
        </div>
        """,
        unsafe_allow_html=True,
    )


def result_to_record(patient: dict, image_name: str, result: dict) -> dict:
    return {
        "record_id": str(uuid.uuid4()),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "patient": patient,
        "image_name": image_name,
        "prediction": {
            "class_id": result["class_id"],
            "class_name": result["class_name"],
            "confidence": result["confidence"],
            "second_class_id": result["second_class_id"],
            "second_class_name": result["second_class_name"],
            "second_confidence": result["second_confidence"],
            "top2_margin": result["top2_margin"],
            "policy_decision": result["policy_decision"],
            "clinical_action": result["clinical_action"],
            "policy_reasons": result["policy_reasons"],
            "probabilities": result["probabilities"],
            "image_quality": result["image_quality"],
            "warnings": result["warnings"],
        },
    }


def diagnosis_view(patient: dict) -> None:
    st.subheader("Diagnostico asistido")
    active_patient_banner(patient)
    st.markdown(
        """
        <div class="soft-panel">
          <h3>Carga de imagen</h3>
          <p>
            Usa una imagen retinal centrada y con buena nitidez. Si la calidad no es suficiente,
            la politica bloquea la prediccion automatica aunque el modelo entregue una clase.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    uploaded = st.file_uploader(
        "Imagen retinal",
        type=["png", "jpg", "jpeg"],
        help="Usa imagenes JPG/PNG. Para mejores resultados, carga una captura enfocada, centrada y sin sombras fuertes.",
    )
    if uploaded is None:
        st.markdown(
            '<div class="notice">Carga una imagen para ver la vista previa, la calidad, la prediccion y la accion recomendada.</div>',
            unsafe_allow_html=True,
        )
        return

    image_bytes = uploaded.read()
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    predictor = load_predictor()

    left, right = st.columns([1.08, 1])
    with left:
        st.image(image, caption=uploaded.name, use_container_width=True)

    with st.spinner("Analizando imagen..."):
        result = predictor.predict(image)

    with right:
        css_class, title, subtitle = decision_label(result["policy_decision"])
        st.markdown(
            f"""
            <div class="decision {css_class}">
              <strong>{title}</strong>
              <span>{subtitle}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.metric("Grado predicho", f"{result['class_id']} - {result['class_name']}")
        st.metric("Confianza", f"{100 * result['confidence']:.2f}%")
        st.metric("Margen top-2", f"{result['top2_margin']:.3f}")
        st.caption(f"Segunda clase: grado {result['second_class_id']} - {result['second_class_name']}")
        render_interpretation(result)

    q = result["image_quality"]
    st.markdown("#### Calidad de imagen")
    quality_cols = st.columns(4)
    quality_cols[0].metric("Calidad", f"{q['quality_score']:.3f}")
    quality_cols[1].metric("Nitidez", f"{q['sharpness']:.1f}")
    quality_cols[2].metric("Contraste", f"{q['contrast']:.1f}")
    quality_cols[3].metric("Cobertura", f"{100 * q['retina_coverage']:.1f}%")

    if result["warnings"]:
        st.warning(" | ".join(result["warnings"]))

    render_probabilities(result["probabilities"])

    record = result_to_record(patient, uploaded.name, result)
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Guardar en historial", use_container_width=True):
            st.session_state.diagnoses.setdefault(patient["patient_id"], []).append(record)
            st.success("Resultado guardado en la sesion.")
    with c2:
        st.download_button(
            "Descargar JSON",
            data=json.dumps(record, indent=2, ensure_ascii=False),
            file_name=f"retinaai_{patient['patient_id']}_{record['record_id']}.json",
            mime="application/json",
            use_container_width=True,
        )


def history_view(patient: dict) -> None:
    records = st.session_state.diagnoses.get(patient["patient_id"], [])
    st.subheader("Historial de la sesion")
    active_patient_banner(patient)
    if not records:
        st.caption("Este paciente aun no tiene diagnosticos guardados.")
        return

    for record in reversed(records):
        pred = record["prediction"]
        css_class, title, _ = decision_label(pred["policy_decision"])
        with st.expander(f"{record['created_at']} - {title} - grado {pred['class_id']}"):
            st.write(f"Imagen: `{record['image_name']}`")
            st.write(f"Prediccion: **{pred['class_name']}**")
            st.write(f"Confianza: `{pred['confidence']:.4f}`")
            st.write(f"Accion: `{pred['clinical_action']}`")
            st.write("Advertencias:", pred["warnings"])
            st.download_button(
                "Descargar este registro",
                data=json.dumps(record, indent=2, ensure_ascii=False),
                file_name=f"retinaai_record_{record['record_id']}.json",
                mime="application/json",
                key=f"download_{record['record_id']}",
            )


def dashboard() -> None:
    render_sidebar()
    header()
    top_left, top_right = st.columns([1, 1])
    with top_left:
        st.markdown(
            """
            <div class="notice">
              Modelo congelado: <strong>original_only_b1</strong>. Prototipo de investigacion,
              no reemplaza revision medica.
            </div>
            """,
            unsafe_allow_html=True,
        )
    with top_right:
        if st.button("Cerrar sesion", use_container_width=True):
            st.session_state.authenticated = False
            st.rerun()

    st.write("")
    st.markdown(
        """
        <div class="model-strip">
          <div class="model-stat"><span>Accuracy checkpoint</span><strong>79.09%</strong></div>
          <div class="model-stat"><span>Accuracy aceptada</span><strong>97.72%</strong></div>
          <div class="model-stat"><span>Cobertura accept</span><strong>39.82%</strong></div>
          <div class="model-stat"><span>Error grande accept</span><strong>0.91%</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tabs = st.tabs(["Pacientes", "Diagnostico", "Historial"])
    with tabs[0]:
        col_a, col_b = st.columns([0.9, 1.1])
        with col_a:
            patient_form()
        with col_b:
            selected_patient = patient_selector()
            if selected_patient:
                st.subheader("Ficha activa")
                st.write(f"**Nombre:** {selected_patient['name']}")
                st.write(f"**Identificacion:** {selected_patient.get('identifier') or 'No registrada'}")
                st.write(f"**Edad:** {selected_patient['age']}")
                st.write(f"**Sexo:** {selected_patient['sex']}")
                st.write(f"**Notas:** {selected_patient.get('notes') or 'Sin notas'}")

    patient = st.session_state.patients.get(st.session_state.selected_patient_id)
    with tabs[1]:
        if patient:
            diagnosis_view(patient)
        else:
            st.info("Crea o selecciona un paciente primero.")
    with tabs[2]:
        if patient:
            history_view(patient)
        else:
            st.info("Crea o selecciona un paciente primero.")

    st.markdown(
        """
        <div class="footer-note">
          RetinaAI es una herramienta de apoyo. Las salidas <code>review</code> y
          <code>reject</code> no deben convertirse en diagnostico automatico.
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="RetinaAI", page_icon="R", layout="wide")
    page_css()
    init_state()
    if not model_available():
        model_missing_view()
        return
    if not st.session_state.authenticated:
        login_view()
        return
    dashboard()


if __name__ == "__main__":
    main()
