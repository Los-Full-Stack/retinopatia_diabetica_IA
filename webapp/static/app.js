const DEMO_USER = "admin";
const DEMO_PASSWORD = "admin123";

const state = {
  patients: JSON.parse(localStorage.getItem("retinaai_patients") || "[]"),
  activePatientId: localStorage.getItem("retinaai_active_patient") || null,
  history: JSON.parse(localStorage.getItem("retinaai_history") || "[]"),
  currentOriginalImage: null,
  currentGradcamImage: null,
};

const gradeDescriptions = {
  0: "Sin signos detectados de retinopatia diabetica.",
  1: "Hallazgos leves; requiere seguimiento segun criterio clinico.",
  2: "Retinopatia moderada; caso referible para evaluacion.",
  3: "Retinopatia severa; prioridad alta de revision.",
  4: "Retinopatia proliferativa; prioridad alta de derivacion.",
};

const diagnosisGuidance = {
  0: {
    title: "Conducta recomendada",
    text: "Continuar controles periodicos de salud visual. Reforzar control de glucosa, presion arterial y seguimiento metabolico. Programar nueva evaluacion retinal segun criterio profesional o protocolo local.",
  },
  1: {
    title: "Conducta recomendada",
    text: "Programar control de seguimiento. Reforzar medidas de control metabolico y educar al paciente sobre signos de alarma visual. Considerar evaluacion oftalmologica si existen sintomas, factores de riesgo o progresion.",
  },
  2: {
    title: "Conducta recomendada",
    text: "Sugerir evaluacion oftalmologica para confirmacion y manejo. Revisar la calidad de la captura y repetir la imagen si no permite una valoracion clara. Registrar la orientacion como apoyo, no como diagnostico definitivo.",
  },
  3: {
    title: "Conducta recomendada",
    text: "Priorizar derivacion oftalmologica. Documentar hallazgos, revisar sintomas de alarma y recomendar atencion especializada oportuna para reducir riesgo de progresion visual.",
  },
  4: {
    title: "Conducta recomendada",
    text: "Indicar derivacion oftalmologica prioritaria. Registrar el caso como alta prioridad y orientar al paciente sobre la necesidad de evaluacion especializada inmediata o segun disponibilidad del servicio.",
  },
};

const classNames = [
  "Sin retinopatia diabetica",
  "Retinopatia diabetica leve",
  "Retinopatia diabetica moderada",
  "Retinopatia diabetica severa",
  "Retinopatia diabetica proliferativa",
];

const decisionContent = {
  accept: {
    title: "Resultado orientativo",
    subtitle: "La imagen y la respuesta del sistema son consistentes para apoyar la evaluacion.",
    action: "Registrar como apoyo a la evaluacion y continuar con el criterio profesional.",
    headline: "Puede usarse como apoyo en la consulta",
  },
  review: {
    title: "Resultado para revisar",
    subtitle: "El sistema encontro una orientacion probable, pero conviene revisarla antes de registrarla.",
    action: "Contrastar la imagen con el criterio profesional y decidir seguimiento o derivacion.",
    headline: "Orientacion preliminar",
  },
  reject: {
    title: "Resultado para confirmar",
    subtitle: "El sistema entrega una orientacion inicial y sugiere confirmarla antes de registrarla como conclusion.",
    action: "Revisar la captura y el contexto del paciente. Si la imagen no es clara, repetir captura o derivar segun criterio profesional.",
    headline: "Orientacion inicial",
  },
};

const reasonLabels = {
  confidence_below_review_threshold: "Confianza limitada",
  confidence_requires_review: "Requiere revision",
  confidence_below_auto_accept_threshold: "Confianza limitada",
  top2_margin_too_low: "Diferencia estrecha",
  top2_margin_requires_review: "Requiere revision",
  image_quality_not_acceptable: "Calidad a mejorar",
  image_quality_rejected: "Calidad a mejorar",
  "Imagen borrosa.": "Imagen borrosa",
  "Confianza insuficiente para aceptacion automatica.": "Confianza limitada",
  "No usar como decision automatica; repetir captura o derivar.": "Confirmar antes de registrar",
};

function $(id) {
  return document.getElementById(id);
}

function saveState() {
  localStorage.setItem("retinaai_patients", JSON.stringify(state.patients));
  localStorage.setItem("retinaai_history", JSON.stringify(state.history));
  if (state.activePatientId) {
    localStorage.setItem("retinaai_active_patient", state.activePatientId);
  }
}

function activePatient() {
  return state.patients.find((patient) => patient.id === state.activePatientId) || null;
}

function setView(authenticated) {
  if ($("loginView")) $("loginView").hidden = authenticated;
  if ($("appView")) $("appView").hidden = !authenticated;
}

function switchTab(name) {
  document.querySelectorAll(".tab").forEach((tab) => {
    tab.classList.toggle("active", tab.dataset.tab === name);
  });
  document.querySelectorAll(".tab-panel").forEach((panel) => {
    panel.classList.toggle("active", panel.id === name);
  });
  if (name === "history") renderHistory();
}

function renderPatients() {
  const select = $("patientSelect");
  select.innerHTML = "";
  if (!state.patients.length) {
    const option = document.createElement("option");
    option.textContent = "Sin pacientes";
    option.value = "";
    select.appendChild(option);
    $("patientCard").className = "patient-card empty";
    $("patientCard").textContent = "Crea o selecciona un paciente.";
    $("activePatientBadge").textContent = "Sin paciente";
    return;
  }

  if (!activePatient()) state.activePatientId = state.patients[0].id;
  for (const patient of state.patients) {
    const option = document.createElement("option");
    option.value = patient.id;
    option.textContent = `${patient.name} - ${patient.identifier || "sin ID"}`;
    select.appendChild(option);
  }
  select.value = state.activePatientId;
  renderActivePatient();
}

function renderActivePatient() {
  const patient = activePatient();
  if (!patient) return renderPatients();
  $("activePatientBadge").textContent = patient.name;
  $("patientCard").className = "patient-card";
  $("patientCard").innerHTML = `
    <strong>${patient.name}</strong>
    <span>Identificacion</span>${patient.identifier || "No registrada"}
    <span>Edad / sexo</span>${patient.age} - ${patient.sex}
    <span>Notas</span>${patient.notes || "Sin notas"}
  `;
  saveState();
}

async function checkHealth() {
  try {
    const response = await fetch("/health");
    const data = await response.json();
    $("healthPill").textContent = data.status === "ok" ? "Sistema listo" : "Sistema no disponible";
    $("healthPill").style.background = data.status === "ok" ? "#ecfdf3" : "#fff7ed";
    $("healthPill").style.color = data.status === "ok" ? "#166534" : "#9a3412";
  } catch {
    $("healthPill").textContent = "Backend sin conexion";
    $("healthPill").style.background = "#fff7f7";
    $("healthPill").style.color = "#b42318";
  }
}

function renderProbabilities(probabilities) {
  return probabilities.map((probability, idx) => {
    const pct = Math.max(0, Math.min(1, probability)) * 100;
    return `
      <div class="probability-row" title="${classNames[idx]}">
        <span class="prob-label">G${idx}</span>
        <div class="prob-track"><div class="prob-fill" style="width:${pct.toFixed(1)}%"></div></div>
        <span class="prob-value">${pct.toFixed(1)}%</span>
      </div>
      <small class="prob-caption">${classNames[idx]}</small>
    `;
  }).join("");
}

function confidenceLabel(confidence) {
  if (confidence >= 0.80) return "Alta";
  if (confidence >= 0.70) return "Media";
  return "Limitado";
}

function imageQualityLabel(quality) {
  if (quality.acceptable) return "Adecuada";
  if (quality.warnings?.length) return "Revisar captura";
  return "Con observaciones";
}

function formatReasons(result) {
  const reasons = [...(result.policy_reasons || []), ...(result.image_quality?.warnings || [])];
  const fallback = result.warnings || [];
  const labels = reasons.length ? reasons : fallback;
  const clean = labels
    .map((reason) => reasonLabels[reason] || reason)
    .filter(Boolean)
    .filter((reason) => !String(reason).includes("_"));
  return [...new Set(clean)].slice(0, 3);
}

function renderResult(payload, filename) {
  const result = payload.result;
  const decision = decisionContent[result.policy_decision] || decisionContent.reject;
  const guidance = diagnosisGuidance[result.class_id] || diagnosisGuidance[0];
  const q = result.image_quality;
  const reasons = formatReasons(result);
  const confidencePct = (result.confidence * 100).toFixed(1);
  const coveragePct = (q.retina_coverage * 100).toFixed(1);
  $("resultEmpty").hidden = true;
  $("resultContent").hidden = false;
  if (result.explanation?.overlay) {
    state.currentGradcamImage = result.explanation.overlay;
    $("gradcamImage").src = result.explanation.overlay;
    $("gradcamFigure").hidden = false;
  } else {
    state.currentGradcamImage = null;
    $("gradcamFigure").hidden = true;
  }
  $("resultContent").innerHTML = `
    <article class="diagnosis-summary ${result.policy_decision}">
      <div class="summary-header">
        <span class="decision-pill">${decision.title}</span>
      </div>
      <h2>${decision.headline}</h2>
      <p>${decision.subtitle}</p>
      ${reasons.length ? `<div class="reason-list">${reasons.map((reason) => `<span>${reason}</span>`).join("")}</div>` : ""}
    </article>

    <section class="clinical-grid">
      <article class="clinical-card primary">
        <span>Posible resultado</span>
        <strong class="diagnosis-name">${result.class_name}</strong>
        <p>${gradeDescriptions[result.class_id]}</p>
      </article>
      <article class="clinical-card">
        <span>Nivel de apoyo</span>
        <strong>${confidenceLabel(result.confidence)}</strong>
        <p>${confidencePct}% de consistencia estimada por el sistema.</p>
      </article>
      <article class="clinical-card">
        <span>Calidad de imagen</span>
        <strong>${imageQualityLabel(q)}</strong>
        <p>${q.warnings?.[0] ? reasonLabels[q.warnings[0]] || q.warnings[0] : "Captura util para evaluacion."}</p>
      </article>
    </section>

    <section class="recommendation-box">
      <span>Plan sugerido</span>
      <strong>${decision.action}</strong>
      <p>Interpretacion: ${gradeDescriptions[result.class_id]}</p>
    </section>

    <section class="indication-box">
      <div>
        <span>${guidance.title}</span>
        <strong>Indicaciones para el paciente</strong>
        <p>Redactadas como apoyo para la consulta; el profesional puede adecuarlas al contexto clinico.</p>
      </div>
      <textarea id="clinicalIndications" rows="4">${guidance.text}</textarea>
    </section>

    <details class="details-panel">
      <summary>Ver detalle tecnico</summary>
      <div class="technical-block">
        <div class="quality-grid">
          <div class="quality-item"><span>Calidad</span><strong>${q.quality_score.toFixed(3)}</strong></div>
          <div class="quality-item"><span>Nitidez</span><strong>${q.sharpness.toFixed(1)}</strong></div>
          <div class="quality-item"><span>Contraste</span><strong>${q.contrast.toFixed(1)}</strong></div>
          <div class="quality-item"><span>Cobertura</span><strong>${coveragePct}%</strong></div>
          <div class="quality-item"><span>Segunda opcion</span><strong>G${result.second_class_id}</strong></div>
          <div class="quality-item"><span>Diferencia</span><strong>${result.top2_margin.toFixed(3)}</strong></div>
        </div>
      </div>
      <div class="probability-list">${renderProbabilities(result.probabilities)}</div>
    </details>

    <div class="result-actions">
      <button id="saveDiagnosisBtn">Guardar resultado</button>
      <button id="downloadDiagnosisBtn" class="ghost-btn">Descargar reporte</button>
    </div>
  `;

  const record = {
    id: crypto.randomUUID(),
    createdAt: new Date().toISOString(),
    filename,
    patient: activePatient(),
    prediction: result,
  };
  $("saveDiagnosisBtn").addEventListener("click", () => {
    record.indications = $("clinicalIndications")?.value.trim() || "";
    state.history.push(record);
    saveState();
    renderHistory();
    alert("Resultado guardado.");
  });
  $("downloadDiagnosisBtn").addEventListener("click", () => {
    record.indications = $("clinicalIndications")?.value.trim() || "";
    downloadJson(record, `retinaai_${record.id}.json`);
  });
}

function downloadJson(data, filename) {
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}

function renderHistory() {
  const list = $("historyList");
  const patient = activePatient();
  const items = state.history.filter((record) => !patient || record.patient?.id === patient.id).reverse();
  if (!items.length) {
    list.innerHTML = `<div class="placeholder">No hay resultados guardados para este paciente.</div>`;
    return;
  }
  list.innerHTML = items.map((record) => {
    const pred = record.prediction;
    const decision = decisionContent[pred.policy_decision] || decisionContent.reject;
    return `
      <article class="history-item">
        <strong>${decision.title}</strong>
        <small>${new Date(record.createdAt).toLocaleString()} - ${record.filename}</small>
        <p>${pred.class_name} - confianza ${(pred.confidence * 100).toFixed(2)}%</p>
      </article>
    `;
  }).join("");
}

function bindEvents() {
  if ($("loginForm")) {
    $("loginForm").addEventListener("submit", (event) => {
    event.preventDefault();
    const ok = $("username").value === DEMO_USER && $("password").value === DEMO_PASSWORD;
    $("loginError").hidden = ok;
    if (ok) {
      sessionStorage.setItem("retinaai_auth", "1");
      window.location.href = "/dashboard";
    }
  });
  }

  if (!$("appView")) return;

  $("logoutBtn").addEventListener("click", () => {
    sessionStorage.removeItem("retinaai_auth");
    window.location.href = "/";
  });

  document.querySelectorAll(".tab").forEach((tab) => {
    tab.addEventListener("click", () => switchTab(tab.dataset.tab));
  });

  $("patientForm").addEventListener("submit", (event) => {
    event.preventDefault();
    const patient = {
      id: crypto.randomUUID(),
      name: $("patientName").value.trim(),
      identifier: $("patientIdentifier").value.trim(),
      age: Number($("patientAge").value || 0),
      sex: $("patientSex").value,
      notes: $("patientNotes").value.trim(),
      createdAt: new Date().toISOString(),
    };
    if (!patient.name) return;
    state.patients.push(patient);
    state.activePatientId = patient.id;
    event.target.reset();
    renderPatients();
    switchTab("diagnosis");
  });

  $("patientSelect").addEventListener("change", (event) => {
    state.activePatientId = event.target.value;
    renderActivePatient();
    renderHistory();
  });

  $("imageInput").addEventListener("change", async (event) => {
    const file = event.target.files?.[0];
    if (!file) return;
    if (!activePatient()) {
      alert("Crea o selecciona un paciente antes de diagnosticar.");
      event.target.value = "";
      switchTab("patients");
      return;
    }

    const previewUrl = URL.createObjectURL(file);
    state.currentOriginalImage = previewUrl;
    state.currentGradcamImage = null;
    $("previewImage").src = previewUrl;
    $("imageComparison").hidden = false;
    $("imagePlaceholder").hidden = true;
    $("gradcamFigure").hidden = true;
    $("gradcamImage").removeAttribute("src");
    $("resultEmpty").hidden = false;
    $("resultEmpty").textContent = "Analizando imagen...";
    $("resultContent").hidden = true;

    const formData = new FormData();
    formData.append("file", file);
    try {
      const response = await fetch("/predict", { method: "POST", body: formData });
      if (!response.ok) {
        const error = await response.json().catch(() => ({}));
        throw new Error(error.detail || "No se pudo ejecutar la prediccion.");
      }
      const payload = await response.json();
      renderResult(payload, file.name);
    } catch (error) {
      $("resultEmpty").hidden = false;
      $("resultEmpty").textContent = `No se pudo conectar con el analisis. Verifica que el servicio de Render este desplegado y vuelve a intentar. Detalle: ${error.message}`;
      $("resultContent").hidden = true;
    }
  });

  $("clearHistoryBtn").addEventListener("click", () => {
    if (!confirm("Limpiar historial local?")) return;
    const patient = activePatient();
    state.history = patient ? state.history.filter((record) => record.patient?.id !== patient.id) : [];
    saveState();
    renderHistory();
  });
}

function init() {
  const authenticated = sessionStorage.getItem("retinaai_auth") === "1";

  if (document.body.classList.contains("dashboard-page") && !authenticated) {
    window.location.replace("/");
    return;
  }

  if (document.body.classList.contains("login-page") && authenticated) {
    window.location.replace("/dashboard");
    return;
  }

  bindEvents();

  if ($("appView")) {
    setView(true);
    if (authenticated) {
      checkHealth();
      renderPatients();
      renderHistory();
    }
  }
}

init();
