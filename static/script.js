/* =================================================================
   AI Resume Analyzer — Frontend Logic
   ================================================================= */

document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("analyze-form");
  const fileInput = document.getElementById("resume-file");
  const uploadZone = document.getElementById("upload-zone");
  const fileName = document.getElementById("file-name");
  const btn = document.getElementById("btn-analyze");
  const errorMsg = document.getElementById("error-msg");
  const resultsSection = document.getElementById("results-section");

  // ---- Drag & Drop ----
  uploadZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    uploadZone.classList.add("drag-over");
  });
  uploadZone.addEventListener("dragleave", () =>
    uploadZone.classList.remove("drag-over")
  );
  uploadZone.addEventListener("drop", (e) => {
    e.preventDefault();
    uploadZone.classList.remove("drag-over");
    if (e.dataTransfer.files.length) {
      fileInput.files = e.dataTransfer.files;
      showFileName(e.dataTransfer.files[0].name);
    }
  });
  fileInput.addEventListener("change", () => {
    if (fileInput.files.length) showFileName(fileInput.files[0].name);
  });

  function showFileName(name) {
    fileName.textContent = "📎 " + name;
  }

  // ---- Form Submit ----
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    hideError();

    const resume = fileInput.files[0];
    const jd = document.getElementById("job-description").value.trim();
    const role = document.getElementById("role-select").value;

    if (!resume) return showError("Please upload your resume PDF.");
    if (!jd) return showError("Please paste the job description.");

    btn.classList.add("loading");
    btn.disabled = true;
    resultsSection.classList.remove("visible");

    try {
      const fd = new FormData();
      fd.append("resume", resume);
      fd.append("job_description", jd);
      if (role) fd.append("role", role);

      const res = await fetch("/analyze", { method: "POST", body: fd });
      const data = await res.json();

      if (!res.ok) {
        showError(data.error || "Analysis failed.");
        return;
      }

      renderResults(data);
    } catch (err) {
      showError("Network error. Please try again.");
    } finally {
      btn.classList.remove("loading");
      btn.disabled = false;
    }
  });

  function showError(msg) {
    errorMsg.textContent = msg;
    errorMsg.classList.add("visible");
  }
  function hideError() {
    errorMsg.classList.remove("visible");
  }

  // ================================================================
  // Render Results
  // ================================================================
  function renderResults(data) {
    renderATSScore(data.ats_score);
    renderSkillGap(data.skill_analysis);
    renderSectionScores(data.section_scores);
    renderKeywordDensity(data.keyword_density);
    renderBeforeAfter(data.ats_score, data.optimized_score, data.optimization_tips);
    renderRecruiterView(data.recruiter_view);
    renderExplainableAI(data.explanation);

    resultsSection.classList.add("visible");
    resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  // ---- ATS Score Gauge ----
  function renderATSScore(score) {
    const circumference = 2 * Math.PI * 72; // r=72
    const offset = circumference - (score / 100) * circumference;
    const gaugeFill = document.getElementById("gauge-fill");
    const scoreValue = document.getElementById("score-value");
    const scoreInfo = document.getElementById("score-info");
    const scoreBadge = document.getElementById("score-badge");

    // Color based on score
    let color, badgeClass, badgeText, infoText;
    if (score < 40) {
      color = "#ef4444";
      badgeClass = "score-low";
      badgeText = "Needs Improvement";
      infoText =
        "Your resume has low alignment with this job description. Significant tailoring is needed to pass ATS screening.";
    } else if (score < 70) {
      color = "#f59e0b";
      badgeClass = "score-mid";
      badgeText = "Good Match";
      infoText =
        "Your resume aligns reasonably well. With targeted keyword optimization, you can significantly improve your chances.";
    } else {
      color = "#10b981";
      badgeClass = "score-high";
      badgeText = "Excellent Match";
      infoText =
        "Your resume is strongly aligned with this job description. You have a high chance of passing ATS screening.";
    }

    gaugeFill.style.strokeDasharray = circumference;
    gaugeFill.style.strokeDashoffset = circumference;
    gaugeFill.style.stroke = color;

    // Trigger animation
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        gaugeFill.style.strokeDashoffset = offset;
      });
    });

    // Animate number
    animateNumber(scoreValue, 0, score, 1200);

    scoreInfo.textContent = infoText;
    scoreBadge.textContent = badgeText;
    scoreBadge.className = "score-badge " + badgeClass;
  }

  function animateNumber(el, start, end, duration) {
    const startTime = performance.now();
    function update(now) {
      const elapsed = now - startTime;
      const progress = Math.min(elapsed / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3); // ease-out cubic
      el.textContent = Math.round(start + (end - start) * eased);
      if (progress < 1) requestAnimationFrame(update);
    }
    requestAnimationFrame(update);
  }

  // ---- Skill Gap ----
  function renderSkillGap(skills) {
    const container = document.getElementById("skill-gap-content");
    const matchBar = document.getElementById("skill-match-fill");

    let html = "";
    // Matched
    html += `<div class="skill-label">✅ Matched Skills (${skills.matched.length})</div>`;
    html += `<div class="skill-tags">`;
    skills.matched.forEach((s) => {
      html += `<span class="skill-tag matched">${escapeHtml(s)}</span>`;
    });
    if (!skills.matched.length) html += `<span class="skill-tag matched" style="opacity:.5">None found</span>`;
    html += `</div>`;

    // Missing
    html += `<div class="skill-label">❌ Missing Skills (${skills.missing.length})</div>`;
    html += `<div class="skill-tags">`;
    skills.missing.slice(0, 20).forEach((s) => {
      html += `<span class="skill-tag missing">${escapeHtml(s)}</span>`;
    });
    if (!skills.missing.length) html += `<span class="skill-tag matched">All covered!</span>`;
    html += `</div>`;

    // Extra
    if (skills.extra && skills.extra.length) {
      html += `<div class="skill-label">💡 Bonus Skills (${skills.extra.length})</div>`;
      html += `<div class="skill-tags">`;
      skills.extra.forEach((s) => {
        html += `<span class="skill-tag extra">${escapeHtml(s)}</span>`;
      });
      html += `</div>`;
    }

    container.innerHTML = html;
    setTimeout(() => {
      matchBar.style.width = skills.match_percent + "%";
    }, 200);
    document.getElementById("skill-match-pct").textContent =
      skills.match_percent + "% match";
  }

  // ---- Section Scores ----
  function renderSectionScores(sections) {
    const names = ["skills", "experience", "projects", "education"];
    const colors = {
      skills: "bar-skills",
      experience: "bar-experience",
      projects: "bar-projects",
      education: "bar-education",
    };
    names.forEach((name) => {
      const fill = document.getElementById(`bar-${name}`);
      const val = document.getElementById(`val-${name}`);
      const score = sections[name] || 0;
      val.textContent = score.toFixed(1) + "%";
      setTimeout(() => {
        fill.style.width = score + "%";
      }, 300);
    });
  }

  // ---- Keyword Density ----
  function renderKeywordDensity(keywords) {
    const tbody = document.getElementById("kw-tbody");
    let html = "";
    keywords.forEach((kw) => {
      const statusClass =
        kw.status === "optimal"
          ? "kw-optimal"
          : kw.status === "underused"
          ? "kw-underused"
          : "kw-missing";
      const statusIcon =
        kw.status === "optimal"
          ? "✅"
          : kw.status === "underused"
          ? "⚠️"
          : "❌";
      html += `<tr>
        <td>${escapeHtml(kw.keyword)}</td>
        <td>${kw.resume_count}</td>
        <td><span class="kw-status ${statusClass}">${statusIcon} ${kw.status}</span></td>
      </tr>`;
    });
    tbody.innerHTML = html;
  }

  // ---- Before / After ----
  function renderBeforeAfter(before, after, tips) {
    const beforeEl = document.getElementById("ba-before");
    const afterEl = document.getElementById("ba-after");
    animateNumber(beforeEl, 0, before, 1000);
    animateNumber(afterEl, 0, after, 1400);

    const tipsList = document.getElementById("tips-list");
    let html = "";
    tips.forEach((tip) => {
      html += `<li><span class="tip-icon">💡</span>${escapeHtml(tip)}</li>`;
    });
    tipsList.innerHTML = html;
  }

  // ---- Recruiter View ----
  function renderRecruiterView(rv) {
    document.getElementById("recruiter-verdict").textContent = rv.decision;
    document.getElementById("recruiter-band").textContent = rv.band;

    // Color the verdict
    const verdictEl = document.getElementById("recruiter-verdict");
    if (rv.decision.includes("Reject")) verdictEl.style.color = "#ef4444";
    else if (rv.decision.includes("Maybe")) verdictEl.style.color = "#f59e0b";
    else verdictEl.style.color = "#10b981";

    const flagList = document.getElementById("red-flags");
    flagList.innerHTML = rv.red_flags
      .map((f) => `<li>🚩 ${escapeHtml(f)}</li>`)
      .join("");

    const strengthList = document.getElementById("strengths");
    strengthList.innerHTML = rv.strengths
      .map((s) => `<li>✨ ${escapeHtml(s)}</li>`)
      .join("");
  }

  // ---- Explainable AI ----
  function renderExplainableAI(factors) {
    const container = document.getElementById("explainer-list");
    let html = "";
    factors.forEach((f, i) => {
      html += `
        <div class="explainer-item" data-idx="${i}">
          <div class="explainer-header">
            <span class="factor-name">📐 ${escapeHtml(f.factor)}</span>
            <div class="factor-meta">
              <span class="factor-weight">Weight: ${f.weight}</span>
              <span class="factor-value">${f.value}</span>
              <span class="explainer-toggle">▼</span>
            </div>
          </div>
          <div class="explainer-body">
            <p>${escapeHtml(f.explanation)}</p>
          </div>
        </div>`;
    });
    container.innerHTML = html;

    // Accordion behaviour
    container.querySelectorAll(".explainer-item").forEach((item) => {
      item.addEventListener("click", () => {
        item.classList.toggle("open");
      });
    });
  }

  // ---- Helpers ----
  function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
  }
});
