/* SECTION SWITCH */
function showSection(id) {
  document.querySelectorAll('.card').forEach(section => {
    section.classList.remove('active');
  });

  const targetSection = document.getElementById(id);
  if (targetSection) {
    targetSection.classList.add('active');

    // Metrics panel: show on deduction page, hide on others (panel lives inside #deduction)
    const metricsPanel = document.querySelector('.metrics-panel');
    if (metricsPanel) {
      metricsPanel.style.display = id === 'deduction' ? 'block' : 'none';
    }
    
    // Clear login form when login section is shown
    if (id === 'login' && typeof clearLoginForm === 'function') {
      clearLoginForm();
    }

    // Update URL hash without scrolling
    if (history.pushState) {
      history.pushState(null, null, '#' + id);
    }
    
    // Smooth scroll to the top of the target section
    setTimeout(() => {
      scrollToSectionTop(targetSection);
    }, 50); // Small delay to ensure DOM is updated
  }
}

/* Smooth scroll to the top of a section */
function scrollToSectionTop(element) {
  if (!element) return;
  
  // Get navbar height for offset
  const navbar = document.querySelector('.navbar');
  const navbarHeight = navbar ? navbar.offsetHeight : 0;
  
  // Get the container element that holds the sections
  const container = element.closest('.container') || document.querySelector('.container');
  
  if (container) {
    // Get container's position relative to document
    const containerRect = container.getBoundingClientRect();
    const containerTop = containerRect.top + window.pageYOffset;
    
    // Calculate target position: container top - navbar offset
    // Add a small padding (10px) for better visual spacing
    const targetPosition = containerTop - navbarHeight - 10;
    
    // Smooth scroll to target
    window.scrollTo({
      top: Math.max(0, targetPosition), // Ensure we don't scroll to negative position
      behavior: 'smooth'
    });
  } else {
    // Fallback: scroll to element itself
    scrollToElementTop(element);
  }
}

/* MOBILE MENU TOGGLE */
function toggleMobileMenu() {
  const mobileMenu = document.getElementById('mobileNavMenu');
  const hamburgerBtn = document.querySelector('.hamburger-btn');
  
  if (mobileMenu && hamburgerBtn) {
    mobileMenu.classList.toggle('active');
    hamburgerBtn.classList.toggle('active');
  }
}

function closeMobileMenu() {
  const mobileMenu = document.getElementById('mobileNavMenu');
  const hamburgerBtn = document.querySelector('.hamburger-btn');
  
  if (mobileMenu && hamburgerBtn) {
    mobileMenu.classList.remove('active');
    hamburgerBtn.classList.remove('active');
  }
}

/* Close mobile menu when clicking outside */
document.addEventListener('click', function(event) {
  const mobileMenu = document.getElementById('mobileNavMenu');
  const hamburgerBtn = document.querySelector('.hamburger-btn');
  const navbar = document.querySelector('.navbar');
  
  if (mobileMenu && hamburgerBtn && navbar) {
    const isClickInsideNav = navbar.contains(event.target);
    const isMenuOpen = mobileMenu.classList.contains('active');
    
    if (isMenuOpen && !isClickInsideNav) {
      closeMobileMenu();
    }
  }
});

/* Close mobile menu when window is resized to desktop size */
window.addEventListener('resize', function() {
  if (window.innerWidth > 768) {
    closeMobileMenu();
  }
});

/* ==================== SMOOTH SCROLLING ==================== */

/* Enable smooth scrolling for all anchor links and buttons */
document.addEventListener('DOMContentLoaded', function() {
  // Get all anchor links on the page
  const anchorLinks = document.querySelectorAll('a[href^="#"]');
  
  anchorLinks.forEach(link => {
    link.addEventListener('click', function(e) {
      const href = this.getAttribute('href');
      
      // Skip if href is just "#" or empty
      if (href === '#' || href === '') {
        return;
      }
      
      const targetId = href.substring(1); // Remove the #
      const targetElement = document.getElementById(targetId);
      
      if (targetElement) {
        e.preventDefault();
        
        // If it's a section (card), use showSection which handles scrolling
        if (targetElement.classList.contains('card')) {
          showSection(targetId);
        } else {
          // For other elements, scroll to top of the element
          scrollToElementTop(targetElement);
        }
      }
    });
  });
  
  // Handle buttons with onclick that call showSection
  const sectionButtons = document.querySelectorAll('button[onclick*="showSection"], .nav-item[onclick*="showSection"]');
  sectionButtons.forEach(button => {
    // Remove existing onclick and add event listener
    const onclickAttr = button.getAttribute('onclick');
    if (onclickAttr && onclickAttr.includes('showSection')) {
      button.removeAttribute('onclick');
      button.addEventListener('click', function(e) {
        // Extract section ID from the onclick string
        const match = onclickAttr.match(/showSection\(['"]([^'"]+)['"]\)/);
        if (match && match[1]) {
          e.preventDefault();
          showSection(match[1]);
        }
      });
    }
  });
});

/* Smooth scroll to the top of any element */
function scrollToElementTop(element) {
  if (!element) return;
  
  // Get navbar height for offset
  const navbar = document.querySelector('.navbar');
  const navbarHeight = navbar ? navbar.offsetHeight : 0;
  
  // Get element's position relative to document
  const elementTop = element.getBoundingClientRect().top + window.pageYOffset;
  
  // Calculate target position with navbar offset
  const targetPosition = elementTop - navbarHeight;
  
  // Smooth scroll to target
  window.scrollTo({
    top: Math.max(0, targetPosition), // Ensure we don't scroll to negative position
    behavior: 'smooth'
  });
}

/* Smooth scrolling for programmatic scrolls */
function smoothScrollTo(element, offset = 0) {
  if (typeof element === 'string') {
    element = document.getElementById(element);
  }
  
  if (element) {
    // If it's a section card, use scrollToSectionTop
    if (element.classList.contains('card')) {
      scrollToSectionTop(element);
    } else {
      const navbar = document.querySelector('.navbar');
      const navbarHeight = navbar ? navbar.offsetHeight : 0;
      const targetPosition = element.getBoundingClientRect().top + window.pageYOffset - navbarHeight - offset;
      
      window.scrollTo({
        top: Math.max(0, targetPosition),
        behavior: 'smooth'
      });
    }
  }
}

/* Smooth scroll to top function */
function smoothScrollToTop() {
  window.scrollTo({
    top: 0,
    behavior: 'smooth'
  });
}

/* Smooth scroll to bottom function */
function smoothScrollToBottom() {
  window.scrollTo({
    top: document.documentElement.scrollHeight,
    behavior: 'smooth'
  });
}

/* Override default scroll behavior for better compatibility */
if ('scrollBehavior' in document.documentElement.style === false) {
  // Fallback for browsers that don't support smooth scroll
  const originalScrollTo = window.scrollTo;
  window.scrollTo = function(options) {
    if (options && options.behavior === 'smooth') {
      const start = window.pageYOffset;
      const target = options.top || 0;
      const distance = target - start;
      const duration = 500; // milliseconds
      let startTime = null;
      
      function animation(currentTime) {
        if (startTime === null) startTime = currentTime;
        const timeElapsed = currentTime - startTime;
        const progress = Math.min(timeElapsed / duration, 1);
        
        // Easing function (ease-in-out)
        const ease = progress < 0.5
          ? 2 * progress * progress
          : 1 - Math.pow(-2 * progress + 2, 2) / 2;
        
        window.scrollTo(0, start + distance * ease);
        
        if (timeElapsed < duration) {
          requestAnimationFrame(animation);
        }
      }
      
      requestAnimationFrame(animation);
    } else {
      originalScrollTo.apply(this, arguments);
    }
  };
}

/* Handle hash navigation on page load */
window.addEventListener('DOMContentLoaded', function() {
  const hash = window.location.hash.substring(1); // Remove the #
  if (hash && ['home', 'add', 'recognize', 'deduction'].includes(hash)) {
    showSection(hash);
  } else {
    // Default to home if no hash
    showSection('home');
  }
});

/* Handle hash changes */
window.addEventListener('hashchange', function() {
  const hash = window.location.hash.substring(1);
  if (hash && ['home', 'add', 'recognize', 'deduction'].includes(hash)) {
    showSection(hash);
  }
});

/* ---------------- ADD USER AUTO-CAPTURE ---------------- */
let addStream = null;
let addCount = 0;
let addInterval = null;
const MAX_IMAGES = 30;

async function startAddUser() {
  const name = document.getElementById("userName").value.trim();
  if (!name) {
    alert("Enter a name first");
    return;
  }

  // Stop any existing capture
  if (addInterval) {
    clearInterval(addInterval);
    addInterval = null;
  }

  // Stop any existing stream
  if (addStream) {
    addStream.getTracks().forEach(t => t.stop());
    addStream = null;
  }

  addCount = 0;
  document.getElementById("addStatus").innerText = "Starting capture. Hold still...";

  const video = document.getElementById("addVideo");
  const box = document.getElementById("videoBox");

  box.classList.add("waiting");

  addStream = await navigator.mediaDevices.getUserMedia({ video: true });
  video.srcObject = addStream;

  await video.play();

  // CAMERA IS ON
  video.classList.remove("camera-off");
  box.classList.remove("waiting");
  box.classList.add("active");

  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");

  addInterval = setInterval(() => {
    if (!video || !video.videoWidth) return;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0);

    const frame = canvas.toDataURL("image/jpeg", 0.9);

    const formData = new FormData();
    formData.append("name", name);
    formData.append("frame", frame);

    fetch("/add_user_frame", {
      method: "POST",
      body: formData
    })
    .then(res => res.json())
    .then(data => {
      if (data.saved) {
        addCount++;
        document.getElementById("addStatus").innerText =
          `Captured ${addCount}/${MAX_IMAGES}`;

        // Show finalize button after 20+ images
        if (addCount >= 20) {
          const finalizeBtn = document.getElementById("finalizeBtn");
          if (finalizeBtn) {
            finalizeBtn.style.display = "block";
          }
        }

        if (addCount >= MAX_IMAGES) {
          stopAddUser();
        }
      }
    })
    .catch(err => {
      console.error("Capture error:", err);
    });

  }, 300); // ~3 frames/sec (similar to Colab)
}

function stopAddUser() {
  if (addInterval) {
    clearInterval(addInterval);
    addInterval = null;
  }

  const video = document.getElementById("addVideo");
  const box = document.getElementById("videoBox");

  if (addStream) {
    addStream.getTracks().forEach(t => t.stop());
    addStream = null;
  }

  if (video) {
    video.srcObject = null;
  }

  // CAMERA OFF
  video.classList.add("camera-off");
  box.classList.remove("active");
  box.classList.add("waiting");

  document.getElementById("addStatus").innerText =
    "Capture complete. 30 images saved. Click 'Finalize User & Train Model' to update the recognition model.";

  // Show the finalize button after capture completes
  const finalizeBtn = document.getElementById("finalizeBtn");
  if (finalizeBtn) {
    finalizeBtn.style.display = "block";
  }

  // Clear the input field after successful capture
  const userNameInput = document.getElementById("userName");
  if (userNameInput) {
    userNameInput.value = "";
  }

  alert("Capture complete. Camera stopped.");
}

/* ---------------- REBUILD EMBEDDINGS ---------------- */
function rebuildEmbeddings() {
  const status = document.getElementById("trainStatus");
  const finalizeBtn = document.getElementById("finalizeBtn");
  
  if (finalizeBtn) {
    finalizeBtn.disabled = true;
    finalizeBtn.innerText = "Training...";
  }
  
  status.innerText = "Training model... please wait ⏳";

  fetch("/rebuild_embeddings", {
    method: "POST"
  })
  .then(res => res.json())
  .then(data => {
    if (data.status === "success") {
      status.innerText =
        "Model updated successfully ✅\nUsers: " + data.users.join(", ");
      if (finalizeBtn) {
        finalizeBtn.style.display = "none";
      }
      // Clear input field after successful rebuild
      const userNameInput = document.getElementById("userName");
      if (userNameInput) {
        userNameInput.value = "";
      }
      // Clear status message after a delay
      setTimeout(() => {
        document.getElementById("addStatus").innerText = "";
      }, 3000);
    } else {
      status.innerText = "Training failed ❌: " + (data.message || "Unknown error");
      if (finalizeBtn) {
        finalizeBtn.disabled = false;
        finalizeBtn.innerText = "Finalize User & Train Model";
      }
    }
  })
  .catch(err => {
    status.innerText = "Server error ❌: " + err.message;
    if (finalizeBtn) {
      finalizeBtn.disabled = false;
      finalizeBtn.innerText = "Finalize User & Train Model";
    }
  });
}

/* CAPTURE FRAME (for recognition) */
function captureFrame(video) {
  const canvas = document.createElement("canvas");
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext("2d").drawImage(video, 0, 0);
  return canvas.toDataURL("image/jpeg", 0.9);
}

/* ---------------- RECOGNITION CAMERA ---------------- */
let videoStream = null;
let video = null;
let capturedFrames = [];
let captureTimer = null;

/* ---------------- START RECOGNITION CAMERA ---------------- */
async function startCamera() {
  video = document.getElementById("videoRec");
  const box = document.getElementById("videoRecBox");
  if (!video) return;

  // Stop any existing stream first
  stopCamera();

  if (box) box.classList.add("waiting");

  videoStream = await navigator.mediaDevices.getUserMedia({ video: true });
  video.srcObject = videoStream;
  await video.play();

  // CAMERA IS ON
  video.classList.remove("camera-off");
  if (box) {
    box.classList.remove("waiting");
    box.classList.add("active");
  }
}

/* ---------------- STOP RECOGNITION CAMERA (CRITICAL FIX) ---------------- */
function stopCamera() {
  const box = document.getElementById("videoRecBox");

  if (videoStream) {
    videoStream.getTracks().forEach(track => track.stop());
    videoStream = null;
  }

  if (video) {
    video.srcObject = null;
    // CAMERA OFF
    video.classList.add("camera-off");
  }

  if (box) {
    box.classList.remove("active");
    box.classList.add("waiting");
  }

  if (captureTimer) {
    clearInterval(captureTimer);
    captureTimer = null;
  }
}

/* ---------------- CAPTURE MULTIPLE FRAMES ---------------- */
/* Colab equivalent: RUN_SECONDS = 3, CAPTURE_INTERVAL = 0.18 */
const RUN_SECONDS = 3;
const CAPTURE_INTERVAL = 180; // 180ms = 0.18 seconds

function setRecognizeOverlay(text, visible) {
  const overlay = document.getElementById("recognizeResultOverlay");
  const textEl = document.getElementById("recognizeResultText");
  const box = document.getElementById("videoRecBox");
  if (!overlay || !textEl) return;
  textEl.textContent = text || "";
  overlay.style.display = visible ? "flex" : "none";
  if (box) {
    if (visible) box.classList.add("showing-result-overlay");
    else box.classList.remove("showing-result-overlay");
  }
}

function captureFrames() {
  capturedFrames = [];

  if (!video || !video.videoWidth) {
    setRecognizeOverlay("Camera not ready", true);
    return;
  }

  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");

  // Capture frame every 180ms (matches Colab CAPTURE_INTERVAL)
  captureTimer = setInterval(() => {
    if (!video || !video.videoWidth) return;
    
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0);

    capturedFrames.push(
      canvas.toDataURL("image/jpeg", 0.9)
    );
  }, CAPTURE_INTERVAL);

  // Stop after 3 seconds
  setTimeout(() => {
    clearInterval(captureTimer);
    captureTimer = null;
    stopCamera();          // ✅ camera turns OFF here
    sendFrames();
  }, RUN_SECONDS * 1000);
}

/* ---------------- SEND FRAMES TO BACKEND ---------------- */
function sendFrames() {
  const outputImg = document.getElementById("outputImg");

  if (capturedFrames.length === 0) {
    setRecognizeOverlay("No frames captured", true);
    return;
  }

  setRecognizeOverlay("Processing...", true);

  fetch("/recognize", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(capturedFrames)
  })
  .then(res => res.json())
  .then(data => {
    // Show result in the capture-area overlay (name only)
    const name = data.result && data.result !== "Unknown" ? data.result : (data.result || "Unknown");
    let message = "Result: " + name;
    if (data.already_in_attendance) {
      message += " — Already in today's attendance";
    }
    setRecognizeOverlay(message, true);

    // Display annotated image with bounding box below
    if (outputImg) {
      outputImg.style.width = "100%";
      outputImg.style.borderRadius = "10px";
      outputImg.style.marginTop = "10px";
      if (data.image_url) {
        outputImg.src = data.image_url + "?t=" + new Date().getTime();
        outputImg.style.display = "block";
      } else if (data.image) {
        outputImg.src = data.image;
        outputImg.style.display = "block";
      }
    }
  })
  .catch(err => {
    setRecognizeOverlay("Recognition failed", true);
    console.error(err);
  });
}

/* ---------------- TODAY'S ATTENDANCE (Face Recognition) — popup + download ---------------- */
let attendanceTodayData = { date: "", entries: [] };

function openAttendanceModal() {
  const modal = document.getElementById("attendanceModal");
  const dateEl = document.getElementById("attendanceModalDate");
  const tbody = document.getElementById("attendanceModalTbody");
  const emptyEl = document.getElementById("attendanceModalEmpty");
  const tableWrap = document.getElementById("attendanceModalTableWrap");
  if (!modal) return;

  modal.classList.add("active");
  modal.setAttribute("aria-hidden", "false");
  document.body.style.overflow = "hidden";
  if (tbody) tbody.innerHTML = "";
  if (emptyEl) emptyEl.style.display = "none";
  if (tableWrap) tableWrap.style.display = "none";

  fetch("/api/attendance/today")
    .then(res => res.json())
    .then(data => {
      attendanceTodayData = { date: data.date || "", entries: data.entries || [] };
      if (dateEl) dateEl.textContent = attendanceTodayData.date ? `Date: ${attendanceTodayData.date}` : "";
      const entries = attendanceTodayData.entries;
      if (entries.length === 0) {
        if (emptyEl) emptyEl.style.display = "block";
      } else {
        if (tableWrap) tableWrap.style.display = "block";
        tbody.innerHTML = entries
          .map((e) => `<tr><td>${escapeHtml(e.name)}</td><td>${escapeHtml(e.time)}</td></tr>`)
          .join("");
      }
    })
    .catch(() => {
      attendanceTodayData = { date: "", entries: [] };
      if (emptyEl) emptyEl.style.display = "block";
    });
}

function closeAttendanceModal() {
  const modal = document.getElementById("attendanceModal");
  if (modal) {
    modal.classList.remove("active");
    modal.setAttribute("aria-hidden", "true");
    document.body.style.overflow = "";
  }
}

document.addEventListener("keydown", function (e) {
  if (e.key === "Escape") {
    closeAttendanceModal();
  }
});

function downloadAttendanceSheet() {
  const entries = attendanceTodayData.entries;
  const date = attendanceTodayData.date || new Date().toISOString().slice(0, 10).split("-").reverse().join("-");
  const header = "Name,Time\n";
  const rows = entries.map((e) => `${escapeCsv(e.name)},${escapeCsv(e.time)}`).join("\n");
  const csv = header + rows;
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `Attendance_${date.replace(/\//g, "-")}.csv`;
  a.click();
  URL.revokeObjectURL(url);
}

function escapeCsv(str) {
  if (str == null) return "";
  const s = String(str);
  if (/[",\n\r]/.test(s)) return `"${s.replace(/"/g, '""')}"`;
  return s;
}

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

/* ---------------- MAIN RECOGNITION BUTTON ---------------- */
async function runRecognition() {
  setRecognizeOverlay("Starting camera...", true);

  await startCamera();

  setRecognizeOverlay("Capturing frames for 3 seconds...", true);
  captureFrames(); // 3 seconds, every 180ms
}

/* START RECOGNITION - Keep for backward compatibility */
async function startRecognition() {
  return runRecognition();
}

/* ---------------- FACE DETECTION CAMERA ---------------- */
let detectStream = null;
let detectVideo = null;

/* ---------------- START DETECTION CAMERA ---------------- */
async function startDetectionCamera() {
  detectVideo = document.getElementById("videoDetect");
  const box = document.getElementById("videoDetectBox");
  if (!detectVideo) return;

  // Stop any existing stream first
  stopDetectionCamera();

  if (box) box.classList.add("waiting");

  detectStream = await navigator.mediaDevices.getUserMedia({ video: true });
  detectVideo.srcObject = detectStream;
  await detectVideo.play();

  // CAMERA IS ON
  detectVideo.classList.remove("camera-off");
  if (box) {
    box.classList.remove("waiting");
    box.classList.add("active");
  }
}

/* ---------------- STOP DETECTION CAMERA ---------------- */
function stopDetectionCamera() {
  const box = document.getElementById("videoDetectBox");

  if (detectStream) {
    detectStream.getTracks().forEach(track => track.stop());
    detectStream = null;
  }

  if (detectVideo) {
    detectVideo.srcObject = null;
    // CAMERA OFF
    detectVideo.classList.add("camera-off");
  }

  if (box) {
    box.classList.remove("active");
    box.classList.add("waiting");
  }
}

/* ---------------- RUN FACE DETECTION ---------------- */
async function runDeduction() {
  const resultEl = document.getElementById("detectResult");
  const outputImg = document.getElementById("detectOutputImg");
  const metricsPanel = document.querySelector(".metrics-panel");
  
  resultEl.innerText = "Starting camera...";
  
  await startDetectionCamera();
  
  // Wait a moment for camera to stabilize
  await new Promise(resolve => setTimeout(resolve, 500));
  
  if (!detectVideo || !detectVideo.videoWidth) {
    resultEl.innerText = "Camera not ready";
    return;
  }

  resultEl.innerText = "Detecting face...";

  // Capture single frame
  const canvas = document.createElement("canvas");
  canvas.width = detectVideo.videoWidth;
  canvas.height = detectVideo.videoHeight;
  canvas.getContext("2d").drawImage(detectVideo, 0, 0);
  const frame = canvas.toDataURL("image/jpeg", 0.9);

  // Send to backend
  fetch("/detect_face", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ frame: frame })
  })
  .then(res => {
    if (!res.ok) {
      return res.json().then(err => {
        throw new Error(err.error || "Request failed");
      });
    }
    return res.json();
  })
  .then(data => {
    if (data.error) {
      resultEl.innerText = "Error: " + data.error;
      return;
    }

    // Display result
    if (data.detected) {
      resultEl.innerText = `Face detected! Confidence: ${data.confidence.toFixed(3)}`;
    } else {
      resultEl.innerText = "No face detected";
    }

    // Display annotated image
    if (outputImg && data.image) {
      outputImg.src = data.image;
      outputImg.style.display = "block";
      outputImg.style.width = "100%";
      outputImg.style.borderRadius = "10px";
      outputImg.style.marginTop = "10px";
    }

    // Update metrics table with static evaluation numbers
    const metrics = [
      { model: "VGG-16 Model", accuracy: "99.57%", precision: "99.79%", recall: "99.94%", meanIoU: "72.71%" },
      { model: "Haar Cascade", accuracy: "78.20%", precision: "80.45%", recall: "82.45%", meanIoU: "72.90%" },
      { model: "Early CNN-style", accuracy: "86.67%", precision: "87.67%", recall: "90.50%", meanIoU: "49.35%" }
    ];
    if (typeof updateMetricsTable === "function") {
      updateMetricsTable(metrics);
    }

    // Reveal metrics panel after a detection run
    if (metricsPanel) {
      metricsPanel.style.display = "block";
      metricsPanel.classList.add("visible");

      // Smooth scroll to metrics so the user notices the comparison
      if (typeof smoothScrollTo === "function") {
        smoothScrollTo(metricsPanel, 20);
      } else {
        metricsPanel.scrollIntoView({ behavior: "smooth", block: "start" });
      }
    }

    // Stop camera after detection
    stopDetectionCamera();
  })
  .catch(err => {
    resultEl.innerText = "Detection failed: " + err.message;
    console.error(err);
    stopDetectionCamera();
  });
}

/* ---------------- METRICS TABLE HELPER ---------------- */
function updateMetricsTable(metrics) {
  const tbody = document.querySelector("#metrics-section tbody");
  if (!tbody) return;

  tbody.innerHTML = ""; // clear existing rows
  metrics.forEach(m => {
    const row = document.createElement("tr");
    row.innerHTML = `
      <td>${m.model}</td>
      <td>${m.accuracy}</td>
      <td>${m.precision}</td>
      <td>${m.recall}</td>
      <td>${m.meanIoU}</td>
    `;
    tbody.appendChild(row);
  });
}

/* ---------------- RUN FACE DETECTION COMPARISON ---------------- */
async function runDetectionComparison() {
  const resHaar = document.getElementById("detectResultHaar");
  const resEarly = document.getElementById("detectResultEarly");
  const resFT = document.getElementById("detectResultFacetracker");
  const imgHaar = document.getElementById("detectOutputImgHaar");
  const imgEarly = document.getElementById("detectOutputImgEarly");
  const imgFT = document.getElementById("detectOutputImgFacetracker");

  if (resHaar) resHaar.innerText = "Starting camera...";
  if (resEarly) resEarly.innerText = "";
  if (resFT) resFT.innerText = "";

  await startDetectionCamera();

  // Small delay for camera to stabilize
  await new Promise(resolve => setTimeout(resolve, 500));

  if (!detectVideo || !detectVideo.videoWidth) {
    if (resHaar) resHaar.innerText = "Camera not ready";
    stopDetectionCamera();
    return;
  }

  if (resHaar) resHaar.innerText = "Capturing frame and running comparison...";

  // Capture single frame
  const canvas = document.createElement("canvas");
  canvas.width = detectVideo.videoWidth;
  canvas.height = detectVideo.videoHeight;
  canvas.getContext("2d").drawImage(detectVideo, 0, 0);
  const frame = canvas.toDataURL("image/jpeg", 0.9);

  fetch("/detect_face_compare", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ frame: frame })
  })
  .then(res => res.json())
  .then(data => {
    // Haarcascade
    if (data.haar) {
      if (resHaar) {
        if (data.haar.detected) {
          const c = typeof data.haar.confidence === "number"
            ? data.haar.confidence.toFixed(3)
            : "N/A";
          resHaar.innerText = `Haarcascade: Face detected (conf: ${c})`;
        } else {
          resHaar.innerText = "Haarcascade: No face detected";
        }
      }
      if (imgHaar && data.haar.image) {
        imgHaar.src = data.haar.image;
        imgHaar.style.display = "block";
        imgHaar.style.width = "100%";
        imgHaar.style.borderRadius = "10px";
        imgHaar.style.marginTop = "10px";
      }
    }

    // Early CNN
    if (data.early_cnn) {
      if (resEarly) {
        if (!data.early_cnn.loaded) {
          resEarly.innerText = "Early CNN: model not loaded on server";
        } else if (data.early_cnn.detected) {
          const c = typeof data.early_cnn.confidence === "number"
            ? data.early_cnn.confidence.toFixed(3)
            : "N/A";
          resEarly.innerText = `Early CNN: Face detected (conf: ${c})`;
        } else {
          resEarly.innerText = "Early CNN: No face detected";
        }
      }
      if (imgEarly && data.early_cnn.image) {
        imgEarly.src = data.early_cnn.image;
        imgEarly.style.display = "block";
        imgEarly.style.width = "100%";
        imgEarly.style.borderRadius = "10px";
        imgEarly.style.marginTop = "10px";
      }
    }

    // Facetracker
    if (data.facetracker) {
      if (resFT) {
        if (!data.facetracker.loaded) {
          resFT.innerText = "Facetracker: model not loaded on server";
        } else if (data.facetracker.detected) {
          const c = typeof data.facetracker.confidence === "number"
            ? data.facetracker.confidence.toFixed(3)
            : "N/A";
          resFT.innerText = `Facetracker: Face detected (conf: ${c})`;
        } else {
          resFT.innerText = "Facetracker: No face detected";
        }
      }
      if (imgFT && data.facetracker.image) {
        imgFT.src = data.facetracker.image;
        imgFT.style.display = "block";
        imgFT.style.width = "100%";
        imgFT.style.borderRadius = "10px";
        imgFT.style.marginTop = "10px";
      }
    }

    // Stop camera when done
    stopDetectionCamera();
  })
  .catch(err => {
    if (resHaar) {
      resHaar.innerText = "Comparison failed: " + err.message;
    }
    console.error(err);
    stopDetectionCamera();
  });
}

/* AUTO START - No auto-start for add user, starts on button click */
window.onload = () => {
  // Cameras start on demand
};
