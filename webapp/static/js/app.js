/* MedReason++ · app.js */

// ── Particles ──────────────────────────────────────────────────
(function(){
  const c = document.getElementById('particles');
  if (!c) return;
  const ctx = c.getContext('2d');
  let W, H, pts = [];

  function resize() { W = c.width = window.innerWidth; H = c.height = window.innerHeight; }
  resize(); window.addEventListener('resize', resize);

  for (let i = 0; i < 70; i++) {
    pts.push({ x: Math.random()*W, y: Math.random()*H, vx: (Math.random()-.5)*.3, vy: (Math.random()-.5)*.3, r: Math.random()*1.5+.5 });
  }

  function draw() {
    ctx.clearRect(0,0,W,H);
    pts.forEach(p => {
      p.x += p.vx; p.y += p.vy;
      if (p.x < 0) p.x = W; if (p.x > W) p.x = 0;
      if (p.y < 0) p.y = H; if (p.y > H) p.y = 0;
      ctx.beginPath(); ctx.arc(p.x,p.y,p.r,0,Math.PI*2);
      ctx.fillStyle = 'rgba(99,130,246,.6)'; ctx.fill();
    });
    pts.forEach((a,i) => pts.slice(i+1).forEach(b => {
      const d = Math.hypot(a.x-b.x, a.y-b.y);
      if (d < 100) { ctx.beginPath(); ctx.moveTo(a.x,a.y); ctx.lineTo(b.x,b.y); ctx.strokeStyle=`rgba(99,130,246,${.12*(1-d/100)})`; ctx.lineWidth=.5; ctx.stroke(); }
    }));
    requestAnimationFrame(draw);
  }
  draw();
})();

// ── Navigation ─────────────────────────────────────────────────
window.addEventListener('scroll', () => {
  document.getElementById('nav')?.classList.toggle('scrolled', window.scrollY > 40);
});

const menuToggle = document.getElementById('navMenuToggle');
const primaryNav = document.getElementById('primaryNav');
if (menuToggle && primaryNav) {
  menuToggle.addEventListener('click', () => {
    const expanded = menuToggle.getAttribute('aria-expanded') === 'true';
    menuToggle.setAttribute('aria-expanded', String(!expanded));
    menuToggle.setAttribute('aria-label', expanded ? 'Open navigation menu' : 'Close navigation menu');
    primaryNav.classList.toggle('open', !expanded);
  });
  primaryNav.querySelectorAll('a').forEach(link => link.addEventListener('click', () => {
    menuToggle.setAttribute('aria-expanded', 'false');
    menuToggle.setAttribute('aria-label', 'Open navigation menu');
    primaryNav.classList.remove('open');
  }));
}

// ── Service status ─────────────────────────────────────────────
async function checkServiceStatus() {
  const badge = document.getElementById('serviceStatus');
  if (!badge) return;
  try {
    const response = await fetch('/health');
    if (!response.ok) throw new Error(`Health check returned ${response.status}`);
    badge.classList.remove('offline');
    badge.querySelector('.status-label').textContent = 'System online';
  } catch (error) {
    badge.classList.add('offline');
    badge.querySelector('.status-label').textContent = 'Service unavailable';
    console.error('Service health check failed:', error);
  }
}
checkServiceStatus();

// ── State ──────────────────────────────────────────────────────
let selFile = null, selPath = null, previewUrl = null;

// ── File ───────────────────────────────────────────────────────
const dz = document.getElementById('dropZone');
const fi = document.getElementById('fi');
const rb = document.getElementById('runBtn');

if (dz) {
  dz.addEventListener('click', event => {
    if (!event.target.closest('.drop-clear')) fi?.click();
  });
  dz.addEventListener('keydown', event => {
    if ((event.key === 'Enter' || event.key === ' ') && !event.target.closest('.drop-clear')) {
      event.preventDefault();
      fi?.click();
    }
  });
  dz.addEventListener('dragover', e => { e.preventDefault(); dz.classList.add('drag'); });
  dz.addEventListener('dragleave', () => dz.classList.remove('drag'));
  dz.addEventListener('drop', e => { e.preventDefault(); dz.classList.remove('drag'); const f=e.dataTransfer.files[0]; if(f) setFile(f); });
}
if (fi) fi.addEventListener('change', e => { if(e.target.files[0]) setFile(e.target.files[0]); });

function setFile(f) {
  if (!f.name.match(/\.(jpeg|jpg|png)$/i)) {
    showError('Please choose a JPEG or PNG image.');
    return;
  }
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(f);
  selFile = f; selPath = null;
  clearError();
  dz.innerHTML = '<img class="upload-preview" alt="Selected X-ray preview"><div class="dz-content"><div class="dz-upload-icon" aria-hidden="true"><svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><polyline points="20 6 9 17 4 12"/></svg></div><p class="dz-title"></p><p class="dz-sub"></p></div><button class="drop-clear" type="button" aria-label="Remove selected image">Remove</button>';
  dz.querySelector('.upload-preview').src = previewUrl;
  dz.querySelector('.dz-title').textContent = f.name;
  dz.querySelector('.dz-sub').textContent = `${(f.size/1024).toFixed(1)} KB · ready to analyse`;
  dz.querySelector('.drop-clear').addEventListener('click', event => {
    event.stopPropagation();
    clearSelection();
  });
  dz.classList.add('ok'); rb.disabled = false;
  document.querySelectorAll('.spill').forEach(button => button.classList.remove('sel'));
}

function clearSelection() {
  selFile = null;
  selPath = null;
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = null;
  if (fi) fi.value = '';
  dz.classList.remove('ok');
  dz.innerHTML = '<div class="dz-content"><div class="dz-upload-icon"><svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.2"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M17 8l-5-5-5 5M12 3v12"/></svg></div><p class="dz-title">Drop image or click to browse</p><p class="dz-sub">JPEG · PNG · Chest radiograph</p></div>';
  rb.disabled = true;
}

// ── Quick fill ──────────────────────────────────────────────────
function qf(t) {
  const m = {
    pn: "Right upper lobe airspace disease consistent with pneumonia given patient's history. The lungs are otherwise clear. No pleural effusions or pneumothoraces. Heart and mediastinum are of normal size.",
    nm: "The lungs are clear bilaterally. No focal consolidation, pleural effusion, or pneumothorax. Cardiomediastinal silhouette is unremarkable. No acute osseous abnormality identified.",
    cf: "There are streaky retrocardiac left lower lobe opacities; in the correct clinical setting this could represent early pneumonia. Otherwise the lungs are relatively clear. No definite pleural effusion."
  };
  document.getElementById('reportText').value = m[t]||'';
  document.getElementById('reportText').dispatchEvent(new Event('input', {bubbles:true}));
}

// ── Samples ────────────────────────────────────────────────────
async function loadSamples() {
  try {
    const response = await fetch('/sample-images');
    if (!response.ok) throw new Error(`Sample request returned ${response.status}`);
    const d = await response.json();
    const c = document.getElementById('sampleBtns'); c.innerHTML='';
    let count = 0;
    for(const [cls,paths] of Object.entries(d)) {
      paths.forEach((p,i) => {
        const b = document.createElement('button');
        b.className = 'spill ' + (cls==='NORMAL'?'nm':'pn');
        b.textContent = `${cls==='NORMAL'?'◇':'◆'} ${cls.charAt(0)+cls.slice(1).toLowerCase()} ${i+1}`;
        b.onclick = ()=> pickSample(p, b);
        c.appendChild(b);
        count += 1;
      });
    }
    if (!count) c.innerHTML = '<span class="sample-loading">No sample images found.</span>';
  } catch(e){
    document.getElementById('sampleBtns').innerHTML = '<span class="sample-loading">Samples are unavailable right now.</span>';
    console.error('Could not load sample images:', e);
  }
}

function pickSample(path, btn) {
  selPath=path; selFile=null;
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = null;
  if (fi) fi.value = '';
  clearError();
  document.querySelectorAll('.spill').forEach(b=>b.classList.remove('sel'));
  btn.classList.add('sel');
  const name=path.split(/[\\/]/).pop(), isPn=path.includes('PNEUMONIA');
  dz.innerHTML=`<div class="sample-preview-mark" aria-hidden="true">${isPn?'P':'N'}</div><div class="dz-content"><div class="dz-upload-icon"><svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><polyline points="20 6 9 17 4 12 12 12"/></svg></div><p class="dz-title"></p><p class="dz-sub"></p></div><button class="drop-clear" type="button" aria-label="Remove selected image">Remove</button>`;
  dz.querySelector('.dz-title').textContent = isPn ? 'Pneumonia sample' : 'Normal sample';
  dz.querySelector('.dz-sub').textContent = name;
  dz.querySelector('.drop-clear').addEventListener('click', event => {
    event.stopPropagation();
    clearSelection();
    document.querySelectorAll('.spill').forEach(button => button.classList.remove('sel'));
  });
  dz.classList.add('ok'); rb.disabled=false;
}

loadSamples();

function showError(message) {
  const error = document.getElementById('analysisError');
  error.textContent = message;
  error.hidden = false;
}

function clearError() {
  const error = document.getElementById('analysisError');
  error.textContent = '';
  error.hidden = true;
}

// ── Run analysis ────────────────────────────────────────────────
async function runAnalysis() {
  if (!selFile && !selPath) {
    showError('Choose an X-ray image or select a sample before running analysis.');
    return;
  }
  clearError();
  rb.disabled = true;
  const report = document.getElementById('reportText').value.trim();
  const age  = document.getElementById('mAge').value;
  const symp = document.getElementById('mSymp').value==='1';
  const fup  = document.getElementById('mFup').value==='1';
  const meta = {has_symptoms:symp, is_followup:fup};
  if(age) meta.age_years=parseFloat(age);

  showLoad(1);
  try {
    let r;
    if(selFile) {
      const fd=new FormData();
      fd.append('file',selFile);
      fd.append('report_text',report);
      fd.append('metadata_age',age||'');
      fd.append('metadata_symptoms',symp?'1':'0');
      fd.append('metadata_followup',fup?'1':'0');
      step(2);
      const res=await fetch('/analyse',{method:'POST',body:fd});
      if(!res.ok) throw new Error(await res.text());
      r=await res.json();
    } else if(selPath) {
      step(2);
      const res=await fetch('/analyse-sample',{method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({image_path:selPath,report_text:report,metadata:meta})});
      if(!res.ok) throw new Error(await res.text());
      r=await res.json();
    }
    hideLoad(); render(r);
  } catch(e) {
    hideLoad();
    showError('Analysis could not be completed. ' + e.message);
  } finally {
    rb.disabled = !(selFile || selPath);
  }
}

// ── Render results ──────────────────────────────────────────────
function render(r) {
  const rp = document.getElementById('resPanel');
  rp.style.display='flex'; rp.style.flexDirection='column';

  const isPn = r.prediction==='PNEUMONIA';
  const pct  = Math.round((r.confidence||0)*100);

  // Prediction card
  document.getElementById('predCard').style.setProperty('--accent', isPn?'var(--red)':'var(--green)');
  const pv = document.getElementById('predVal');
  pv.textContent = r.prediction||'—';
  pv.className   = 'pred-val '+(isPn?'pneumonia':'normal');
  document.getElementById('predConf').textContent = `Confidence · ${pct}%`;

  // Gauge
  const circ=2*Math.PI*40, fill=(pct/100)*circ;
  document.getElementById('gFill').setAttribute('stroke-dasharray',`${fill} ${circ-fill}`);
  document.getElementById('gFill').style.stroke = isPn?'var(--red)':'var(--green)';
  document.getElementById('gaugeNum').textContent = pct+'%';

  // Prob bar
  const pn=Math.round((r.prob_pneumonia||0)*100), nm=Math.round((r.prob_normal||0)*100);
  document.getElementById('pbN').style.width=nm+'%';
  document.getElementById('pbP').style.width=pn+'%';
  document.getElementById('pbNmL').textContent=`NM ${nm}%`;
  document.getElementById('pbPnL').textContent=`PN ${pn}%`;

  // Activation
  document.getElementById('predAct').textContent =
    `Highest activation · ${r.activation_region||'—'} · intensity ${Math.round((r.activation_level||0)*100)}%`;

  // Modality weights
  const mc=document.getElementById('modCard');
  if(r.modality_weights && Object.keys(r.modality_weights).length) {
    mc.style.display='block';
    const cols=['#6366f1','#0ea5e9','#f59e0b','#ec4899'];
    document.getElementById('modBars').innerHTML = Object.entries(r.modality_weights).map(([k,v],i)=>
      `<div class="mod-item"><div class="mod-row"><span>${esc(k)}</span><span>${v}%</span></div>
       <div class="mod-track"><div class="mod-bar" style="width:${v}%;background:${cols[i%4]}"></div></div></div>`
    ).join('');
  } else mc.style.display='none';

  // KG
  const kgc=document.getElementById('kgResCard');
  if(r.kg_consistency) {
    kgc.style.display='block';
    const sc=r.kg_score||0, cls=sc>.1?'pos':sc<-.1?'neg':'neu';
    const con=(r.kg_consistency||'').toLowerCase();
    let h=`<div class="kg-score-row"><span class="kg-score-num ${cls}">${sc.toFixed(3)}</span><span class="kg-cons ${con}">${r.kg_consistency}</span></div>`;
    if(r.kg_supporting?.length)    h+=`<div class="kg-tags-row" style="margin-bottom:.4rem">${r.kg_supporting.map(f=>`<span class="kg-tag-s">${esc(f)}</span>`).join('')}</div>`;
    if(r.kg_contradicting?.length) h+=`<div class="kg-tags-row">${r.kg_contradicting.map(f=>`<span class="kg-tag-c">${esc(f)}</span>`).join('')}</div>`;
    document.getElementById('kgResBody').innerHTML=h;
  } else kgc.style.display='none';

  // Conflict
  const cc=document.getElementById('conflictRes');
  const sev=(r.conflict_severity||'NONE').toLowerCase();
  cc.style.display='block'; cc.className='conflict-res '+sev;
  const sicons={none:'✓',low:'△',medium:'▲',high:'⚠'};
  let ch=`<div class="cr-title">${sicons[sev]||'△'} ${sev==='none'?'No evidence conflicts detected':'Conflict detected — '+r.conflict_severity}</div>`;
  (r.conflict_list||[]).forEach(c=>{ch+=`<div class="cr-item">· ${esc(c)}</div>`;});
  cc.innerHTML=ch;

  // GradCAM
  if(r.gradcam_heatmap) document.getElementById('hmImg').src='data:image/png;base64,'+r.gradcam_heatmap;
  if(r.gradcam_overlay) document.getElementById('ovImg').src='data:image/png;base64,'+r.gradcam_overlay;

  // Scroll
  document.getElementById('analyse').scrollIntoView({behavior:'smooth', block:'start'});
  setTimeout(()=> rp.scrollIntoView({behavior:'smooth', block:'nearest'}), 300);

  // Agents
  const aw=document.getElementById('agentWrap');
  const findings=r.agent_findings||[];
  const report=r.agent_report||'';
  const isQuota=findings.some(f=>(f.finding||'').toLowerCase().includes('quota')) || (report.toLowerCase().includes('quota'));

  if(report || findings.length) {
    aw.style.display='block';
    const names={visual_analyst:'Visual Analyst',clinical_reasoner:'Clinical Reasoner',evidence_aggregator:'Evidence Aggregator'};
    const classes={visual_analyst:'a1',clinical_reasoner:'a2',evidence_aggregator:'a3'};

    if(isQuota) {
      document.getElementById('aSteps').innerHTML=`
        <div class="ao-step quota" style="grid-column:1/-1">
          <div class="ao-hdr">⏳ Reasoning Quota</div>
          <div class="ao-body">Daily API quota reached. Image classification, GradCAM++ activation maps, knowledge graph verification, conflict detection, and modality weights above are complete and fully functional — they operate independently of the reasoning API.</div>
        </div>`;
      document.getElementById('aReport').textContent='Structured clinical report will be available after the daily quota resets.\n\nAll analysis components are fully operational and shown above.';
    } else {
      document.getElementById('aSteps').innerHTML=findings.slice(0,2).map(f=>
        `<div class="ao-step ${classes[f.agent]||''}"><div class="ao-hdr">${names[f.agent]||f.agent}</div><div class="ao-body">${esc(f.finding)}</div></div>`
      ).join('');
      document.getElementById('aReport').textContent=report;
    }
    setTimeout(()=> aw.scrollIntoView({behavior:'smooth',block:'nearest'}), 600);
  } else aw.style.display='none';
}

// ── Gallery ─────────────────────────────────────────────────────
function gal(id) {
  document.querySelectorAll('.gal-pill').forEach(b=> b.classList.toggle('active', b.getAttribute('onclick').includes(`'${id}'`)));
  document.querySelectorAll('.gal-slide').forEach(p=> p.classList.toggle('active', p.id==='gal-'+id));
}

// ── Loading ─────────────────────────────────────────────────────
const lmsgs=['','Extracting image features…','Encoding report text…','Verifying knowledge graph…','Running multi-agent reasoning…'];
function showLoad(s) {
  document.getElementById('loadScr').style.display='flex';
  document.querySelectorAll('.ls-step').forEach(e=>e.classList.remove('act','done'));
  document.getElementById('lss'+s)?.classList.add('act');
  document.getElementById('lsMsg').textContent=lmsgs[s]||'Analysing…';
}
function step(n) {
  for(let i=1;i<n;i++){const e=document.getElementById('lss'+i);e?.classList.remove('act');e?.classList.add('done');}
  document.getElementById('lss'+n)?.classList.add('act');
  document.getElementById('lsMsg').textContent=lmsgs[n]||'Processing…';
}
function hideLoad() { document.getElementById('loadScr').style.display='none'; }

// ── Utils ───────────────────────────────────────────────────────
function esc(s){ return String(s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/\n/g,'<br>'); }
