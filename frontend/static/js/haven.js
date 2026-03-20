// Haven v3 — VAD audio + burst frames + Edge TTS

const S = {
  voiceOn:true,
  // Camera
  camStream:null, camRec:false, frameInt:null,
  passiveFrameInt:null, frameCount:0,
  // Audio
  micStream:null, mediaRec:null, audioCtx:null,
  analyser:null, vadInt:null, vizRaf:null,
  isSpeaking:false, silenceTimer:null,
  audioChunkCount:0,
  // STT
  sttRecog:null, sttActive:false,
  // State
  phase:1, breathPhase:'idle',
  signalHistory:[], currentAudio:null,
  permissionsGranted:{cam:false,mic:false},
  chatStarted:false,
};

const VAD_THRESHOLD   = 0.015;  // RMS threshold for speech detection
const VAD_SILENCE_MS  = 1200;   // ms of silence before stopping chunk
const PASSIVE_FRAME_S = 2000;   // passive frame every 2s
const BURST_FRAMES    = 3;      // extra frames when speaking

const SECS = ['s-hero','s-well','s-chat','s-res'];

// ── Edge TTS ──────────────────────────────────────────────
async function speak(text) {
  if (!S.voiceOn || !text?.trim()) return;
  try {
    if (S.currentAudio) { S.currentAudio.pause(); S.currentAudio = null; }
    const resp = await fetch('/tts', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({text})
    });
    if (!resp.ok) return;
    const blob  = await resp.blob();
    const url   = URL.createObjectURL(blob);
    const audio = new Audio(url);
    S.currentAudio = audio;
    const rb = document.querySelector('.remi-b,.rav');
    if (rb) rb.classList.add('spk');
    audio.onended = () => {
      if (rb) rb.classList.remove('spk');
      URL.revokeObjectURL(url);
      S.currentAudio = null;
    };
    await audio.play();
  } catch(e) { console.warn('[TTS]', e); }
}

function toggleVoice() {
  S.voiceOn = !S.voiceOn;
  if (!S.voiceOn && S.currentAudio) { S.currentAudio.pause(); S.currentAudio = null; }
  const dot = document.querySelector('.vdot');
  const lbl = document.getElementById('vlabel');
  if (dot) dot.classList.toggle('on', S.voiceOn);
  if (lbl) lbl.textContent = S.voiceOn ? 'Voice on' : 'Voice off';
}

// ── Permissions ───────────────────────────────────────────
async function requestPermissions() {
  const overlay = document.getElementById('perm-overlay');
  if (overlay) overlay.style.display = 'flex';

  const results = { cam: false, mic: false };

  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video:true, audio:true });
    results.cam = true;
    results.mic = true;
    // Keep streams ready
    stream.getVideoTracks().forEach(t => t.stop());
    stream.getAudioTracks().forEach(t => t.stop());
  } catch(e) {
    // Try individually
    try {
      const ms = await navigator.mediaDevices.getUserMedia({ audio:true });
      results.mic = true;
      ms.getTracks().forEach(t => t.stop());
    } catch(_) {}
    try {
      const cs = await navigator.mediaDevices.getUserMedia({ video:true });
      results.cam = true;
      cs.getTracks().forEach(t => t.stop());
    } catch(_) {}
  }

  S.permissionsGranted = results;
  if (overlay) overlay.style.display = 'none';
  return results;
}

// ── VAD + Audio recording ─────────────────────────────────
async function startAudioCapture() {
  if (!S.permissionsGranted.mic) return;
  try {
    S.micStream  = await navigator.mediaDevices.getUserMedia({ audio: {
      channelCount:1, sampleRate:16000, echoCancellation:true, noiseSuppression:true
    }});
    S.audioCtx   = new AudioContext({ sampleRate: 16000 });
    S.analyser   = S.audioCtx.createAnalyser();
    S.analyser.fftSize = 512;
    const src    = S.audioCtx.createMediaStreamSource(S.micStream);
    src.connect(S.analyser);

    // Start VAD loop
    startVAD();
    // Start waveform visualizer
    drawWave();
    updateMicUI(true);
    console.log('[Audio] VAD started');
  } catch(e) {
    console.warn('[Audio] Mic error:', e);
  }
}

function stopAudioCapture() {
  stopVAD();
  if (S.vizRaf) cancelAnimationFrame(S.vizRaf);
  if (S.micStream) { S.micStream.getTracks().forEach(t => t.stop()); S.micStream = null; }
  if (S.audioCtx)  { S.audioCtx.close(); S.audioCtx = null; }
  updateMicUI(false);
}

function startVAD() {
  const buf = new Float32Array(S.analyser.fftSize);

  S.vadInt = setInterval(() => {
    if (!S.analyser) return;
    S.analyser.getFloatTimeDomainData(buf);
    const rms = Math.sqrt(buf.reduce((s, v) => s + v*v, 0) / buf.length);

    if (rms > VAD_THRESHOLD) {
      // Speech detected
      if (!S.isSpeaking) {
        S.isSpeaking = true;
        startChunk();
        burstFrames();  // capture extra frames when speaking
      }
      clearTimeout(S.silenceTimer);
      S.silenceTimer = setTimeout(() => {
        // Silence detected — end chunk
        S.isSpeaking = false;
        stopChunk();
      }, VAD_SILENCE_MS);
    }
  }, 100);
}

function stopVAD() {
  clearInterval(S.vadInt);
  clearTimeout(S.silenceTimer);
  if (S.mediaRec && S.mediaRec.state === 'recording') stopChunk();
}

function startChunk() {
  if (!S.micStream || S.mediaRec?.state === 'recording') return;
  try {
    S.mediaRec = new MediaRecorder(S.micStream, { mimeType: 'audio/webm;codecs=opus' });
    const chunks = [];
    S.mediaRec.ondataavailable = e => { if (e.data.size > 0) chunks.push(e.data); };
    S.mediaRec.onstop = async () => {
      if (chunks.length === 0) return;
      const blob   = new Blob(chunks, { type: 'audio/webm' });
      const reader = new FileReader();
      reader.onloadend = async () => {
        const b64 = reader.result;
        try {
          await fetch('/remi/audio_chunk', {
            method:'POST', headers:{'Content-Type':'application/json'},
            body: JSON.stringify({ chunk: b64 })
          });
          S.audioChunkCount++;
          updateAudioIndicator();
        } catch(e) { console.warn('[Audio] Chunk send error', e); }
      };
      reader.readAsDataURL(blob);
    };
    S.mediaRec.start();
  } catch(e) { console.warn('[Audio] Chunk start error', e); }
}

function stopChunk() {
  try {
    if (S.mediaRec && S.mediaRec.state === 'recording') {
      S.mediaRec.stop();
    }
  } catch(e) {}
}

function updateAudioIndicator() {
  const el = document.getElementById('audio-indicator');
  if (el) el.textContent = `${S.audioChunkCount} audio segments captured`;
}

function updateMicUI(active) {
  const ind = document.getElementById('mic-active');
  if (ind) ind.style.display = active ? 'flex' : 'none';
}

// ── Webcam capture ────────────────────────────────────────
async function startVideoCapture() {
  if (!S.permissionsGranted.cam) return;
  try {
    S.camStream = await navigator.mediaDevices.getUserMedia({
      video: { width:640, height:480, facingMode:'user', frameRate:15 }, audio:false
    });
    const vid = document.getElementById('cv');
    const ov  = document.getElementById('cov');
    if (vid) vid.srcObject = S.camStream;
    if (ov)  ov.style.display = 'none';

    // Passive frame every 2s
    S.passiveFrameInt = setInterval(captureAndSendFrame, PASSIVE_FRAME_S);
    updateCamUI(true);
    console.log('[Video] Passive capture started');
  } catch(e) {
    console.warn('[Video] Camera error:', e);
  }
}

function stopVideoCapture() {
  clearInterval(S.passiveFrameInt);
  if (S.camStream) { S.camStream.getTracks().forEach(t => t.stop()); S.camStream = null; }
  const vid = document.getElementById('cv');
  const ov  = document.getElementById('cov');
  if (vid) vid.srcObject = null;
  if (ov)  ov.style.display = 'flex';
  updateCamUI(false);
}

function captureAndSendFrame() {
  const vid = document.getElementById('cv');
  if (!vid || !S.camStream) return;
  const canvas = document.createElement('canvas');
  canvas.width  = vid.videoWidth  || 320;
  canvas.height = vid.videoHeight || 240;
  canvas.getContext('2d').drawImage(vid, 0, 0);
  const b64 = canvas.toDataURL('image/jpeg', 0.7);

  fetch('/remi/frame', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({ frame: b64 })
  }).then(r => r.json()).then(d => {
    updateFrameIndicator(d.frame_count);
  }).catch(() => {});
}

function burstFrames() {
  // Capture BURST_FRAMES extra frames when speech starts
  for (let i = 0; i < BURST_FRAMES; i++) {
    setTimeout(captureAndSendFrame, i * 200);
  }
}

function updateFrameIndicator(count) {
  const el = document.getElementById('frame-indicator');
  if (el) el.textContent = `${count} frames captured`;
}

function updateCamUI(active) {
  const ind = document.getElementById('cam-active');
  if (ind) ind.style.display = active ? 'flex' : 'none';
}

// ── Waveform visualizer ───────────────────────────────────
function drawWave() {
  const canvas = document.getElementById('wfc');
  if (!canvas || !S.analyser) return;
  const ctx = canvas.getContext('2d');
  const buf = new Uint8Array(S.analyser.frequencyBinCount);
  function draw() {
    S.vizRaf = requestAnimationFrame(draw);
    S.analyser.getByteFrequencyData(buf);
    canvas.width = canvas.offsetWidth; canvas.height = canvas.offsetHeight;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const bw = canvas.width / buf.length;
    buf.forEach((v, i) => {
      const h = (v/255) * canvas.height * 0.88;
      // Color shifts when speaking
      const col = S.isSpeaking
        ? `rgba(64,145,108,${0.4+v/300})`
        : `rgba(100,180,140,${0.2+v/500})`;
      ctx.fillStyle = col;
      ctx.fillRect(i*bw, canvas.height-h, Math.max(bw-1,1), h);
    });
  }
  draw();
}

// ── STT ───────────────────────────────────────────────────
function initSTT() {
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) return;
  S.sttRecog = new SR();
  S.sttRecog.continuous     = false;
  S.sttRecog.interimResults = true;
  S.sttRecog.lang            = 'en-US';
  S.sttRecog.onresult = (e) => {
    let interim='', final='';
    for (const r of e.results) {
      if (r.isFinal) final += r[0].transcript;
      else interim += r[0].transcript;
    }
    const inp = document.getElementById('cin');
    if (inp) inp.value = final || interim;
  };
  S.sttRecog.onend = () => {
    S.sttActive = false;
    const btn = document.getElementById('stt-btn');
    if (btn) btn.classList.remove('on');
    const st = document.getElementById('stt-st');
    if (st) st.textContent = '';
    const inp = document.getElementById('cin');
    if (inp && inp.value.trim()) sendMsg();
  };
  S.sttRecog.onerror = () => {
    S.sttActive = false;
    const btn = document.getElementById('stt-btn');
    if (btn) btn.classList.remove('on');
  };
}

function toggleSTT() {
  if (!S.sttRecog) { alert('Speech recognition requires Chrome or Edge.'); return; }
  if (S.sttActive) {
    S.sttRecog.stop(); S.sttActive = false;
  } else {
    S.sttActive = true;
    const btn = document.getElementById('stt-btn');
    const st  = document.getElementById('stt-st');
    if (btn) btn.classList.add('on');
    if (st)  st.textContent = 'Listening...';
    S.sttRecog.start();
  }
}

// ── Passive signal ────────────────────────────────────────
async function sendSignal(text) {
  if (!text?.trim()) return;
  try {
    const r = await fetch('/remi/signal', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({text})
    });
    const d = await r.json();
    updateOrb(d.signal);
  } catch(e) {}
}

function updateOrb(signal) {
  S.signalHistory.push(signal);
  if (S.signalHistory.length > 6) S.signalHistory.shift();
  const avg = S.signalHistory.reduce((a,b) => a+b, 0) / S.signalHistory.length;
  const orb = document.querySelector('.orb');
  if (!orb) return;
  orb.classList.remove('calm','mixed','low');
  if (avg > 0.6)      orb.classList.add('calm');
  else if (avg > 0.4) orb.classList.add('mixed');
  else                orb.classList.add('low');
}

// ── Navigation ────────────────────────────────────────────
function goTo(i) {
  const el = document.getElementById(SECS[i]);
  if (el) { el.scrollIntoView({behavior:'smooth'}); updateDots(i); }
}
function updateDots(i) {
  document.querySelectorAll('.ndot').forEach((d,j) => d.classList.toggle('on', i===j));
}
function initObs() {
  const obs = new IntersectionObserver(entries => {
    entries.forEach(e => {
      if (e.isIntersecting) {
        e.target.classList.add('vis');
        const i = SECS.indexOf(e.target.id);
        if (i >= 0) updateDots(i);
      }
    });
  }, {threshold:0.3});
  SECS.forEach(id => { const el=document.getElementById(id); if(el) obs.observe(el); });
}

// ── Particles ─────────────────────────────────────────────
function initParticles() {
  const c = document.querySelector('.particles');
  if (!c) return;
  const cols = ['#b7e4c7','#74c69d','#ade8f4','#48cae4','#d8f3dc'];
  for (let i=0;i<16;i++) {
    const p=document.createElement('div'); p.className='pt';
    const s=Math.random()*13+4;
    p.style.cssText=`width:${s}px;height:${s}px;left:${Math.random()*100}%;`
      +`background:${cols[Math.floor(Math.random()*cols.length)]};`
      +`animation-duration:${Math.random()*12+7}s;animation-delay:${Math.random()*-14}s;`;
    c.appendChild(p);
  }
}

// ── Breathing ─────────────────────────────────────────────
let bCount=0;
function startBreathing() {
  const ring=document.getElementById('bring'),txt=document.getElementById('btxt'),btn=document.getElementById('bbtn');
  if (!ring) return;
  if (S.breathPhase!=='idle') { S.breathPhase='idle'; ring.classList.remove('inh','exh'); if(btn) btn.textContent='Start'; return; }
  S.breathPhase='go'; btn.textContent='Stop'; bCount=0;
  function cycle() {
    if (S.breathPhase!=='go') return;
    ring.classList.add('inh'); ring.classList.remove('exh');
    if (txt) txt.textContent='Breathe in...';
    setTimeout(()=>{
      if (S.breathPhase!=='go') return;
      if (txt) txt.textContent='Hold...';
      setTimeout(()=>{
        if (S.breathPhase!=='go') return;
        ring.classList.remove('inh'); ring.classList.add('exh');
        if (txt) txt.textContent='Breathe out...';
        setTimeout(()=>{
          if (S.breathPhase!=='go') return;
          bCount++;
          if (bCount>=4) { S.breathPhase='idle'; ring.classList.remove('inh','exh');
            if(txt) txt.textContent='Well done'; if(btn) btn.textContent='Start'; return; }
          cycle();
        },6000);
      },4000);
    },4000);
  }
  cycle();
}

// ── Wellness ──────────────────────────────────────────────
async function loadMeme() {
  try {
    const d=await fetch('/wellness/meme').then(r=>r.json());
    const el=document.getElementById('meme-c');
    if(el) el.innerHTML=`<div class="mt">${d.top}</div><div class="mb">${d.bottom}</div>`;
  } catch(e){}
}
async function loadQuote() {
  try {
    const d=await fetch('/wellness/quote').then(r=>r.json());
    const t=document.getElementById('qt'),a=document.getElementById('qa');
    if(t) t.textContent='"'+d.text+'"';
    if(a) a.textContent='— '+d.author;
  } catch(e){}
}
async function loadAffirmation() {
  try {
    const d=await fetch('/wellness/affirmation').then(r=>r.json());
    const el=document.getElementById('aff-txt');
    if(el){el.style.opacity=0;setTimeout(()=>{el.textContent=d.text;el.style.opacity=1;},280);}
  } catch(e){}
}
function saveGratitude() {
  const inp=document.getElementById('ginp'),sav=document.getElementById('gsav');
  if(!inp||!inp.value.trim()) return;
  const e=JSON.parse(localStorage.getItem('haven_grat')||'[]');
  e.unshift({text:inp.value.trim(),date:new Date().toLocaleDateString()});
  localStorage.setItem('haven_grat',JSON.stringify(e.slice(0,30)));
  inp.value='';
  if(sav){sav.style.display='block';setTimeout(()=>sav.style.display='none',2000);}
}
const MUSIC={calm:'https://open.spotify.com/playlist/37i9dQZF1DX3Ogo9pFvBkY',
  focus:'https://open.spotify.com/playlist/37i9dQZF1DXc8kgYqQLMfH',
  happy:'https://open.spotify.com/playlist/37i9dQZF1DX9XIFQuFvzM4',
  sleep:'https://open.spotify.com/playlist/37i9dQZF1DWZd79rJ6a7lp'};
function selectMood(m) {
  document.querySelectorAll('.mbtn').forEach(b=>b.classList.toggle('on',b.dataset.mood===m));
  const l=document.getElementById('mlink');
  if(l){l.href=MUSIC[m]||'#';l.style.display='inline-flex';}
}

// ── Chat ──────────────────────────────────────────────────
async function startChat() {
  if (!S.chatStarted) {
    // Request permissions first
    const perms = await requestPermissions();
    S.chatStarted = true;
    // Start captures
    if (perms.mic) await startAudioCapture();
    if (perms.cam) await startVideoCapture();
  }
  goTo(2);
  const msgs = document.getElementById('cmsgs');
  if (msgs) msgs.innerHTML = '';
  S.signalHistory = [];
  S.audioChunkCount = 0;

  try {
    const d = await fetch('/remi/start',{method:'POST'}).then(r=>r.json());
    addRemiMsg(d.message, d.card);
    speak(d.tts_text || d.message);
    updatePhase(1);
  } catch(e) {
    addRemiMsg("Hey! Really glad you're here.", {
      type:'emoji_mood', emojis:['😄','🙂','😐','😔','😞'],
      labels:['Great','Good','Okay','Low','Rough']
    });
  }
}

async function sendMsg() {
  const inp = document.getElementById('cin');
  if (!inp) return;
  const txt = inp.value.trim();
  if (!txt) return;
  inp.value=''; inp.style.height='auto';
  addUserMsg(txt);
  sendSignal(txt);
  showTyping();
  try {
    const d = await fetch('/remi/chat',{
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({message:txt})
    }).then(r=>r.json());
    hideTyping();
    addRemiMsg(d.message, d.card);
    speak(d.tts_text||d.message);
    updatePhase(d.phase);
    if (d.analysis_ready) setTimeout(runAnalysis, 1600);
  } catch(e) {
    hideTyping();
    addRemiMsg("I'm here. Tell me more about that.");
  }
}

function addRemiMsg(text, card) {
  const c = document.getElementById('cmsgs');
  if (!c) return;
  const d = document.createElement('div');
  d.className='msg mr';
  d.innerHTML=`<div class="mav">🤖</div><div class="bub">${text}</div>`;
  c.appendChild(d);
  if (card) {
    const cd=document.createElement('div');
    cd.className='icard';
    cd.innerHTML=buildCard(card);
    c.appendChild(cd);
  }
  c.scrollTop=c.scrollHeight;
}

function addUserMsg(text) {
  const c=document.getElementById('cmsgs');
  if(!c) return;
  const d=document.createElement('div');
  d.className='msg mu';
  d.innerHTML=`<div class="bub">${text}</div>`;
  c.appendChild(d); c.scrollTop=c.scrollHeight;
}

function showTyping() {
  const c=document.getElementById('cmsgs');
  if(!c) return;
  const d=document.createElement('div');
  d.id='typ'; d.className='msg mr';
  d.innerHTML='<div class="mav">🤖</div><div class="bub"><div class="tdots"><div class="td"></div><div class="td"></div><div class="td"></div></div></div>';
  c.appendChild(d); c.scrollTop=c.scrollHeight;
  const sb=document.getElementById('sbtn'); if(sb) sb.disabled=true;
}
function hideTyping() {
  const el=document.getElementById('typ'); if(el) el.remove();
  const sb=document.getElementById('sbtn'); if(sb) sb.disabled=false;
}
function updatePhase(p) {
  S.phase=p;
  document.querySelectorAll('.phd').forEach((d,i) => {
    d.classList.remove('act','done');
    if(i+1<p) d.classList.add('done');
    else if(i+1===p) d.classList.add('act');
  });
}
function cinKey(e) {
  if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();sendMsg();}
  const i=document.getElementById('cin');
  if(i){i.style.height='auto';i.style.height=Math.min(i.scrollHeight,88)+'px';}
}

// ── Interactive cards ─────────────────────────────────────
function buildCard(card) {
  switch(card.type){
    case 'emoji_mood':
      return `<div class="emoji-row">${card.emojis.map((e,i)=>
        `<button class="emoj-btn" onclick="submitCard('${e} ${card.labels[i]}',this)">
          <span class="emoj-e">${e}</span><span class="emoj-l">${card.labels[i]}</span>
        </button>`).join('')}</div>`;
    case 'this_or_that':
      return `<div><div class="tot-q">${card.question}</div>
        <div class="tot-wrap">
          <button class="tot-card" onclick="submitCard('${card.option_a}',this,'tot')">${card.option_a}</button>
          <button class="tot-card" onclick="submitCard('${card.option_b}',this,'tot')">${card.option_b}</button>
        </div></div>`;
    case 'finish_sentence':
      return `<div class="fs-wrap"><div class="fs-prompt">${card.prompt}</div>
        <input class="fs-input" placeholder="...complete the thought" onkeydown="fsKey(event,this)"/>
        <button class="fs-send" onclick="submitFS(this)">Send →</button></div>`;
    case 'word_assoc':
      return `<div class="wa-wrap"><div class="wa-prompt">${card.prompt}</div>
        <input class="wa-input" placeholder="first word..." onkeydown="waKey(event,this)"/>
        <button class="wa-send" onclick="submitWA(this)">→</button></div>`;
    case 'memory':
      return `<div class="mem-card"><div class="mem-prompt">${card.prompt}</div>
        <textarea class="mem-input" rows="3" placeholder="Take your time..."></textarea>
        <button class="mem-send" onclick="submitMem(this)">Share →</button></div>`;
    case 'color_mood':
      return `<div><div style="font-size:11px;color:var(--tmu);margin-bottom:8px;">Pick the color that matches your vibe right now</div>
        <div class="col-wrap">${card.colors.map(c=>
          `<div style="display:flex;flex-direction:column;align-items:center;gap:3px;">
            <button class="col-btn" style="background:${c.hex};" title="${c.label}"
              onclick="submitCard('${c.label}',this,'col')"></button>
            <span class="col-label">${c.label}</span>
          </div>`).join('')}</div></div>`;
    default: return '';
  }
}

function submitCard(value, btn, type='') {
  if(type==='tot') btn.closest('.tot-wrap')?.querySelectorAll('.tot-card').forEach(b=>b.classList.remove('sel'));
  else if(type==='col') btn.closest('.col-wrap')?.querySelectorAll('.col-btn').forEach(b=>b.classList.remove('sel'));
  else btn.closest('.emoji-row')?.querySelectorAll('button').forEach(b=>b.classList.remove('sel'));
  btn.classList.add('sel');
  setTimeout(()=>{
    btn.closest('.icard')?.querySelectorAll('button,input').forEach(el=>el.disabled=true);
    addUserMsg(value); sendSignal(value); sendMsgDirect(value);
  },280);
}
function submitFS(btn) {
  const inp=btn.previousElementSibling;
  if(!inp||!inp.value.trim()) return;
  const val=inp.value.trim();
  btn.closest('.icard')?.querySelectorAll('button,input').forEach(el=>el.disabled=true);
  addUserMsg(val); sendSignal(val); sendMsgDirect(val);
}
function submitWA(btn) {
  const inp=btn.closest('.wa-wrap')?.querySelector('.wa-input');
  if(!inp||!inp.value.trim()) return;
  const val=inp.value.trim();
  btn.closest('.icard')?.querySelectorAll('button,input').forEach(el=>el.disabled=true);
  addUserMsg(val); sendSignal(val); sendMsgDirect(val);
}
function submitMem(btn) {
  const inp=btn.previousElementSibling;
  if(!inp||!inp.value.trim()) return;
  const val=inp.value.trim();
  btn.closest('.icard')?.querySelectorAll('button,input,textarea').forEach(el=>el.disabled=true);
  addUserMsg(val); sendSignal(val); sendMsgDirect(val);
}
function fsKey(e,inp){if(e.key==='Enter'){e.preventDefault();inp.nextElementSibling.click();}}
function waKey(e,inp){if(e.key==='Enter'){e.preventDefault();inp.nextElementSibling.click();}}

async function sendMsgDirect(text) {
  showTyping();
  try {
    const d=await fetch('/remi/chat',{
      method:'POST', headers:{'Content-Type':'application/json'},
      body:JSON.stringify({message:text})
    }).then(r=>r.json());
    hideTyping();
    addRemiMsg(d.message, d.card);
    speak(d.tts_text||d.message);
    updatePhase(d.phase);
    if(d.analysis_ready) setTimeout(runAnalysis,1600);
  } catch(e) { hideTyping(); addRemiMsg("That means a lot. Tell me more."); }
}

// ── Analysis ──────────────────────────────────────────────
async function runAnalysis() {
  // Stop captures
  stopVAD();
  clearInterval(S.passiveFrameInt);

  addRemiMsg("Give me just a second... putting something special together for you ✨");
  speak("Give me just a second. I'm putting something special together for you.");

  const fd = new FormData();

  try {
    const d = await fetch('/remi/analyze',{method:'POST',body:fd}).then(r=>r.json());
    if(d.error) throw new Error(d.error);
    showWellnessCard(d);
  } catch(e) {
    showWellnessCard({
      wellness_label:'Balancing', vibe_score:62,
      tips:[
        "Take a few deep breaths when things feel heavy.",
        "Reach out to someone you trust this week.",
        "Small moments of rest are valid and necessary.",
        "One kind act for yourself today."
      ],
      inputs_used:{text:true,audio:false,video:false}
    });
  }

  // Stop remaining captures
  stopAudioCapture();
  stopVideoCapture();
}

function showWellnessCard(data) {
  const sec=document.getElementById('s-res');
  if(!sec) return;
  const score=data.vibe_score||65;
  const ring=document.querySelector('.vring');
  if(ring) ring.style.setProperty('--pct', score+'%');

  const vn=document.getElementById('vnum'),ws=document.getElementById('wls'),wm=document.getElementById('wlm');
  if(vn) vn.textContent=score;
  if(ws) ws.textContent=data.wellness_label||'Balancing';
  const msgs={
    'Thriving':"You're radiating great energy. Keep nurturing what's working.",
    'Balancing':"You're navigating a mixed season. That takes real strength.",
    'Recharging':"It sounds like you've been carrying a lot. You deserve rest and care.",
    'Seeking support':"It takes real courage to check in with yourself. You're not alone.",
  };
  if(wm) wm.textContent=msgs[data.wellness_label]||msgs['Balancing'];

  // Show which modalities contributed
  const iu = data.inputs_used || {};
  const modRow = document.getElementById('modality-row');
  if (modRow) {
    modRow.innerHTML = ['text','audio','video'].map(m =>
      `<span class="mod-tag ${iu[m]?'on':'off'}">${m==='text'?'💬':m==='audio'?'🎙️':'📷'} ${m} ${iu[m]?'✓':'—'}</span>`
    ).join('');
  }

  const tc=document.getElementById('tc');
  const icons=['🌿','💙','🌤️','✨','🫂','🧘'];
  if(tc&&data.tips){
    tc.innerHTML=data.tips.map((t,i)=>
      `<div class="tcard"><span class="tico">${icons[i%icons.length]}</span><span>${t}</span></div>`
    ).join('');
  }
  sec.classList.add('show');
  setTimeout(()=>goTo(3),500);
  speak("Your wellness card is ready. "+(msgs[data.wellness_label]||msgs['Balancing']));
}

// ── Init ──────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  initParticles(); initObs(); initSTT();
  loadMeme(); loadQuote(); loadAffirmation();
  const vd=document.querySelector('.vdot');
  if(vd) vd.classList.add('on');
  document.querySelectorAll('.ndot').forEach((d,i)=>d.addEventListener('click',()=>goTo(i)));
  const cin=document.getElementById('cin');
  if(cin) cin.addEventListener('keydown',cinKey);

  Object.assign(window,{
    goTo,toggleVoice,startChat,sendMsg,toggleSTT,
    startBreathing,loadMeme,loadAffirmation,selectMood,saveGratitude,
    submitCard,submitFS,submitWA,submitMem,fsKey,waKey,
  });
});