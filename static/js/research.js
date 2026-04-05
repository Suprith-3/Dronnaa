let currentMarkdown = '', currentTopic = '', currentSource = '';
let currentSurveyMarkdown = '', currentSurveyTopic = '';

// ── PAPER GENERATION ────────────────────────────────────────────
async function generatePaper(){
  const topic=document.getElementById('topic-input').value.trim();
  const language=document.getElementById('language-select').value;
  if(!topic){ showToast('Please enter a research topic.','error'); return; }
  currentTopic=topic;
  showSkeleton('paper-output-area');
  const btn=document.getElementById('generate-paper-btn');
  btn.disabled=true; btn.textContent='Generating...';
  const msgs=['Analyzing academic sources...','Structuring your paper...','Generating references...','Finalizing content...'];
  let mi=0; const interval=setInterval(()=>{ const el=document.getElementById('loading-msg'); if(el) el.textContent=msgs[mi++%msgs.length]; },750);
  try{
    const res=await fetch('/research/generate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({topic,language})});
    const data=await res.json();
    clearInterval(interval);
    if(data.success){
      currentMarkdown=data.content;
      const previewEl=document.getElementById('paper-preview');
      previewEl.innerHTML=marked.parse(currentMarkdown);
      window.renderMath(previewEl);
      document.getElementById('paper-raw').value=currentMarkdown;
      document.getElementById('paper-output-area').classList.remove('hidden');
      document.getElementById('skeleton-area').classList.add('hidden');
      document.getElementById('qa-checklist').classList.remove('hidden');
      const wc=currentMarkdown.trim().split(/\s+/).length;
      document.getElementById('word-count').textContent=`~${wc} words`;
      showToast('Paper generated successfully!','success');
    } else { hideSkeleton(); showToast(data.message||'Generation failed.','error'); }
  } catch(e){ clearInterval(interval); hideSkeleton(); showToast('Network error.','error'); }
  finally{ btn.disabled=false; btn.textContent='Generate Paper'; }
}

async function downloadPaper(){
  if(!currentMarkdown){ showToast('Generate a paper first.','error'); return; }
  const fmt=document.getElementById('download-format').value.toLowerCase();
  const btn=document.getElementById('download-btn');
  btn.disabled=true; btn.textContent='Preparing...';
  try{
    const res=await fetch('/research/download',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({content:currentMarkdown,format:fmt})});
    const ct=res.headers.get('content-type')||'';
    if(ct.includes('application/json')){ const d=await res.json(); throw new Error(d.message); }
    const blob=await res.blob();
    const url=URL.createObjectURL(blob);
    const a=document.createElement('a'); a.href=url; a.download=`research_paper.${fmt==='txt'?'txt':fmt}`; a.click(); URL.revokeObjectURL(url);
    showToast(`${fmt.toUpperCase()} downloaded!`,'success');
  } catch(e){ showToast('Download failed: '+e.message,'error'); }
  finally{ btn.disabled=false; btn.textContent='⬇ Download'; }
}

async function copyPaper(){
  if(!currentMarkdown){ showToast('Nothing to copy.','error'); return; }
  await navigator.clipboard.writeText(currentMarkdown);
  showToast('Copied to clipboard!','success');
}

function regeneratePaper(){ if(currentTopic){ document.getElementById('topic-input').value=currentTopic; generatePaper(); } }

// ── SURVEY PAPER (DUAL SYNTHESIS) ───────────────────────────────
async function generateSurvey(){
  const topic=document.getElementById('survey-topic-input').value.trim();
  const language=document.getElementById('survey-language-select').value;
  if(!topic){ showToast('Please enter a survey topic.','error'); return; }
  currentSurveyTopic=topic;
  
  document.getElementById('survey-skeleton-area').classList.remove('hidden');
  document.getElementById('survey-output-area').classList.add('hidden');
  
  const btn=document.getElementById('generate-survey-btn');
  btn.disabled=true; btn.textContent='Synthesizing...';
  
  const msgs=['Generating Perspective A...', 'Generating Perspective B...', 'Comparing theoretical frameworks...', 'Synthesizing final paper...', 'Checking plagiarism...'];
  let mi=0; const msgEl=document.getElementById('survey-loading-msg');
  const interval=setInterval(()=>{ if(msgEl) msgEl.textContent=msgs[mi++%msgs.length]; },1000);
  
  try{
    const res=await fetch('/research/generate-survey',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({topic,language})});
    const data=await res.json();
    clearInterval(interval);
    if(data.success){
      currentSurveyMarkdown=data.content;
      const previewEl=document.getElementById('survey-preview');
      previewEl.innerHTML=marked.parse(currentSurveyMarkdown);
      window.renderMath?.(previewEl);
      document.getElementById('survey-output-area').classList.remove('hidden');
      document.getElementById('survey-skeleton-area').classList.add('hidden');
      const wc=currentSurveyMarkdown.trim().split(/\s+/).length;
      document.getElementById('survey-word-count').textContent=`~${wc} words`;
      showToast('Synthesized Survey generated successfully!','success');
    } else { 
      document.getElementById('survey-skeleton-area').classList.add('hidden'); 
      showToast(data.message||'Generation failed.','error'); 
    }
  } catch(e){ 
    clearInterval(interval); 
    document.getElementById('survey-skeleton-area').classList.add('hidden'); 
    showToast('Network error.','error'); 
  }
  finally{ btn.disabled=false; btn.textContent='Generate Survey'; }
}

async function downloadSurvey(){
  if(!currentSurveyMarkdown){ showToast('Generate a survey first.','error'); return; }
  const fmt=document.getElementById('survey-download-format').value.toLowerCase();
  const btn=document.getElementById('survey-download-btn');
  btn.disabled=true; btn.textContent='Preparing...';
  try{
    const res=await fetch('/research/download',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({content:currentSurveyMarkdown,format:fmt})});
    const ct=res.headers.get('content-type')||'';
    if(ct.includes('application/json')){ const d=await res.json(); throw new Error(d.message); }
    const blob=await res.blob();
    const url=URL.createObjectURL(blob);
    const a=document.createElement('a'); a.href=url; a.download=`survey_synthesis.${fmt==='txt'?'txt':fmt}`; a.click(); URL.revokeObjectURL(url);
    showToast(`${fmt.toUpperCase()} downloaded!`,'success');
  } catch(e){ showToast('Download failed: '+e.message,'error'); }
  finally{ btn.disabled=false; btn.textContent='⬇ Download Survey'; }
}

// ── CITATION ────────────────────────────────────────────────────
async function generateCitation(){
  const source=document.getElementById('citation-input').value.trim();
  const style=document.getElementById('citation-style').value;
  if(!source){ showToast('Enter source information.','error'); return; }
  currentSource=source;
  const out=document.getElementById('citation-output');
  const btn=document.getElementById('citation-btn');
  out.value='Generating citation...'; btn.disabled=true; btn.textContent='Generating...';
  try{
    const res=await fetch('/research/generate-citation',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({source,style})});
    const data=await res.json();
    out.value=data.success?data.citation:data.message||'Error';
    if(data.success) showToast('Citation generated!','success');
    else showToast(data.message,'error');
  } catch(e){ out.value='Error.'; showToast('Network error.','error'); }
  finally{ btn.disabled=false; btn.textContent='Generate Citation'; }
}

async function copyCitation(){
  const txt=document.getElementById('citation-output').value;
  if(!txt){ showToast('Nothing to copy.','error'); return; }
  await navigator.clipboard.writeText(txt);
  showToast('Citation copied!','success');
}

// ── TAB SWITCHING ───────────────────────────────────────────────
function switchMainTab(tab){
  ['paper', 'survey', 'citation'].forEach(t=>{ 
    const el = document.getElementById(t + '-tab');
    if(el) el.classList.add('hidden'); 
    document.querySelector(`[data-tab="${t}-tab"]`)?.classList.remove('border-b-2','border-violet-600','font-bold'); 
  });
  const active = document.getElementById(tab + '-tab');
  if(active) active.classList.remove('hidden');
  document.querySelector(`[data-tab="${tab}-tab"]`)?.classList.add('border-b-2','border-violet-600','font-bold');
}

function switchOutputTab(tab){
  document.getElementById('paper-preview').classList.toggle('hidden', tab!=='preview');
  document.getElementById('paper-raw').classList.toggle('hidden', tab!=='raw');
  document.querySelectorAll('.output-tab').forEach(t=>t.classList.remove('bg-violet-100','text-violet-700'));
  document.querySelector(`[onclick="switchOutputTab('${tab}')"]`)?.classList.add('bg-violet-100','text-violet-700');
}

// ── SKELETON + TOAST ─────────────────────────────────────────────
function showSkeleton(area){
  document.getElementById('skeleton-area').classList.remove('hidden');
  const out = document.getElementById(area);
  if(out) out.classList.add('hidden');
  const msg = document.getElementById('loading-msg');
  if(msg) msg.textContent='Analyzing academic sources...';
}
function hideSkeleton(){ document.getElementById('skeleton-area').classList.add('hidden'); }

function showToast(msg, type='success'){
  const t=document.createElement('div');
  t.className=`fixed top-4 right-4 z-[9999] px-5 py-3 rounded-xl shadow-xl text-white text-sm font-medium transition-all ${type==='success'?'bg-green-600':'bg-red-600'}`;
  t.textContent=msg;
  document.body.appendChild(t);
  setTimeout(()=>t.remove(),3500);
}
