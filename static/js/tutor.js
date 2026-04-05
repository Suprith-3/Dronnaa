// ── AI CHAT (chat page) ──────────────────────────────────────────
async function sendChatMessage(){
  const input=document.getElementById('chat-input');
  if(!input) return;
  const msg=input.value.trim();
  if(!msg) return;
  appendChatBubble(msg,'user');
  input.value='';
  showTypingIndicator();
  try{
    const res=await fetch('/tutor/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message:msg})});
    const data=await res.json();
    hideTypingIndicator();
    appendChatBubble(data.reply||data.message||'Error.','bot');
  } catch(e){ hideTypingIndicator(); appendChatBubble('Network error.','bot'); }
}

function appendChatBubble(text, role){
  const box=document.getElementById('chat-messages');
  if(!box) return;
  const div=document.createElement('div');
  div.className=role==='user'?'flex justify-end mb-4 fade-in':'flex justify-start mb-4 fade-in';
  const bubble=document.createElement('div');
  bubble.className=role==='user'?'bg-green-50 border border-green-200 px-5 py-3.5 rounded-3xl rounded-br-md max-w-sm text-sm text-gray-800 shadow-sm transition-all hover:shadow-md':'bg-blue-50 border border-blue-100 px-5 py-3.5 rounded-3xl rounded-bl-md max-w-lg text-sm text-gray-800 shadow-sm transition-all hover:shadow-md';
  if(role==='bot') {
    bubble.innerHTML=marked.parse(text);
    window.renderMath(bubble);
  } else bubble.textContent=text;
  div.appendChild(bubble);
  box.appendChild(div);
  box.scrollTo({top: box.scrollHeight, behavior: 'smooth'});
}

function showTypingIndicator(){
  const box=document.getElementById('chat-messages');
  if(box && !document.getElementById('typing')) {
    const div = document.createElement('div');
    div.id = 'typing';
    div.className = 'flex justify-start mb-4 fade-in';
    div.innerHTML = `
        <div class="bg-blue-50 border border-blue-100 px-5 py-3.5 rounded-3xl rounded-bl-md text-sm shadow-sm flex items-center gap-1">
            <span class="typing-dot"></span>
            <span class="typing-dot"></span>
            <span class="typing-dot"></span>
        </div>`;
    box.appendChild(div);
    box.scrollTo({top: box.scrollHeight, behavior: 'smooth'});
  }
}
function hideTypingIndicator(){ document.getElementById('typing')?.remove(); }

// ── QUIZ PAGE ────────────────────────────────────────────────────
let quiz=[], selectedAnswers={}, currentQuestion=0, timerInterval;

async function generateQuiz(){
  const topic=document.getElementById('quiz-topic').value.trim();
  if(!topic){ alert('Enter a topic.'); return; }
  document.getElementById('quiz-loading').classList.remove('hidden');
  document.getElementById('quiz-input-area').classList.add('hidden');
  try{
    const res=await fetch('/tutor/api/generate-quiz',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({topic})});
    const data=await res.json();
    if(data.success){ quiz=data.questions; currentQuestion=0; selectedAnswers={}; showQuestion(); }
    else showQuizError(data.message);
  } catch(e){ showQuizError('Network error.'); }
  finally{ document.getElementById('quiz-loading').classList.add('hidden'); }
}

function showQuestion(){
  const q=quiz[currentQuestion];
  const display = document.getElementById('quiz-display');
  display.classList.remove('hidden');
  display.classList.remove('active-question');
  void display.offsetWidth; // Force reflow
  display.classList.add('active-question');

  document.getElementById('q-number').textContent=`Question ${currentQuestion+1} of ${quiz.length}`;
  document.getElementById('q-text').textContent=q.question;
  document.getElementById('q-progress').style.width=`${((currentQuestion+1)/quiz.length)*100}%`;
  const opts=document.getElementById('q-options');
  opts.innerHTML='';
  Object.entries(q.options).forEach(([k,v])=>{
    const btn=document.createElement('button');
    btn.className='w-full text-left px-5 py-4 border-2 border-gray-100 rounded-2xl hover:border-blue-400 hover:bg-blue-50 transition-all text-sm font-semibold bg-white active:scale-[0.98] shadow-sm';
    btn.textContent=`${k}. ${v}`;
    btn.onclick=()=>selectOption(k, btn, q);
    opts.appendChild(btn);
  });
  startTimer();
}

function selectOption(key, btn, q){
  clearInterval(timerInterval);
  document.querySelectorAll('#q-options button').forEach(b=>b.disabled=true);
  selectedAnswers[currentQuestion]=key;
  const correct=q.answer;
  if(key===correct) btn.className=btn.className.replace('border-gray-200','border-green-500 bg-green-50');
  else{
    btn.className=btn.className.replace('border-gray-200','border-red-400 bg-red-50');
    document.querySelectorAll('#q-options button').forEach(b=>{ if(b.textContent.startsWith(correct)) b.className=b.className.replace('border-gray-200','border-green-500 bg-green-50'); });
  }
  setTimeout(()=>{ if(currentQuestion<quiz.length-1){ currentQuestion++; showQuestion(); } else showResults(); },1500);
}

function startTimer(){
  let t=30;
  const timerEl = document.getElementById('timer');
  if(timerEl) timerEl.textContent=t;
  clearInterval(timerInterval);
  timerInterval=setInterval(()=>{ t--; if(timerEl) timerEl.textContent=t; if(t<=0){ clearInterval(timerInterval); if(currentQuestion<quiz.length-1){currentQuestion++;showQuestion();}else showResults(); }},1000);
}

function showResults(){
  document.getElementById('quiz-display').classList.add('hidden');
  const correct=Object.entries(selectedAnswers).filter(([i,a])=>quiz[i].answer===a).length;
  const pct=Math.round((correct/quiz.length)*100);
  document.getElementById('results-display').classList.remove('hidden');
  document.getElementById('final-score').textContent=`${correct}/${quiz.length}`;
  document.getElementById('final-pct').textContent=`${pct}%`;
  // Save results
  fetch('/tutor/api/save-quiz',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({topic:document.getElementById('quiz-topic').value,score:correct,total:quiz.length,percentage:pct,details:quiz.map((q,i)=>({question:q.question,user_answer:selectedAnswers[i]||'Skipped',correct_answer:q.answer,is_correct:selectedAnswers[i]===q.answer}))})});
}

function showQuizError(msg){
  alert(msg);
  document.getElementById('quiz-input-area').classList.remove('hidden');
}

// ── NOTES GENERATOR (dashboard) ──────────────────────────────────
async function generateAndDownloadNotes(){
  const topicInput = document.getElementById('notes-topic-input');
  if(!topicInput) return;
  const topic=topicInput.value.trim();
  if(!topic){ alert('Enter a topic.'); return; }
  const btn=document.querySelector('[onclick="generateAndDownloadNotes()"]');
  if(btn){ btn.disabled=true; btn.textContent='Generating...'; }
  try{
    const res=await fetch('/tutor/api/generate-notes',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({topic,format:'docx'})});
    const data=await res.json();
    if(data.success&&data.download_url){ 
        window.location.href=data.download_url; 
    } else alert(data.message||'Error generating notes.');
  } catch(e){ alert('Network error.'); }
  finally{ if(btn){ btn.disabled=false; btn.textContent='Generate & Download'; } }
}

// ── ENTER KEY SUPPORT ─────────────────────────────────────────────
document.addEventListener('DOMContentLoaded',()=>{
  document.getElementById('chat-input')?.addEventListener('keydown',e=>{ if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();sendChatMessage();} });
  document.getElementById('quiz-topic')?.addEventListener('keydown',e=>{ if(e.key==='Enter') generateQuiz(); });
  document.getElementById('topic-input')?.addEventListener('keydown',e=>{ if(e.ctrlKey&&e.key==='Enter') generatePaper(); });
});
