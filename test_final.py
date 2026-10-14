import requests
import time

BASE_URL = 'https://smallbrat--prompttrace-backend-fastapi-app.modal.run'

# 1. Health check
print('=== Health Check ===')
r = requests.get(f'{BASE_URL}/health', timeout=30)
print(f'Status: {r.status_code}, Latency: {r.elapsed.total_seconds()*1000:.0f}ms')
print(f'Response: {r.json()}')

# 2. GPU Status
print('\n=== GPU Status ===')
r = requests.get(f'{BASE_URL}/api/gpu', timeout=30)
gpu = r.json()
print('Plan modes:', gpu['plan']['modes'])
for name, info in gpu['models'].items():
    if isinstance(info, dict):
        print(f'  {name}: loaded={info.get("loaded")}, device={info.get("device")}, mode={info.get("load_mode")}')

# 2. GPU memory
if 'models' in gpu and 'gpu' in gpu['models']:
    gpu_info = gpu['models']['gpu']
    print('GPU allocated:', gpu_info['allocated_mb'], 'MB')
    print('GPU peak:', gpu_info['max_allocated_mb'], 'MB')

# 2. Workflow status
r = requests.get('https://smallbrat--prompttrace-backend-fastapi-app.modal.run/api/workflow/status', timeout=30)
ws = r.json()
print('\nWorkflow status:', ws['status'])
print('Components:', ws['components'])
if 'vector_store_stats' in ws:
    vs = ws['vector_store_stats']
    print('FAISS use_gpu:', vs.get('use_gpu'))
    print('Total chunks:', vs.get('total_chunks'))
    print('Index type:', vs.get('index_type'))
    print('Embedding dim:', vs.get('embedding_dim'))

# 2. Upload PDF
print('\n=== Upload PDF ===')
with open('sample2.pdf', 'rb') as f:
    files = {'file': ('sample2.pdf', f, 'application/pdf')}
    r = requests.post('https://smallbrat--prompttrace-backend-fastapi-app.modal.run/api/upload', files=files, timeout=60)
print('Upload:', r.status_code)
if r.status_code == 200:
    print('Chunks:', r.json().get('chunks_created'))
    print('Vector store use_gpu:', r.json().get('vector_store_stats', {}).get('use_gpu'))

# 2. Simulate
print('\n=== Simulate ===')
payload = {'query': 'What are the key security considerations?', 'inject_malicious': True}
start = time.time()
r = requests.post('https://smallbrat--prompttrace-backend-fastapi-app.modal.run/api/simulate', json=payload, timeout=300)
elapsed = (time.time() - start) * 1000
if r.status_code == 200:
    resp = r.json()
    wall = resp.get('metrics_snapshot', {}).get('wall_clock_ms', 0)
    proc = resp.get('execution_time_ms', 0)
    events = len(resp.get('workflow_state', {}).get('propagation_history', []))
    print(f'Simulate: {r.status_code}, HTTP={elapsed:.0f}ms, Wall={wall:.0f}ms, Proc={proc:.0f}ms, Events={events}')
    print('Status:', resp.get('status'))
    
    ws = resp.get('workflow_state', {})
    for key in ['research_output', 'analyst_output', 'action_output']:
        if key in ws:
            data = ws[key]
            if isinstance(data, dict):
                print(f'  {key}: summary={len(data.get("summary", ""))} chars, findings={len(data.get("key_findings", []))}')
            else:
                print(f'  {key}: {type(data)}')

# GPU check after
r = requests.get('https://smallbrat--prompttrace-backend-fastapi-app.modal.run/api/gpu', timeout=30)
gpu = r.json()
if 'models' in gpu and 'gpu' in gpu['models']:
    gpu_info = gpu['models']['gpu']
    print(f'GPU allocated: {gpu_info["allocated_mb"]:.1f} MB')
    print(f'GPU peak: {gpu_info["max_allocated_mb"]:.1f} MB')