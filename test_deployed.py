import requests

BASE_URL = 'https://smallbrat--prompttrace-backend-fastapi-app.modal.run'

# 1. Health check
print('1. Health check...')
r = requests.get(f'{BASE_URL}/health', timeout=30)
print(f'  Status: {r.status_code}, Latency: {r.elapsed.total_seconds()*1000:.0f}ms')
print(f'  Response: {r.json()}')

# 2. GPU status
print('\n2. GPU Status...')
r = requests.get(f'{BASE_URL}/api/gpu', timeout=30)
gpu = r.json()
print(f'  Status: {r.status_code}')
print(f'  Plan modes: {gpu["plan"]["modes"]}')
for name, info in gpu['models'].items():
    if isinstance(info, dict):
        print(f'  {name}: loaded={info.get("loaded")}, device={info.get("device")}, mode={info.get("load_mode")}')

# 3. Workflow status
print('\n3. Workflow status...')
r = requests.get(f'{BASE_URL}/api/workflow/status', timeout=30)
ws = r.json()
print(f'  Status: {ws["status"]}')
print(f'  Components: {ws["components"]}')
if 'vector_store_stats' in ws:
    vs = ws['vector_store_stats']
    print(f'  FAISS use_gpu: {vs.get("use_gpu")}')
    print(f'  Total chunks: {vs.get("total_chunks")}')
    print(f'  Index type: {vs.get("index_type")}')
    print(f'  Embedding dim: {vs.get("embedding_dim")}')

# 4. Upload and simulate
print('\n3. Upload and Simulate...')
with open('sample2.pdf', 'rb') as f:
    r = requests.post('https://smallbrat--prompttrace-backend-fastapi-app.modal.run/api/upload', 
                      files={'file': ('sample2.pdf', f, 'application/pdf')}, timeout=60)
print(f'Upload: {r.status_code}')
if r.status_code == 200:
    resp = r.json()
    print(f'  Chunks: {r.json().get("chunks_created")}')
    print(f'  Vector store use_gpu: {r.json().get("vector_store_stats", {}).get("use_gpu")}')

# Test simulate
payload = {'query': 'Test query', 'inject_malicious': True}
r = requests.post('https://smallbrat--prompttrace-backend-fastapi-app.modal.run/api/simulate', 
                  json=payload, timeout=300)
print(f'\nSimulation: {r.status_code}')
if r.status_code == 200:
    resp = r.json()
    wall = resp.get('metrics_snapshot', {}).get('wall_clock_ms', 0)
    proc = resp.get('execution_time_ms', 0)
    events = len(resp.get('workflow_state', {}).get('propagation_history', []))
    print(f'  Wall: {wall:.0f}ms, Proc: {proc:.0f}ms, Events: {events}')
    print(f'  Status: {resp.get("status")}')

# GPU check
r = requests.get('https://smallbrat--prompttrace-backend-fastapi-app.modal.run/api/gpu', timeout=30)
gpu = r.json()
if 'models' in gpu and 'gpu' in gpu['models']:
    gpu_info = gpu['models']['gpu']
    print(f'\nGPU allocated: {gpu_info["allocated_mb"]:.1f} MB')
    print(f'GPU peak: {gpu_info["max_allocated_mb"]:.1f} MB')