"""Tests not loading Qwen weights (backend contract, privacy, UI availability)."""
import io
import json

import pytest
from fastapi.testclient import TestClient
import main


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(main, 'RESULTS_DIR', tmp_path)
    return TestClient(main.app)


def test_status(client):
    r=client.get('/api/status')
    assert r.status_code==200
    assert r.json()['status']=='ok'
    assert r.json()['model'].endswith('Qwen2-VL-2B-Instruct')


def test_static_site(client):
    r=client.get('/')
    assert r.status_code==200
    assert 'AquaLens Studio' in r.text
    assert 'Gizlilik modu' in r.text
    assert client.get('/app.js').status_code==200
    assert client.get('/styles.css').status_code==200


def test_save_private_default(client, tmp_path):
    payload={'results':[{'file_name':'Kimlikli_Kisi_123.jpg','status':'success','m3_degeri':'01923','seri_no':'12345678','message':'Adres bilgisi'}]}
    r=client.post('/api/save-results',json=payload)
    assert r.status_code==200
    path=tmp_path/r.json()['file_name']
    saved=json.loads(path.read_text(encoding='utf-8'))
    row=saved['results'][0]
    assert saved['anonymized'] is True
    assert row['file_name']=='goruntu_001'
    assert row['seri_no']=='••••5678'
    assert row['message'] is None
    assert 'Kimlikli_Kisi' not in path.read_text(encoding='utf-8')
    history=client.get('/api/history').json()['sessions']
    assert len(history)==1
    assert history[0]['anonymized'] is True
    detail=client.get('/api/history/'+path.name)
    assert detail.json()['results'][0]['m3_degeri']=='01923'


def test_explicit_unmasked(client, tmp_path):
    payload={'anonymize':False,'results':[{'file_name':'a.jpg','status':'success','m3_degeri':'100','seri_no':'87654321'}]}
    r=client.post('/api/save-results',json=payload)
    saved=json.loads((tmp_path/r.json()['file_name']).read_text(encoding='utf-8'))
    assert saved['results'][0]['seri_no']=='87654321'


def test_invalid_image(client):
    r=client.post('/api/analyze',files={'file':('broken.png',b'not-an-image','image/png')})
    assert r.status_code==422
    assert r.json()['status']=='error'


def test_history_path_guard(client):
    assert client.get('/api/history/other.json').status_code==400
    assert client.get('/api/history/oturum_does_not_exist.json').status_code==404
