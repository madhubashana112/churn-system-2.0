import asyncio
import base64
import io
import zipfile

from fastapi.testclient import TestClient
from pypdf import PdfReader

from churn_platform.main import app
from churn_platform.domain.models.analysis_run import AnalysisRun, OriginalUpload
from churn_platform.presentation.api.dependencies import get_analysis_repo
from test_accounts_exports import isolated_store, signup, workspace


def analyzed_client():
    client = TestClient(app)
    signup(client)
    tenant = workspace(client)
    samples = client.get('/api/v1/upload/demo-data', params={'tenant_id':tenant}).json()['files']
    originals = {f['name']:base64.b64decode(f['content_base64']) for f in samples}
    response = client.post('/api/v1/upload/analyze', data={'tenant_id':tenant,'engine':'system'},
        files=[('files',(name,contents,'text/csv')) for name,contents in originals.items()])
    assert response.status_code == 200, response.text
    assert 'original_files' not in response.json()
    return client, tenant, originals


def test_zip_preserves_uploaded_bytes_and_latest_run(isolated_store):
    client, tenant, originals = analyzed_client()
    response = client.get('/api/v1/exports', params={'tenant_id':tenant,'format':'zip','tier':'LOW','search':'absent'})
    assert response.status_code == 200
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        assert len(archive.namelist()) == len(originals)
        assert {name.split('/',1)[1]:archive.read(name) for name in archive.namelist()} == originals
    run = asyncio.run(get_analysis_repo().latest(tenant))
    run.original_files = [OriginalUpload(filename='../../unsafe.csv',content_base64=base64.b64encode(b'unchanged\x00bytes').decode()),
                          OriginalUpload(filename='unsafe.csv',content_base64=base64.b64encode(b'second file').decode())]
    asyncio.run(get_analysis_repo().save(run))
    response = client.get('/api/v1/exports', params={'tenant_id':tenant,'format':'zip'})
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        assert archive.namelist() == ['001/unsafe.csv','002/unsafe.csv']
        assert archive.read('001/unsafe.csv') == b'unchanged\x00bytes'


def test_pdf_filters_metadata_pagination_and_legacy_uploads(isolated_store):
    client, tenant, _ = analyzed_client()
    response = client.get('/api/v1/exports',params={'tenant_id':tenant,'format':'pdf','tier':'CRITICAL','search':'usr_2'})
    assert response.status_code == 200 and response.content.startswith(b'%PDF-')
    text = '\n'.join(page.extract_text() for page in PdfReader(io.BytesIO(response.content)).pages)
    assert 'usr_2' in text and 'usr_1 |' not in text
    assert 'deterministic local scoring' in text and 'Recommended action' in text
    run = asyncio.run(get_analysis_repo().latest(tenant))
    run.original_files = []
    run.outcomes[0].playbook.action_payload = '<script>Literal text & details</script> ' * 1000
    asyncio.run(get_analysis_repo().save(run))
    response = client.get('/api/v1/exports',params={'tenant_id':tenant,'format':'pdf'})
    assert response.status_code == 200
    pages = PdfReader(io.BytesIO(response.content)).pages
    assert len(pages) > 2
    assert 'Literal text & details' in '\n'.join(p.extract_text() for p in pages)
    missing = client.get('/api/v1/exports',params={'tenant_id':tenant,'format':'zip'})
    assert missing.status_code == 404 and 'again' in missing.json()['detail']
    empty = client.get('/api/v1/exports',params={'tenant_id':tenant,'format':'pdf','search':'does-not-exist'})
    assert 'No customers match' in PdfReader(io.BytesIO(empty.content)).pages[0].extract_text()
    legacy = run.model_dump(); legacy.pop('original_files')
    assert AnalysisRun.model_validate(legacy).original_files == []


def test_new_downloads_require_owner_and_oversize_upload_does_not_replace_run(isolated_store):
    client, tenant, _ = analyzed_client()
    other = TestClient(app)
    signup(other)
    for format in ['zip','pdf']:
        assert other.get('/api/v1/exports',params={'tenant_id':tenant,'format':format}).status_code == 404
        assert TestClient(app).get('/api/v1/exports',params={'tenant_id':tenant,'format':format}).status_code == 401
    response = client.post('/api/v1/upload/analyze',data={'tenant_id':tenant},files={'files':('large.csv', b'x'*(4*1024*1024+1))})
    assert response.status_code == 413
    assert len(asyncio.run(get_analysis_repo().latest(tenant)).original_files) == 4


def test_fintech_pdf_charts_use_filtered_run_without_mutation():
    from churn_platform.presentation.pdf_report import analysis_pdf
    from churn_platform.application.use_cases.summarize_analysis import dashboard_charts
    from churn_platform.domain.models.analysis_run import EntityOutcome
    from churn_platform.domain.models.schema_mapping import SchemaMapping
    run = AnalysisRun(tenant_id='chart-test',sector='FinTech',schema_mapping=SchemaMapping(primary_entity_key='id',tables=[]),
        entities_uploaded=2,entities_analyzed=2,outcomes=[EntityOutcome(
            prediction={'entity_id':name,'risk_tier':tier,'churn_probability':prob,'dormancy_type':dormancy},
            playbook={'action_type':'REVIEW','action_payload':'Check evidence','channel':'Email'},
            features={'balance_drain_ratio':drain,'withdrawal_share_recent':drain})
            for name,tier,prob,dormancy,drain in [('a','HIGH',.7,'HARD',.8),('b','LOW',.1,'ACTIVE',.1)]])
    filtered=run.model_copy(update={'outcomes':[run.outcomes[0]]})
    charts=dashboard_charts(filtered)
    assert sum(charts['tier_mix'].datasets[0].values)==1
    assert sum(charts['probability_histogram'].datasets[0].values)==1
    assert sum(sum(d.values) for d in charts['withdrawal_share'].datasets)==1
    assert sum(len(d.points) for d in charts['drain_vs_risk'].datasets)==1
    data=analysis_pdf(run,[run.outcomes[0]],'Chart QA','HIGH','')
    pages=PdfReader(io.BytesIO(data)).pages
    text='\n'.join(p.extract_text() for p in pages)
    for title in ['Dormancy profile','Recent withdrawal share of settled flow','Customers by risk tier','Churn probability distribution','Balance drain against predicted churn']:
        assert title in text
    assert 'Selected population: 1 customers' in text
    assert len(run.outcomes)==2
    assert any(b' c' in p.get_contents().get_data() for p in pages)  # Vector curves, not a text-only export.
