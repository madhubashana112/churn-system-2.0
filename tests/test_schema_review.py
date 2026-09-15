"""Contract tests for paused ingestion, trusted mapping and contextual feature names."""
import asyncio
import base64
import gzip
import io
import json
import time
from copy import deepcopy
from unittest.mock import AsyncMock

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from conftest import authenticated_client
from churn_platform.domain.models.schema_mapping import ColumnMapping, SchemaMapping, TableClassification
from churn_platform.infrastructure.parsers.feature_synthesizer import PandasFeatureSynthesizer, resolve_temporal_anchor
from churn_platform.infrastructure.parsers.schema_resolver import AISchemaResolver
from churn_platform.infrastructure.ai.mock_qwen_gateway import MockQwenGateway
from churn_platform.infrastructure.ai.cores.saas_core import SaasCore
from churn_platform.infrastructure.repositories.state_store import store
from churn_platform.infrastructure.persistence.redis_repos import key_for, TenantSchemaMemoryRepository
from churn_platform.presentation.api.dependencies import get_analysis_repo

RAW = b'c_uid_v2,ts_x,usr_rt_val_3,junk\nu1,2025-05-30,Gold,abc\nu1,2025-05-31,Platinum,def\nu2,2025-05-25,Silver,ghi\n'


def run(coro):
    return asyncio.run(coro)


def column(name, role, confidence=1, **kw):
    return ColumnMapping(source_column=name, canonical_role=role, confidence=confidence, **kw)


def schema(columns, role='TIME_SERIES_EVENT'):
    return SchemaMapping(primary_entity_key='c_uid_v2', tables=[TableClassification(
        file_name='legacy.csv', role=role, primary_entity_key='c_uid_v2', timestamp_column='ts_x', columns=columns)])


@pytest.mark.parametrize('confidence,status', [(0,'REQUIRES_HUMAN_REVIEW'),(.7999,'REQUIRES_HUMAN_REVIEW'),(.8,'AUTO_RESOLVED'),(1,'AUTO_RESOLVED')])
def test_confidence_boundary(confidence, status):
    value = schema([column('c_uid_v2','CUSTOMER_ID',confidence), column('ts_x','TIMESTAMP')])
    assert value.status == status


@pytest.mark.parametrize('confidence', [-.1,1.1,float('nan')])
def test_invalid_confidence_rejected(confidence):
    with pytest.raises(ValidationError):
        column('id','CUSTOMER_ID',confidence)


def test_unknown_and_missing_critical_roles_force_review():
    assert schema([column('x','UNKNOWN')]).status == 'REQUIRES_HUMAN_REVIEW'
    value = schema([column('c_uid_v2','CUSTOMER_ID'), column('ts_x','TIMESTAMP')], 'TRANSACTIONAL')
    assert value.assess('FinTech').status == 'REQUIRES_HUMAN_REVIEW'
    value.tables[0].columns.append(column('amt','TRANSACTION_AMOUNT'))
    assert value.assess('FinTech').status == 'AUTO_RESOLVED'


def test_duplicate_canonical_or_custom_names_need_review():
    cols=[column('c_uid_v2','CUSTOMER_ID'),column('ts_x','TIMESTAMP'),
          column('a','CUSTOM',custom_label='Priority SLA Tier'),column('b','CUSTOM',custom_label='Priority SLA Tier')]
    assert schema(cols).status == 'REQUIRES_HUMAN_REVIEW'
    with pytest.raises(ValueError,match='unique'):
        PandasFeatureSynthesizer().prepare(schema(cols),{'legacy.csv':pd.DataFrame(columns=['c_uid_v2','ts_x','a','b'])})


def test_custom_names_rename_copies_and_reach_ai_prompt():
    mapping=schema([column('c_uid_v2','CUSTOMER_ID'),column('ts_x','TIMESTAMP'),
                    column('usr_rt_val_3','CUSTOM',custom_label='Priority SLA Tier'),column('junk','NOISE_IGNORE')])
    original=pd.read_csv(io.BytesIO(RAW))
    synth=PandasFeatureSynthesizer()
    prepared,frames=synth.prepare(mapping,{'legacy.csv':original})
    assert list(frames['legacy.csv'].columns)==['customer_id','timestamp','Priority SLA Tier']
    assert 'usr_rt_val_3' in original and 'junk' in original
    assert mapping.tables[0].primary_entity_key=='c_uid_v2'
    features=synth.synthesize(prepared,frames)
    assert features[0].features['custom_metrics']['legacy.csv']['Priority SLA Tier']==['Gold','Platinum']
    gateway=AsyncMock()
    gateway.generate_json.return_value={'predictions':[]}
    run(SaasCore(gateway).analyze(features))
    prompt=gateway.generate_json.call_args.args[1]
    assert 'Priority SLA Tier' in prompt and 'usr_rt_val_3' not in prompt


def test_custom_numeric_metrics_and_reserved_name_remain_nested():
    mapping=schema([column('c_uid_v2','CUSTOMER_ID'),column('ts_x','TIMESTAMP'),column('v','CUSTOM',custom_label='custom_metrics')], 'DIMENSION')
    frame=pd.DataFrame({'c_uid_v2':['u1','u1'],'ts_x':['2025-05-30']*2,'v':[2.,4.]})
    result=PandasFeatureSynthesizer().synthesize(mapping,{'legacy.csv':frame})[0]
    assert result.features['custom_metrics']['legacy.csv']['custom_metrics']=={'mean':3.,'sum':6.}


def test_reference_date_is_fixed_even_for_newer_uploads():
    mapping=schema([column('c_uid_v2','CUSTOMER_ID'),column('ts_x','TIMESTAMP')])
    for date in ('2025-05-15','2026-09-15'):
        anchor,_=resolve_temporal_anchor(mapping,{'legacy.csv':pd.DataFrame({'ts_x':[date]})})
        assert anchor==pd.Timestamp('2025-06-01T12:00:00')


def test_live_resolver_uses_samples_and_distrusts_missing_or_invalid_columns():
    gateway=AsyncMock()
    gateway.generate_json.return_value={'tables':[{'file_name':'legacy.csv','role':'TIME_SERIES_EVENT',
        'primary_entity_key':'c_uid_v2','timestamp_column':'ts_x','columns':[
            {'source_column':'c_uid_v2','canonical_role':'CUSTOMER_ID','confidence':.79,'reasoning':'uncertain key'},
            {'source_column':'ts_x','canonical_role':'TIMESTAMP','confidence':9}]}]}
    result=run(AISchemaResolver(gateway).resolve({'legacy.csv':RAW.decode()}))
    assert result.status=='REQUIRES_HUMAN_REVIEW'
    assert result.tables[0].columns[1].canonical_role=='UNKNOWN'
    assert result.tables[0].columns[2].sample_values==['Gold','Platinum','Silver']
    assert 'Gold' in gateway.generate_json.call_args.args[1]
    assert 'column_profiles' in gateway.generate_json.call_args.args[1]


@pytest.fixture
def client_tenant(tmp_path, monkeypatch):
    monkeypatch.setattr(store,'path',tmp_path/'schema-test.sqlite3')
    client=authenticated_client()
    tenant=client.post('/api/v1/tenants/',json={'name':'Review test','sector':'SaaS'}).json()['tenant_id']
    return client,tenant


def upload(client,tenant):
    response=client.post('/api/v1/upload/analyze',data={'tenant_id':tenant,'engine':'system'},files={'files':('legacy.csv',RAW,'text/csv')})
    assert response.status_code==200,response.text
    return response.json()


def confirmation(tenant,review):
    roles={'c_uid_v2':'CUSTOMER_ID','ts_x':'TIMESTAMP','usr_rt_val_3':'CUSTOM','junk':'NOISE_IGNORE'}
    return {'tenant_id':tenant,'upload_session_id':review['upload_session_id'],
        'table_roles':{'legacy.csv':'TIME_SERIES_EVENT'},
        'mappings':[{'file_name':'legacy.csv','source_column':name,'canonical_role':role,
                     'custom_label':'Priority SLA Tier' if role=='CUSTOM' else None} for name,role in roles.items()]}


def test_pause_confirm_remember_replay_and_original_bytes(client_tenant, monkeypatch):
    client,tenant=client_tenant
    review=upload(client,tenant)
    assert review['requires_human_review'] is True
    assert run(get_analysis_repo().latest(tenant)) is None
    key=key_for(tenant,review['upload_session_id'])
    with store.connect() as db:
        raw,expires=db.execute('SELECT value,expires FROM state WHERE key=?',(key,)).fetchone()
    assert 3590 < expires-time.time() <= 3600
    cached=json.loads(gzip.decompress(base64.b64decode(json.loads(raw))))
    assert base64.b64decode(cached['files'][0]['content_base64'])==RAW
    payload=confirmation(tenant,review)
    response=client.post('/api/v1/upload/confirm-mapping',json=payload)
    assert response.status_code==200,response.text
    assert response.json()['requires_human_review'] is False
    assert response.json()['entities_analyzed']==2
    saved=run(get_analysis_repo().latest(tenant))
    assert saved.outcomes[0].features['custom_metrics']['legacy.csv']['Priority SLA Tier']==['Gold','Platinum']
    assert base64.b64decode(saved.original_files[0].content_base64)==RAW
    assert client.post('/api/v1/upload/confirm-mapping',json=payload).json()==response.json()
    async def fail_if_schema(self,system_prompt,user_prompt):
        if 'Data Engineer' in system_prompt:
            raise AssertionError('Remembered schema must skip inference')
        return await original(self,system_prompt,user_prompt)
    original=MockQwenGateway.generate_json
    monkeypatch.setattr(MockQwenGateway,'generate_json',fail_if_schema)
    again=upload(client,tenant)
    assert again['requires_human_review'] is False
    assert all(c['confidence']==1 for c in again['schema_mapping']['tables'][0]['columns'])


@pytest.mark.parametrize('change', ['blank','duplicate','missing','unknown','ignore_key','extra','invalid_role'])
def test_bad_confirmations_do_not_score_or_remember(client_tenant,change):
    client,tenant=client_tenant
    review=upload(client,tenant)
    payload=confirmation(tenant,review)
    if change=='blank': payload['mappings'][2]['custom_label']='   '
    if change=='duplicate': payload['mappings'][2]['custom_label']='customer_id'
    if change=='missing': payload['mappings'].pop()
    if change=='unknown': payload['mappings'][2]['canonical_role']='UNKNOWN'
    if change=='ignore_key': payload['mappings'][0]['canonical_role']='NOISE_IGNORE'
    if change=='extra': payload['mappings'].append(deepcopy(payload['mappings'][0]))
    if change=='invalid_role': payload['mappings'][0]['canonical_role']='MAGIC'
    response=client.post('/api/v1/upload/confirm-mapping',json=payload)
    assert response.status_code==400,response.text
    assert run(get_analysis_repo().latest(tenant)) is None
    assert run(TenantSchemaMemoryRepository().get(tenant,['legacy.csv']))=={}


def test_session_expiry_and_tenant_isolation(client_tenant):
    client,tenant=client_tenant
    review=upload(client,tenant)
    stranger=authenticated_client()
    payload=confirmation(tenant,review)
    assert stranger.post('/api/v1/upload/confirm-mapping',json=payload).status_code==404
    other=stranger.post('/api/v1/tenants/',json={'name':'Other','sector':'SaaS'}).json()['tenant_id']
    assert stranger.post('/api/v1/upload/confirm-mapping',json={**payload,'tenant_id':other}).status_code==410
    with store.connect() as db:
        db.execute('UPDATE state SET expires=? WHERE key=?',(time.time()-1,key_for(tenant,review['upload_session_id'])))
    assert client.post('/api/v1/upload/confirm-mapping',json=payload).status_code==410


def test_inflight_confirmation_returns_conflict(client_tenant):
    client,tenant=client_tenant
    review=upload(client,tenant)
    run(store.set(key_for(tenant,review['upload_session_id'])+':lock',True,ttl=900))
    assert client.post('/api/v1/upload/confirm-mapping',json=confirmation(tenant,review)).status_code==409


@pytest.mark.parametrize('sector',['SaaS','Telecom','FinTech'])
def test_bundled_sector_datasets_auto_resolve(client_tenant,sector):
    client,_=client_tenant
    tenant=client.post('/api/v1/tenants/',json={'name':'Demo','sector':sector}).json()['tenant_id']
    demo=client.get('/api/v1/upload/demo-data',params={'tenant_id':tenant}).json()
    files=[('files',(f['name'],base64.b64decode(f['content_base64']),'text/csv')) for f in demo['files']]
    response=client.post('/api/v1/upload/analyze',data={'tenant_id':tenant,'engine':'system'},files=files)
    assert response.status_code==200,response.text
    assert response.json()['requires_human_review'] is False
    assert response.json()['entities_analyzed']==100


def test_fintech_requirement_survives_response_serialization():
    from churn_platform.application.dtos.schema_review_dto import SchemaReviewResponse
    mapping=schema([column('c_uid_v2','CUSTOMER_ID'),column('ts_x','TIMESTAMP')], 'TRANSACTIONAL').assess('FinTech')
    response=SchemaReviewResponse(upload_session_id='test',schema_mapping=mapping)
    restored=SchemaReviewResponse.model_validate_json(response.model_dump_json())
    assert restored.schema_mapping.status=='REQUIRES_HUMAN_REVIEW'
    assert any('transaction amount' in reason for reason in restored.schema_mapping.review_reasons)


def test_pending_sessions_use_gzip_and_redis_ttl(client_tenant,monkeypatch):
    from churn_platform.domain.models.upload_session import PendingUploadSession
    from churn_platform.domain.models.analysis_run import OriginalUpload
    from churn_platform.infrastructure.persistence.redis_repos import PendingUploadRepository
    _,tenant=client_tenant
    redis=AsyncMock()
    redis.set.return_value='OK'
    monkeypatch.setattr(store,'redis',redis)
    session=PendingUploadSession(upload_session_id='review',tenant_id=tenant,sector='SaaS',engine='system',
        files=[OriginalUpload(filename='legacy.csv',content_base64=base64.b64encode(RAW).decode())],
        schema_mapping=schema([column('c_uid_v2','UNKNOWN')]))
    repo=PendingUploadRepository()
    run(repo.save(session))
    args=redis.set.call_args
    assert args.kwargs['ex']==3600
    packed=args.args[1]
    assert gzip.decompress(base64.b64decode(json.loads(packed)))
    redis.get.return_value=packed
    assert run(repo.get(tenant,'review')).files==session.files


def test_new_columns_preserve_confirmed_aliases_and_table_role():
    gateway=AsyncMock()
    gateway.generate_json.return_value={'tables':[]}
    remembered={'legacy.csv':{'role':'TIME_SERIES_EVENT','columns':{
        'c_uid_v2':{'canonical_role':'CUSTOMER_ID'},'ts_x':{'canonical_role':'TIMESTAMP'},
        'usr_rt_val_3':{'canonical_role':'CUSTOM','custom_label':'Priority SLA Tier'}}}}
    result=run(AISchemaResolver(gateway).resolve({'legacy.csv':RAW.decode()},remembered))
    table=result.tables[0]
    assert table.role=='TIME_SERIES_EVENT'
    assert table.columns[2].custom_label=='Priority SLA Tier' and table.columns[2].confidence==1
    assert table.columns[3].canonical_role=='UNKNOWN'
    assert result.status=='REQUIRES_HUMAN_REVIEW'


def test_multi_sheet_custom_mappings_are_scoped(client_tenant):
    client,tenant=client_tenant
    workbook=io.BytesIO()
    with pd.ExcelWriter(workbook,engine='openpyxl') as writer:
        for sheet,value in [('Customers','Gold'),('Support','Platinum')]:
            pd.DataFrame({'cryptic_key':['u1'],'metric_v2':[value]}).to_excel(writer,sheet_name=sheet,index=False)
    raw=workbook.getvalue()
    def submit():
        response=client.post('/api/v1/upload/analyze',data={'tenant_id':tenant,'engine':'system'},
            files={'files':('legacy.xlsx',raw,'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')})
        assert response.status_code==200,response.text
        return response.json()
    review=submit()
    assert review['requires_human_review']
    mappings=[]
    for table in review['schema_mapping']['tables']:
        mappings.extend([{'file_name':table['file_name'],'source_column':'cryptic_key','canonical_role':'CUSTOMER_ID'},
            {'file_name':table['file_name'],'source_column':'metric_v2','canonical_role':'CUSTOM',
             'custom_label':'Priority SLA Tier' if table['file_name'].endswith('Customers') else 'Support priority'}])
    response=client.post('/api/v1/upload/confirm-mapping',json={'tenant_id':tenant,
        'upload_session_id':review['upload_session_id'],'mappings':mappings})
    assert response.status_code==200,response.text
    stored=run(get_analysis_repo().latest(tenant))
    custom=stored.outcomes[0].features['custom_metrics']
    assert custom['legacy.xlsx::Customers']['Priority SLA Tier']==['Gold']
    assert custom['legacy.xlsx::Support']['Support priority']==['Platinum']
    assert base64.b64decode(stored.original_files[0].content_base64)==raw
    assert submit()['requires_human_review'] is False


def test_repeated_snapshot_metrics_keep_original_names_for_review():
    gateway=AsyncMock()
    gateway.generate_json.return_value={'tables':[{'file_name':'snapshot.csv','role':'DIMENSION','primary_entity_key':'customer_id',
        'columns':[{'source_column':name,'canonical_role':role,'confidence':.95} for name,role in
        [('customer_id','CUSTOMER_ID'),('monthly_charges','TRANSACTION_AMOUNT'),('wallet_balance','TRANSACTION_AMOUNT'),('avg_monthly_txn_val','TRANSACTION_AMOUNT')]]}]}
    csv='customer_id,monthly_charges,wallet_balance,avg_monthly_txn_val\nc1,40,100,25\n'
    mapping=run(AISchemaResolver(gateway).resolve({'snapshot.csv':csv}))
    assert mapping.status=='REQUIRES_HUMAN_REVIEW'
    assert all(c.canonical_role=='ATTRIBUTE' for c in mapping.tables[0].columns[1:])
    assert all(c.confidence < .8 for c in mapping.tables[0].columns[1:])
    features=PandasFeatureSynthesizer().synthesize(mapping,{'snapshot.csv':pd.read_csv(io.StringIO(csv))})[0].features
    assert features['monthly_charges']==40 and features['wallet_balance']==100 and features['avg_monthly_txn_val']==25


def test_repeated_transaction_amount_keeps_one_primary_and_preserves_memory():
    from churn_platform.infrastructure.parsers.schema_resolver import preserve_repeated_attributes
    table=TableClassification(file_name='payments.csv',role='TRANSACTIONAL',primary_entity_key='id',columns=[
        column('payment','TRANSACTION_AMOUNT',.95),column('fee','TRANSACTION_AMOUNT',.99),column('id','CUSTOMER_ID')])
    preserve_repeated_attributes(table,{'payment'})
    assert table.columns[0].canonical_role=='TRANSACTION_AMOUNT'
    assert table.columns[1].canonical_role=='ATTRIBUTE'
    table.columns[1]=column('fee','TRANSACTION_AMOUNT')
    preserve_repeated_attributes(table,{'payment','fee'})
    assert table.columns[1].canonical_role=='TRANSACTION_AMOUNT'  # Never override a human choice.


def test_conflict_error_identifies_columns_and_target():
    value=schema([column('c_uid_v2','CUSTOMER_ID'),column('ts_x','TIMESTAMP'),
                  column('monthly_charges','TRANSACTION_AMOUNT'),column('wallet_balance','TRANSACTION_AMOUNT')])
    error=' '.join(value.review_reasons)
    assert 'monthly_charges, wallet_balance' in error
    assert "'transaction_amount'" in error
    assert 'Attribute' in error


def test_failed_scoring_keeps_pending_upload_for_retry(client_tenant,monkeypatch):
    from churn_platform.presentation.api.v1 import upload as api
    from churn_platform.infrastructure.persistence.redis_repos import PendingUploadRepository
    client,tenant=client_tenant
    review=upload(client,tenant)
    runner=AsyncMock()
    runner.execute.return_value=[]
    monkeypatch.setattr(api,'get_analysis_use_case',lambda engine:runner)
    response=client.post('/api/v1/upload/confirm-mapping',json=confirmation(tenant,review))
    assert response.status_code==503,response.text
    assert run(get_analysis_repo().latest(tenant)) is None
    assert run(PendingUploadRepository().get(tenant,review['upload_session_id'])) is not None
    assert run(store.get(key_for(tenant,review['upload_session_id'])+':lock')) is None


def test_schema_provider_quota_returns_actionable_429(client_tenant,monkeypatch):
    from churn_platform.domain.ai_errors import AIServiceError
    from churn_platform.presentation.api.v1 import upload as api
    client,tenant=client_tenant
    resolver=AsyncMock()
    resolver.resolve.side_effect=AIServiceError('Gemini daily quota reached. Use Analyze with system model.',429,'AI_DAILY_QUOTA')
    monkeypatch.setattr(api,'get_schema_resolver',lambda engine:resolver)
    response=client.post('/api/v1/upload/analyze',data={'tenant_id':tenant,'engine':'system'},files={'files':('legacy.csv',RAW)})
    assert response.status_code==429,response.text
    assert response.json()['code']=='AI_DAILY_QUOTA'
    assert 'daily quota' in response.json()['detail']
    assert run(get_analysis_repo().latest(tenant)) is None


def test_scoring_provider_quota_keeps_pending_session(client_tenant,monkeypatch):
    from churn_platform.domain.ai_errors import AIServiceError
    from churn_platform.presentation.api.v1 import upload as api
    from churn_platform.infrastructure.persistence.redis_repos import PendingUploadRepository
    client,tenant=client_tenant
    review=upload(client,tenant)
    core=AsyncMock()
    core.analyze.side_effect=AIServiceError('Gemini daily quota reached.',429,'AI_DAILY_QUOTA')
    monkeypatch.setattr(api,'get_sector_core',lambda *args:core)
    response=client.post('/api/v1/upload/confirm-mapping',json=confirmation(tenant,review))
    assert response.status_code==429,response.text
    assert run(PendingUploadRepository().get(tenant,review['upload_session_id'])) is not None
    assert run(get_analysis_repo().latest(tenant)) is None


@pytest.mark.parametrize('raw,expected',[('$49.99',49.99),('$199.00',199),('$1,249.99',1249.99),('LKR 250.00',250),('(£15.50)',-15.5),(1e20,1e20)])
def test_canonical_amount_parses_currency(raw,expected):
    from churn_platform.infrastructure.parsers.metric_context import currency_number
    assert currency_number(raw)==expected


def test_custom_telemetry_keeps_extreme_values_outside_three_samples():
    from churn_platform.infrastructure.parsers.metric_context import metric_evidence
    result=metric_evidence(pd.Series(['-85dBm','-90dBm','-95dBm','-110dBm']))
    assert result['sample_values']==['-85dBm','-90dBm','-95dBm']
    assert result['numeric_summary']['minimum_observation']=='-110dBm'
    assert result['numeric_summary']['unit']=='dBm'
    assert result['observation_count']==4
    assert 'numeric_summary' not in metric_evidence(pd.Series(['12ms','3s']))


def test_review_option_can_override_confident_and_remembered_mappings(client_tenant):
    client,tenant=client_tenant
    raw=b'customer_id,plan_tier\nu1,Tier 1\n'
    def submit(review):
        return client.post('/api/v1/upload/analyze',data={'tenant_id':tenant,'engine':'system','review_mapping':str(review).lower()},files={'files':('users.csv',raw)})
    review=submit(True).json()
    assert review['requires_human_review']
    assert all(c['confidence']>=.8 for c in review['schema_mapping']['tables'][0]['columns'])
    response=client.post('/api/v1/upload/confirm-mapping',json={'tenant_id':tenant,'upload_session_id':review['upload_session_id'],
        'mappings':[{'file_name':'users.csv','source_column':'customer_id','canonical_role':'CUSTOMER_ID'},
                    {'file_name':'users.csv','source_column':'plan_tier','canonical_role':'CUSTOM','custom_label':'Priority SLA Tier'}]})
    assert response.status_code==200,response.text
    assert submit(False).json()['requires_human_review'] is False
    second=submit(True).json()
    assert second['requires_human_review']
    assert second['schema_mapping']['tables'][0]['columns'][1]['custom_label']=='Priority SLA Tier'


@pytest.mark.parametrize('sector',['SaaS','Telecom'])
def test_user_scenarios_resume_with_semantic_names_and_supporting_evidence(client_tenant,monkeypatch,sector):
    from churn_platform.presentation.api import dependencies
    from churn_platform.config import Settings
    from churn_platform.infrastructure.ai.cores.telecom_core import TelecomCore
    client,_=client_tenant
    tenant=client.post('/api/v1/tenants/',json={'name':'Semantic scenario','sector':sector}).json()['tenant_id']
    registry=dependencies.build_engine_registry(Settings(_env_file=None,gemini_api_key='test-key'))
    gateway=AsyncMock()
    calls=[]
    if sector=='SaaS':
        raw=b'customer_id,timestamp,tx_amt_net_99,sys_stat_code,usr_rt_val_3,latency_ms\nu1,2025-05-30,$49.99,x8,Tier 1,300\nu1,2025-05-31,$199.00,y2,Tier 1,350\n'
        roles={'customer_id':'CUSTOMER_ID','timestamp':'TIMESTAMP','tx_amt_net_99':'UNKNOWN','sys_stat_code':'UNKNOWN','usr_rt_val_3':'SUBSCRIPTION_PLAN','latency_ms':'ATTRIBUTE'}
        label,source='Priority SLA Tier','usr_rt_val_3'
        table_role='TRANSACTIONAL'
    else:
        raw=b'msisdn,timestamp,call_dur_sec,tw_pwr_drp,dropped_calls_30d\nu1,2025-05-28,45,-85dBm,1\nu1,2025-05-29,30,-90dBm,2\nu1,2025-05-30,20,-95dBm,4\nu1,2025-05-31,10,-110dBm,12\n'
        roles={'msisdn':'CUSTOMER_ID','timestamp':'TIMESTAMP','call_dur_sec':'USAGE_ACTIVITY','tw_pwr_drp':'UNKNOWN','dropped_calls_30d':'ATTRIBUTE'}
        label,source='Cell Tower Signal Strength','tw_pwr_drp'
        table_role='TIME_SERIES_EVENT'
    async def respond(system_prompt,user_prompt):
        calls.append((system_prompt,user_prompt))
        if 'Data Engineer' in system_prompt:
            return {'tables':[{'file_name':'export.csv','role':table_role,'primary_entity_key':next(iter(roles)),'timestamp_column':'timestamp',
                'columns':[{'source_column':name,'canonical_role':role,'confidence':.3 if role=='UNKNOWN' else .98} for name,role in roles.items()]}]}
        return {'predictions':[{'entity_id':'u1','churn_prediction':{'churn_probability':.85,'primary_drivers':[label], 'root_cause':label},
            'retention_playbook':{'action_type':'REVIEW','action_payload':'Investigate the supplied evidence','channel':'Email'}}]}
    gateway.generate_json.side_effect=respond
    registry.gateways['ai']=gateway
    registry.resolvers['ai']=AISchemaResolver(gateway)
    registry.cores['ai'][sector.lower()]=SaasCore(gateway) if sector=='SaaS' else TelecomCore(gateway)
    monkeypatch.setattr(dependencies,'_REGISTRY',registry)
    def submit():
        r=client.post('/api/v1/upload/analyze',data={'tenant_id':tenant,'engine':'ai'},files={'files':('export.csv',raw)})
        assert r.status_code==200,r.text
        return r.json()
    review=submit()
    assert review['requires_human_review']
    mappings=[]
    for name,role in roles.items():
        choice='CUSTOM' if name==source else 'TRANSACTION_AMOUNT' if name=='tx_amt_net_99' else 'NOISE_IGNORE' if name=='sys_stat_code' else role
        mappings.append({'file_name':'export.csv','source_column':name,'canonical_role':choice,'custom_label':label if choice=='CUSTOM' else None})
    r=client.post('/api/v1/upload/confirm-mapping',json={'tenant_id':tenant,'upload_session_id':review['upload_session_id'],'mappings':mappings})
    assert r.status_code==200,r.text
    prompt=calls[-1][1]
    assert label in prompt and source not in prompt
    assert 'sys_stat_code' not in prompt
    features=json.loads(prompt.split('\n',1)[1])[0]['features']
    if sector=='SaaS':
        assert features['export_amount_total']==pytest.approx(248.99)
        assert features['custom_metrics']['export.csv'][label]==['Tier 1']
        assert features['additional_attributes']['export.csv']['latency_ms']['numeric_summary']['max']==350
    else:
        telemetry=features['custom_metric_evidence']['export.csv'][label]
        assert telemetry['numeric_summary']['minimum_observation']=='-110dBm'
        assert features['additional_attributes']['export.csv']['usage_activity']['numeric_summary']['min']==10
        assert features['additional_attributes']['export.csv']['dropped_calls_30d']['numeric_summary']['max']==12
    assert submit()['requires_human_review'] is False
    assert sum('Data Engineer' in system for system,_ in calls)==1

@pytest.mark.parametrize('remembered',[True,False])
def test_invalid_amount_role_returns_to_review_and_cannot_be_confirmed(client_tenant,monkeypatch,remembered):
    from churn_platform.presentation.api.v1 import upload as api
    client,tenant=client_tenant
    raw=b'account_id,timestamp,swipe_id,amount\na1,2025-05-30,1,$49.99\na1,2025-05-30,2,$20.00\na1,2025-05-30,3,$15.00\na1,2025-05-30,SWIPE004,$9.00\n'
    bad=SchemaMapping(primary_entity_key='account_id',tables=[TableClassification(file_name='card_swipes.csv',role='TRANSACTIONAL',primary_entity_key='account_id',timestamp_column='timestamp',columns=[column('account_id','CUSTOMER_ID'),column('timestamp','TIMESTAMP'),column('swipe_id','TRANSACTION_AMOUNT'),column('amount','ATTRIBUTE')])])
    memory=TenantSchemaMemoryRepository()
    if remembered:
        run(memory.save(tenant,bad))
    else:
        resolver=AsyncMock(); resolver.resolve.return_value=bad
        monkeypatch.setattr(api,'get_schema_resolver',lambda *args:resolver)
    response=client.post('/api/v1/upload/analyze',data={'tenant_id':tenant,'engine':'system'},files={'files':('card_swipes.csv',raw)})
    assert response.status_code==200,response.text
    review=response.json()
    assert review['requires_human_review']
    offending=review['schema_mapping']['tables'][0]['columns'][2]
    assert offending['confidence']==0 and 'SWIPE004' in offending['sample_values']
    choices=[{'file_name':'card_swipes.csv','source_column':c.source_column,'canonical_role':c.canonical_role} for c in bad.tables[0].columns]
    payload={'tenant_id':tenant,'upload_session_id':review['upload_session_id'],'mappings':choices}
    before=run(memory.get(tenant,['card_swipes.csv']))
    rejected=client.post('/api/v1/upload/confirm-mapping',json=payload)
    assert rejected.status_code==400 and 'swipe_id' in rejected.json()['detail']
    assert run(memory.get(tenant,['card_swipes.csv']))==before
    choices[2]['canonical_role']='ATTRIBUTE'
    choices[3]['canonical_role']='TRANSACTION_AMOUNT'
    fixed=client.post('/api/v1/upload/confirm-mapping',json=payload)
    assert fixed.status_code==200,fixed.text
    saved=run(memory.get(tenant,['card_swipes.csv']))['card_swipes.csv']['columns']
    assert saved['swipe_id']['canonical_role']=='ATTRIBUTE'
    assert saved['amount']['canonical_role']=='TRANSACTION_AMOUNT'
