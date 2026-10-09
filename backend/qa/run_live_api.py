"""Real Uvicorn/HTTP QA. Uses configured providers only, never extraction/model doubles.

Run from backend: .venv/bin/python -m qa.run_live_api
Exit 2 means the critical end-to-end path is blocked or failed, not demo-ready.
"""
import argparse
import asyncio
import json
import os
import secrets
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pymupdf
from docx import Document
from openpyxl import Workbook

from ai.config import Settings
from ai.insurance_schemas import InsuranceDocumentResult
from ai.intelligence_schemas import UnderwritingAnalysis
from ai.schemas import ResponseEnvelope
from ai.store import AssessmentStore
from ai.model_backend import UnavailableUnderwritingBackend
from ai.exposure_schemas import ExposureContract
from api.ai_routes import build_services
from app.schemas.dashboard import DashboardResponse

ROOT = Path(__file__).resolve().parents[2]
QA = Path(__file__).resolve().parent


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def generate_documents():
    """Expected outcomes are generated independently; never injected into extraction."""
    token = secrets.token_hex(4)
    name = 'QA Sample Estate ' + token
    records = []
    for index in range(3):
        records.append({'asset_id': 'QA-' + token + '-' + str(index + 1), 'name': 'QA Building ' + str(index + 1) + ' ' + token,
                        'lat': round(0.05 + index / 1000, 6), 'lon': round(34.1 + index / 1000, 6),
                        'construction': 'reinforced masonry', 'floor_area_m2': 120 * (index + 1),
                        'insured_value_kes': (secrets.randbelow(90) + 10) * 10000})
    total = sum(r['insured_value_kes'] for r in records)
    book = Workbook()
    assets = book.active
    assets.title = 'Buildings'
    headers = list(records[0])
    assets.append(headers)
    for row in records:
        assets.append([row[h] for h in headers])
    facts = book.create_sheet('Insured and terms')
    for row in [('Client name', name), ('Industry', 'property storage'), ('Currency', 'KES'),
                ('Property sum insured (only the three buildings; mutually exclusive values)', total),
                ('Requested flood limit (proposed)', total // 2), ('Flood deductible (proposed)', 'KES 10000'),
                ('Facility location', 'Synthetic property schedule for QA; actual raster coverage is unverified'),
                ('Flood history', 'Not provided'), ('Building condition', 'Not provided')]:
        facts.append(row)
    schedule = QA / 'fixtures' / 'generated_property_schedule.xlsx'
    book.save(schedule)
    expected = {'fixture_only': True, 'insured_name': name, 'assets': records, 'property_sum_insured_kes': total,
                'requested_flood_limit_kes': total // 2, 'historical_claims': None, 'risk_level': None,
                'construction_class_applicability': 'unverified', 'model_losses': None}
    write_json(QA / 'fixtures' / 'generated_property_schedule.expected.json', expected)
    unfamiliar_name = 'QA Unfamiliar Ceramic Works ' + token
    offer = Document()
    offer.add_heading('Unfamiliar industrial insurance submission — generated QA fixture', 0)
    for line in [f'Insured entity: {unfamiliar_name}', 'Facility location: Not supplied',
                 'Coordinates: Not supplied. Do not infer coordinates from a company name.',
                 'Industry: ceramics manufacturing', f'Property sum insured: KES {total}',
                 f'Requested flood limit: KES {total // 2} (broker proposal)',
                 'Machinery insured value: KES 120000. Its relationship with property sum insured is not stated.',
                 'Warehouse East floor area: 240 m2; reinforced masonry; structure insured value not provided.',
                 'Inspection on 2025-01-01: Warehouse East structural condition is poor.',
                 'Same inspection on 2025-01-01: Warehouse East structural condition is good.',
                 'Historical flood on 2024-04-11: observed water depth 0.3 m; claimed KES 1500; settled KES 1000.',
                 'Broker recommends drainage inspection; this is a broker proposal, not an AI finding.']:
        offer.add_paragraph(line)
    unfamiliar = QA / 'fixtures' / 'unfamiliar_insurance_offer.docx'
    offer.save(unfamiliar)
    write_json(QA / 'fixtures' / 'unfamiliar_insurance_offer.expected.json',
               {'fixture_only': True, 'insured_name': unfamiliar_name, 'property_sum_insured_kes': total,
                'requested_flood_limit_kes': total // 2, 'building_value': None, 'coordinates': None,
                'historical_claimed_kes': 1500, 'historical_settled_kes': 1000, 'reported_depth_m': '0.3',
                'contradiction_requires_review': True, 'industrial_model_losses': None})
    return schedule, unfamiliar, expected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=0)
    args = parser.parse_args()
    settings = Settings()
    configured_llm = bool(settings.openai_api_key if settings.llm_provider == 'openai' else settings.gemini_api_key)
    schedule, unfamiliar, expected = generate_documents()
    pdf = ROOT / 'OFFER_NZOIA_GRAIN_PROCESSING.pdf'
    csv_candidates = [p for p in ROOT.glob('exposure_nzoia_synthetic*.csv') if p.is_file()]
    if not pdf.is_file() or len(csv_candidates) != 1:
        raise SystemExit('Supplied files must be present and unambiguous; do not substitute fixtures')
    exposure_csv = csv_candidates[0]
    storage = ROOT / 'backend' / '.nzoia-data' / ('qa-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '-' + secrets.token_hex(3))
    storage.mkdir(parents=True, mode=0o700)
    probe = build_services(settings.model_copy(update={'storage_directory': str(storage / 'preflight')}))
    model_configured = not isinstance(probe.model_backend, UnavailableUnderwritingBackend)
    probe.store.close()
    close = getattr(probe.extraction.llm, 'aclose', None)
    if close:
        asyncio.run(close())
    rasters = [p for p in ROOT.rglob('*') if p.suffix.lower() in {'.tif', '.tiff'} and '.venv' not in p.parts and p.is_file()]
    contract_path = os.getenv('NZOIA_EXPOSURE_CONTRACT_PATH')
    try:
        verified_contract = bool(contract_path and ExposureContract.model_validate_json(Path(contract_path).read_text()).verified)
    except (OSError, ValueError):
        verified_contract = False
    response_dir = storage / 'responses'
    response_dir.mkdir(mode=0o700)
    api_token = settings.api_token.get_secret_value() if settings.api_token else secrets.token_urlsafe(32)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', args.port))
        port = sock.getsockname()[1]
    env = dict(os.environ, NZOIA_STORAGE_DIRECTORY=str(storage), NZOIA_API_TOKEN=api_token)
    command = [sys.executable, '-m', 'uvicorn', 'app.main:app', '--host', '127.0.0.1', '--port', str(port), '--no-access-log']
    log_path = storage / 'server.log'
    cases, assessment_ids, extracted = [], {}, {}

    def record(identifier, outcome, detail, *, layer='real_http', http_status=None):
        cases.append({'id': identifier, 'outcome': outcome, 'layer': layer, 'http_status': http_status, 'detail': detail})

    with log_path.open('w') as server_log:
        process = subprocess.Popen(command, cwd=ROOT / 'backend', env=env, stdout=server_log, stderr=server_log)
        try:
            with httpx.Client(base_url=f'http://127.0.0.1:{port}', headers={'Authorization': 'Bearer ' + api_token}, timeout=180) as client:
                for _ in range(100):
                    if process.poll() is not None:
                        raise RuntimeError('Backend exited during startup; inspect private server log')
                    try:
                        if client.get('/openapi.json').status_code == 200:
                            break
                    except httpx.TransportError:
                        pass
                    time.sleep(0.1)
                else:
                    raise RuntimeError('Backend did not start')
                record('01_backend_start', 'passed', 'Real Uvicorn process responds over loopback HTTP; isolated durable SQLite storage and authenticated API.')

                def request(identifier, method, path, **kwargs):
                    response = client.request(method, path, **kwargs)
                    try:
                        payload = response.json()
                    except ValueError:
                        payload = {'invalid_json_response': True}
                    write_json(response_dir / (identifier + '.json'), {'http_status': response.status_code, 'response': payload})
                    return response, payload

                for kind, path in [('offer_pdf', pdf), ('exposure_csv', exposure_csv), ('generated_schedule', schedule), ('unfamiliar_offer', unfamiliar)]:
                    response, payload = request(kind + '_create', 'POST', '/api/ai/assessments')
                    if response.status_code != 201:
                        record(kind + '_upload', 'failed', 'Assessment creation failed', http_status=response.status_code)
                        continue
                    assessment = payload['assessment_id']
                    assessment_ids[kind] = assessment
                    base = '/api/ai/assessments/' + assessment
                    with path.open('rb') as stream:
                        response, payload = request(kind + '_upload', 'POST', base + '/documents', files={'file': (path.name, stream)})
                    good = response.status_code == 201 and bool(payload.get('data', {}).get('document_id'))
                    record(kind + '_upload', 'passed' if good else 'failed', f'Actual multipart upload of {path.name}', http_status=response.status_code)
                    if not good:
                        continue
                    document_id = payload['data']['document_id']
                    # Read persisted parser evidence solely to VERIFY the actual upload.
                    # No extracted fact or provider output is written by this runner.
                    store = AssessmentStore(str(storage))
                    try:
                        document = store.documents(assessment)[document_id]
                    finally:
                        store.close()
                    if kind == 'offer_pdf':
                        with pymupdf.open(pdf) as source:
                            expected_pages = set(range(1, len(source) + 1))
                        actual_pages = {s.page_number for s in document.segments if s.kind == 'text'}
                        record('03_pdf_all_pages', 'passed' if actual_pages == expected_pages else 'failed', f'Expected {len(expected_pages)} pages; stored text on {len(actual_pages)} pages; missing pages {sorted(expected_pages - actual_pages)}.', layer='persisted_api_upload_verification')
                    else:
                        record(kind + '_parsed_rows', 'passed' if document.segments else 'failed', f'{len(document.segments)} actual parsed source segments.', layer='persisted_api_upload_verification')
                    if kind == 'generated_schedule':
                        building_rows = [json.loads(s.text) for s in document.segments if s.locator.startswith('sheet:Buildings:row:')][1:]
                        expected_rows = [[str(row[field]) for field in row] for row in expected['assets']]
                        record('generated_source_cells', 'passed' if building_rows == expected_rows else 'failed', 'Actual uploaded spreadsheet retains all generated building names, coordinates, areas and values; this verifies parser cells, not AI extraction.', layer='persisted_api_upload_verification')
                    response, payload = request(kind + '_insurance_extract', 'POST', base + f'/documents/{document_id}/insurance-extract')
                    if response.status_code == 200:
                        try:
                            result = InsuranceDocumentResult.model_validate(payload['data'])
                            extracted[kind] = result
                            record(kind + '_ai_extraction', 'passed', 'Actual configured LLM returned strict, source-validated insurance facts.', http_status=200)
                        except Exception:
                            record(kind + '_ai_extraction', 'failed', 'Provider/API output did not satisfy the insurance schema', http_status=200)
                    else:
                        record(kind + '_ai_extraction', 'blocked' if response.status_code == 503 and not configured_llm else 'failed', 'AI extraction did not run: ' + (payload.get('errors') or [{'code': 'unknown'}])[0]['code'], http_status=response.status_code)
                    response, payload = request(kind + '_canonical_extract', 'POST', base + '/extract')
                    record(kind + '_canonical_extraction', 'blocked' if response.status_code == 503 and not configured_llm else 'passed' if response.status_code == 200 else 'failed', 'Actual canonical AI extraction request; no expected outcomes were injected.', http_status=response.status_code)
                    response, payload = request(kind + '_dashboard', 'GET', '/api/assessments/' + assessment + '/dashboard')
                    try:
                        dashboard = ResponseEnvelope[DashboardResponse].model_validate(payload).data
                        good = response.status_code == 200 and dashboard is not None
                        if kind not in extracted:
                            good &= dashboard.status != 'calculation_completed' and not any(c.basis == 'modelled' for c in dashboard.charts)
                            good &= all(k.value is None for k in dashboard.kpis if k.basis != 'workflow')
                        record(kind + '_partial_dashboard', 'passed' if good else 'failed', 'Strict dashboard JSON; unavailable figures remain null and no modelled chart or completed-calculation status appears.', http_status=response.status_code)
                    except Exception:
                        record(kind + '_partial_dashboard', 'failed', 'Dashboard response did not validate', http_status=response.status_code)

                # Do not mark source-grounded field checks passed without actual extraction.
                for identifier, detail in [('04_insured_entity', 'Correct insured identity'), ('05_financial_terms', 'Property, inventory and insurance terms'), ('06_assets', 'Building-specific assets and values'), ('07_historical_claims', 'Event dates, claimed/settled amounts and observed depths'), ('08_contradictions', 'Missing-field and contradiction detection')]:
                    record(identifier, 'blocked', detail + ': no actual offer LLM extraction is available; human/text observations are not substituted as AI output.')
                if 'generated_schedule' in extracted:
                    facts = extracted['generated_schedule'].facts
                    good = facts.insured.client_name.value == expected['insured_name'] and facts.financial_exposure.property_sum_insured.value is not None and str(facts.financial_exposure.property_sum_insured.value.amount) == str(expected['property_sum_insured_kes'])
                    good &= len(facts.assets) == len(expected['assets'])
                    record('generated_expected_outcomes', 'passed' if good else 'failed', 'Actual provider extraction compared with independently generated identity, property sum and asset count.')
                else:
                    record('generated_expected_outcomes', 'blocked', 'Expected outcomes saved independently; cannot verify them without real AI extraction.')
                record('09_hazard_extent', 'blocked', 'No flood raster, nonempty hazard catalog or verified extracted site coordinate is supplied. No guessed coverage test was substituted.')
                assessment = assessment_ids['offer_pdf']
                base = '/api/ai/assessments/' + assessment
                response, payload = request('10_loss_calculation', 'POST', base + '/analyze')
                record('10_loss_calculation', 'blocked', 'No supported numerical flood calculation can execute; actual endpoint returned ' + str(response.status_code), http_status=response.status_code)
                record('loss_failure_no_fake_success', 'passed' if response.status_code == 503 and payload.get('status') == 'error' and payload.get('data') is None else 'failed', 'Unavailable model fails explicitly without accepted numerical results.', http_status=response.status_code)
                response, payload = request('12_underwriting_analysis', 'POST', base + '/underwriting-analysis')
                try:
                    analysis = UnderwritingAnalysis.model_validate(payload['data'])
                    limitation_codes = {l.code for l in analysis.limitations}
                    safe = response.status_code == 200 and payload['status'] == 'partial' and 'ai_narrative_unavailable' in limitation_codes and not analysis.model_findings and analysis.risk_level.value is None
                except Exception:
                    safe = False
                record('12_ai_summary', 'blocked', 'Actual endpoint provides a labelled deterministic partial fallback, not an AI-generated interpretation.', http_status=response.status_code)
                record('analysis_fallback_honest', 'passed' if safe else 'failed', 'Fallback is partial, has explicit AI/model limitations and no predicted loss or risk score.')
                for index, question in enumerate(['What information is missing from this insurance offer?', 'Run the 100-year flood scenario.', 'Which buildings contribute the largest losses?', 'Compare the 100-year and 500-year flood scenarios.']):
                    response, payload = request('13_chat_' + str(index), 'POST', base + '/chat', json={'message': question})
                    record('13_chat_' + str(index), 'blocked' if response.status_code == 503 and not configured_llm else 'failed', 'Follow-up attempted over HTTP; live tool selection cannot execute without a configured provider.', http_status=response.status_code)
                    record('chat_failure_' + str(index), 'passed' if response.status_code == 503 and payload.get('data') is None else 'failed', 'No fabricated answer is returned when chat dependency is unavailable.')
                response, payload = request('14_scenario', 'POST', base + '/scenarios', json={'name': 'QA floor elevation review', 'parameters': {'return_periods': [100], 'floor_elevation_delta_m': 0.5}})
                record('14_supported_what_if', 'blocked', 'Scenario parameter support and calibrated execution are unavailable; attempted request returns ' + str(response.status_code), http_status=response.status_code)
                record('scenario_failure_no_fake_success', 'passed' if response.status_code == 503 and payload.get('data') is None else 'failed', 'No unexecuted scenario results appear.')
                response, findings = request('15_findings', 'GET', base + '/findings')
                response2, dashboard = request('15_dashboard', 'GET', '/api/assessments/' + assessment + '/dashboard')
                safe = findings.get('data', {}).get('model_results') is None and all(k['value'] is None for k in dashboard.get('data', {}).get('kpis', []) if k['basis'] != 'workflow')
                record('15_financial_consistency_unavailable', 'passed' if safe else 'failed', 'Unavailable financial/model values stay unavailable across findings and dashboard; no zeros or claims-as-projections.')
                record('15_extracted_financial_consistency', 'blocked', 'Actual numeric reconciliation requires successful AI extraction and model results.')
                empty_response, empty = request('empty_create', 'POST', '/api/ai/assessments')
                empty_base = '/api/ai/assessments/' + empty['assessment_id']
                response, payload = request('empty_extract', 'POST', empty_base + '/extract')
                record('missing_documents', 'passed' if response.status_code == 422 else 'failed', 'Extraction rejects missing documents.', http_status=response.status_code)
                invalid = [('unsupported.exe', b'invalid'), ('empty.pdf', b''), ('corrupt.pdf', b'%PDF-1.7\ncorrupted'), ('spoofed.xlsx', b'name,value\na,1'), ('invalid.csv', b'\xff\x00'), ('oversized.pdf', b'x' * (settings.max_upload_bytes + 1)), ('extra_cells.csv', b'loc_id,name\n001,Building A,extra\n'), ('duplicate_headers.csv', b'loc_id,loc_id\n001,002\n')]
                for index, (filename, content) in enumerate(invalid):
                    response, payload = request('invalid_upload_' + str(index), 'POST', empty_base + '/documents', files={'file': (filename, content)})
                    record('invalid_upload_' + str(index), 'passed' if response.status_code == 422 and payload.get('status') == 'error' else 'failed', filename + ' rejected through the actual multipart API.', http_status=response.status_code)
                for index, body in enumerate([{'message': ''}, {'message': 'test', 'unknown_field': 'private-input'}]):
                    response, payload = request('invalid_chat_' + str(index), 'POST', base + '/chat', json=body)
                    record('invalid_chat_' + str(index), 'passed' if response.status_code == 422 and 'private-input' not in response.text else 'failed', 'Invalid chat schema rejected with sanitized errors.', http_status=response.status_code)
                response, payload = request('scenario_identity_override', 'POST', base + '/scenarios', json={'name': 'Rejected identity override', 'parameters': {'nested': [{'assessment_id': 'PRIVATE-OTHER-ASSESSMENT'}]}})
                record('scenario_identity_override', 'passed' if response.status_code == 422 and 'PRIVATE-OTHER-ASSESSMENT' not in response.text else 'failed', 'Actual HTTP scenario rejects nested assessment identity overrides before backend access.', http_status=response.status_code)
                unauthorized = client.get('/api/assessments/' + assessment + '/dashboard', headers={'Authorization': 'Bearer invalid-token'})
                record('invalid_token', 'passed' if unauthorized.status_code == 401 else 'failed', 'Actual server rejects invalid bearer token.', http_status=unauthorized.status_code)
                unknown = client.get('/api/assessments/unknown-assessment/dashboard')
                record('unknown_assessment', 'passed' if unknown.status_code == 404 else 'failed', 'Unknown assessment returns a sanitized not-found envelope.', http_status=unknown.status_code)
                record('18_live_tool_selection', 'blocked', 'No live LLM tool selection or actual catastrophe function call succeeded; unit-double coverage is reported separately.')
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
    totals = {outcome: sum(c['outcome'] == outcome for c in cases) for outcome in ('passed', 'failed', 'blocked')}
    readiness = 'NOT_READY' if totals['failed'] or totals['blocked'] else 'READY'
    report = {'readiness': readiness, 'run_utc': datetime.now(timezone.utc).isoformat(), 'configuration': {'llm_provider': settings.llm_provider, 'llm_credentials_configured': configured_llm, 'hazard_rasters_present': bool(rasters), 'verified_vulnerability_contract_present': verified_contract, 'real_model_integration_configured': model_configured, 'test_doubles_used_in_http_run': False, 'api_authentication_enabled_for_qa': True},
              'files': {'insurance_offer': str(pdf.relative_to(ROOT)), 'exposure_schedule': str(exposure_csv.relative_to(ROOT)), 'generated_property_schedule': str(schedule.relative_to(ROOT)), 'unfamiliar_offer': str(unfamiliar.relative_to(ROOT))},
              'totals': totals, 'assessment_ids': assessment_ids, 'private_artifact_directory': str(storage), 'server_stopped': process.poll() is not None, 'cases': cases}
    write_json(QA / 'live_api_results.json', report)
    print(json.dumps({'readiness': report['readiness'], 'totals': totals, 'report': str(QA / 'live_api_results.json'), 'private_artifact_directory': str(storage), 'server_stopped': report['server_stopped']}))
    return 2 if readiness == 'NOT_READY' else 0


if __name__ == '__main__':
    raise SystemExit(main())
