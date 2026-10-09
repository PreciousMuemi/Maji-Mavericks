"""Small durable SQLite store. Replace via dependency injection for shared hosting."""
import json
import hashlib
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4
from .schemas import DocumentExtraction, ParsedDocument, AIAnalysis
from .insurance_schemas import InsuranceDocumentResult


class AssessmentNotFound(LookupError):
    pass


class AssessmentStore:
    def __init__(self, directory: str):
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
        database = path / "assessments.sqlite3"
        self.connection = sqlite3.connect(database, check_same_thread=False)
        database.chmod(0o600)
        self.lock = threading.RLock()
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("CREATE TABLE IF NOT EXISTS assessments (id TEXT PRIMARY KEY, documents TEXT NOT NULL, extractions TEXT NOT NULL, analysis TEXT)")
        self.connection.execute("CREATE TABLE IF NOT EXISTS insurance_extractions (assessment_id TEXT NOT NULL, document_id TEXT NOT NULL, result TEXT NOT NULL, PRIMARY KEY (assessment_id, document_id))")
        self.connection.execute('CREATE TABLE IF NOT EXISTS assessment_owners (assessment_id TEXT PRIMARY KEY, owner_id TEXT NOT NULL)')
        self.connection.execute('CREATE TABLE IF NOT EXISTS chat_turns (assessment_id TEXT NOT NULL, conversation_id TEXT NOT NULL, turn_id INTEGER PRIMARY KEY AUTOINCREMENT, message TEXT NOT NULL, answer TEXT NOT NULL)')
        self.connection.execute('CREATE TABLE IF NOT EXISTS dashboard_inputs (assessment_id TEXT PRIMARY KEY, fingerprint TEXT NOT NULL, mapping TEXT)')
        self.connection.execute('CREATE TABLE IF NOT EXISTS underwriting_intelligence (assessment_id TEXT PRIMARY KEY, fingerprint TEXT NOT NULL, result TEXT NOT NULL)')
        self.connection.commit()

    def close(self):
        self.connection.close()

    @contextmanager
    def _snapshot_transaction(self):
        with self.lock:
            self.connection.execute('SAVEPOINT dashboard_read')
            try:
                yield
            finally:
                self.connection.execute('RELEASE dashboard_read')

    def dashboard_snapshot(self, assessment_id: str) -> dict:
        """Read all local dashboard inputs atomically; stale derived data is excluded."""
        from .exposure_schemas import ExposureMappingResult
        from .intelligence_schemas import UnderwritingAnalysis
        with self._snapshot_transaction():
            documents, extractions, _ = self._read(assessment_id)
            rich = [json.loads(row[0]) for row in self.connection.execute(
                'SELECT result FROM insurance_extractions WHERE assessment_id = ? ORDER BY document_id', (assessment_id,)).fetchall()]
            payload = {'documents': documents, 'extractions': extractions, 'insurance': rich}
            source_hash = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
            mapping_row = self.connection.execute('SELECT fingerprint, mapping FROM dashboard_inputs WHERE assessment_id = ?', (assessment_id,)).fetchone()
            mapping = json.loads(mapping_row[1]) if mapping_row and mapping_row[0] == source_hash else None
            fingerprint = hashlib.sha256(json.dumps([source_hash, mapping], sort_keys=True).encode()).hexdigest()
            ai_row = self.connection.execute('SELECT fingerprint, result FROM underwriting_intelligence WHERE assessment_id = ?', (assessment_id,)).fetchone()
            intelligence = UnderwritingAnalysis.model_validate_json(ai_row[1]) if ai_row and ai_row[0] == fingerprint else None
        records = [DocumentExtraction.model_validate(value) for value in extractions.values()]
        return {'documents': {key: ParsedDocument.model_validate(value) for key, value in documents.items()},
                'insurance': [InsuranceDocumentResult.model_validate(value) for value in rich],
                'extraction': DocumentExtraction(assets=[a for r in records for a in r.assets], claims=[c for r in records for c in r.claims], warnings=[w for r in records for w in r.warnings]),
                'mapping': ExposureMappingResult.model_validate(mapping) if mapping else None,
                'intelligence': intelligence, 'fingerprint': fingerprint, 'source_hash': source_hash}

    def save_exposure_mapping(self, assessment_id: str, mapping, *, source_hash: str):
        """Host integration persists a reviewed mapping against its exact source revision."""
        from .exposure_mapping import validate_exposure_records
        mapping = validate_exposure_records(mapping)
        with self.lock, self.connection:
            self.connection.execute('BEGIN IMMEDIATE')
            snapshot = self.dashboard_snapshot(assessment_id)
            if snapshot['source_hash'] != source_hash:
                raise ValueError('Assessment changed; remap current inputs')
            self.connection.execute('INSERT OR REPLACE INTO dashboard_inputs VALUES (?, ?, ?)',
                                    (assessment_id, source_hash, mapping.model_dump_json()))

    def save_intelligence(self, assessment_id: str, result, *, fingerprint: str) -> bool:
        """Do not publish a briefing if its source inputs changed during generation."""
        from .intelligence_schemas import UnderwritingAnalysis
        result = UnderwritingAnalysis.model_validate_json(result.model_dump_json())
        if any(c.assessment_id != assessment_id for c in result.citations):
            raise ValueError('Intelligence belongs to another assessment')
        with self.lock, self.connection:
            self.connection.execute('BEGIN IMMEDIATE')
            if self.dashboard_snapshot(assessment_id)['fingerprint'] != fingerprint:
                return False
            self.connection.execute('INSERT OR REPLACE INTO underwriting_intelligence VALUES (?, ?, ?)',
                                    (assessment_id, fingerprint, result.model_dump_json()))
        return True

    def create(self, owner_id: str | None = None) -> str:
        assessment_id = str(uuid4())
        with self.lock, self.connection:
            self.connection.execute("INSERT INTO assessments VALUES (?, '{}', '{}', NULL)", (assessment_id,))
            if owner_id is not None:
                self.connection.execute('INSERT INTO assessment_owners VALUES (?, ?)', (assessment_id, owner_id))
        return assessment_id

    def owner(self, assessment_id: str) -> str | None:
        with self.lock:
            self._read(assessment_id)
            row = self.connection.execute('SELECT owner_id FROM assessment_owners WHERE assessment_id = ?', (assessment_id,)).fetchone()
        return row[0] if row else None

    def conversation(self, assessment_id: str, conversation_id: str, limit: int = 8) -> list[dict]:
        with self.lock:
            self._read(assessment_id)
            rows = self.connection.execute('SELECT message, answer FROM chat_turns WHERE assessment_id = ? AND conversation_id = ? ORDER BY turn_id DESC LIMIT ?', (assessment_id, conversation_id, limit)).fetchall()
        return [item for message, answer in reversed(rows) for item in ({'role': 'user', 'content': message}, {'role': 'assistant', 'content': answer})]

    def save_chat_turn(self, assessment_id: str, conversation_id: str, message: str, answer: str):
        with self.lock, self.connection:
            self._read(assessment_id)
            self.connection.execute('INSERT INTO chat_turns (assessment_id, conversation_id, message, answer) VALUES (?, ?, ?, ?)', (assessment_id, conversation_id, message, answer))

    def insurance_results(self, assessment_id: str) -> list[InsuranceDocumentResult]:
        with self.lock:
            self._read(assessment_id)
            rows = self.connection.execute('SELECT result FROM insurance_extractions WHERE assessment_id = ?', (assessment_id,)).fetchall()
        return [InsuranceDocumentResult.model_validate_json(row[0]) for row in rows]

    def _read(self, assessment_id):
        row = self.connection.execute("SELECT documents, extractions, analysis FROM assessments WHERE id = ?", (assessment_id,)).fetchone()
        if row is None:
            raise AssessmentNotFound("Assessment does not exist")
        return json.loads(row[0]), json.loads(row[1]), AIAnalysis.model_validate_json(row[2]) if row[2] else None

    def documents(self, assessment_id) -> dict[str, ParsedDocument]:
        with self.lock:
            documents, _, _ = self._read(assessment_id)
        return {key: ParsedDocument.model_validate(value) for key, value in documents.items()}

    def add_document(self, assessment_id: str, document: ParsedDocument):
        with self.lock, self.connection:
            documents, extractions, _ = self._read(assessment_id)
            replacement = document.model_dump(mode='json')
            previous = documents.get(document.document_id)
            if previous == replacement:
                return
            if previous is not None:
                extractions.pop(document.document_id, None)
                self.connection.execute('DELETE FROM insurance_extractions WHERE assessment_id = ? AND document_id = ?', (assessment_id, document.document_id))
            documents[document.document_id] = replacement
            self.connection.execute("UPDATE assessments SET documents = ?, extractions = ?, analysis = NULL WHERE id = ?", (json.dumps(documents), json.dumps(extractions), assessment_id))

    def save_extractions(self, assessment_id, values: dict[str, DocumentExtraction]):
        with self.lock, self.connection:
            documents, extractions, _ = self._read(assessment_id)
            if not set(values).issubset(documents):
                raise ValueError("Unknown document")
            extractions.update({key: value.model_dump(mode="json") for key, value in values.items()})
            self.connection.execute("UPDATE assessments SET extractions = ?, analysis = NULL WHERE id = ?", (json.dumps(extractions), assessment_id))

    def extraction(self, assessment_id) -> DocumentExtraction:
        with self.lock:
            _, values, _ = self._read(assessment_id)
        records = [DocumentExtraction.model_validate(value) for value in values.values()]
        return DocumentExtraction(assets=[a for r in records for a in r.assets], claims=[c for r in records for c in r.claims], warnings=[w for r in records for w in r.warnings])

    def analysis(self, assessment_id) -> AIAnalysis | None:
        with self.lock:
            return self._read(assessment_id)[2]

    def save_analysis(self, assessment_id, value: AIAnalysis, *, fingerprint: str | None = None) -> bool:
        with self.lock, self.connection:
            if fingerprint is not None:
                self.connection.execute('BEGIN IMMEDIATE')
                if self.dashboard_snapshot(assessment_id)['fingerprint'] != fingerprint:
                    return False
            self._read(assessment_id)
            self.connection.execute("UPDATE assessments SET analysis = ? WHERE id = ?", (value.model_dump_json(), assessment_id))
        return True

    def save_insurance_extraction(self, assessment_id: str, value: InsuranceDocumentResult):
        with self.lock, self.connection:
            documents, _, _ = self._read(assessment_id)
            if value.document_id not in documents:
                raise ValueError('Unknown document')
            self.connection.execute('INSERT OR REPLACE INTO insurance_extractions VALUES (?, ?, ?)', (assessment_id, value.document_id, value.model_dump_json()))
            self.connection.execute('UPDATE assessments SET analysis = NULL WHERE id = ?', (assessment_id,))

    def insurance_extraction(self, assessment_id: str, document_id: str) -> InsuranceDocumentResult | None:
        with self.lock:
            self._read(assessment_id)
            row = self.connection.execute('SELECT result FROM insurance_extractions WHERE assessment_id = ? AND document_id = ?', (assessment_id, document_id)).fetchone()
        return InsuranceDocumentResult.model_validate_json(row[0]) if row else None
