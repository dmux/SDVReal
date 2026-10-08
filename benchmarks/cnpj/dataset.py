"""Build SDV data and metadata from the prepared CNPJ sample.

Two schema variants are available:

* ``core``: ``empresas`` -> ``estabelecimentos`` / ``socios`` / ``simples``. The domain codes
  (municipio, CNAE, ...) are categorical columns. 4 tables, depth 2: accepted by
  ``HMASynthesizer``, so both synthesizers can be compared.
* ``full``: the domain tables (``cnaes``, ``motivos``, ``municipios``, ``naturezas``,
  ``paises``, ``qualificacoes``) become parent tables referenced by foreign keys. 10 tables,
  depth 3, multiple foreign keys to the same parent: rejected by ``HMASynthesizer``.

Composite keys are not supported in SDV Community, so the establishment primary key is the
14 digit ``cnpj`` (``cnpj_basico`` + ``cnpj_ordem`` + ``cnpj_dv``).
"""

import os

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from sdv.metadata import Metadata

NUM_BUCKETS = 10_000
CATEGORICAL = {'sdtype': 'categorical'}
DATETIME = {'sdtype': 'datetime'}
UNKNOWN = {'sdtype': 'unknown'}

COLUMNS = {
    'empresas': {
        'cnpj_basico': {'sdtype': 'id', 'regex_format': '[0-9]{8}'},
        'razao_social': {'sdtype': 'company', 'pii': True},
        'natureza_juridica': CATEGORICAL,
        'qualificacao_responsavel': CATEGORICAL,
        'capital_social': {'sdtype': 'numerical'},
        'porte_empresa': CATEGORICAL,
        'ente_federativo_responsavel': CATEGORICAL,
    },
    'estabelecimentos': {
        'cnpj': {'sdtype': 'id', 'regex_format': '[0-9]{14}'},
        'cnpj_basico': {'sdtype': 'id'},
        'identificador_matriz_filial': CATEGORICAL,
        'nome_fantasia': {'sdtype': 'company', 'pii': True},
        'situacao_cadastral': CATEGORICAL,
        'data_situacao_cadastral': DATETIME,
        'motivo_situacao_cadastral': CATEGORICAL,
        'nome_cidade_exterior': CATEGORICAL,
        'pais': CATEGORICAL,
        'data_inicio_atividade': DATETIME,
        'cnae_fiscal_principal': CATEGORICAL,
        'cnae_fiscal_secundaria': UNKNOWN,
        'tipo_logradouro': CATEGORICAL,
        'logradouro': {'sdtype': 'street_name', 'pii': True},
        'numero': UNKNOWN,
        'complemento': UNKNOWN,
        'bairro': CATEGORICAL,
        'cep': {'sdtype': 'postcode', 'pii': True},
        'uf': CATEGORICAL,
        'municipio': CATEGORICAL,
        'ddd_1': CATEGORICAL,
        'telefone_1': {'sdtype': 'phone_number', 'pii': True},
        'ddd_2': CATEGORICAL,
        'telefone_2': {'sdtype': 'phone_number', 'pii': True},
        'ddd_fax': CATEGORICAL,
        'fax': {'sdtype': 'phone_number', 'pii': True},
        'correio_eletronico': {'sdtype': 'email', 'pii': True},
        'situacao_especial': CATEGORICAL,
        'data_situacao_especial': DATETIME,
    },
    'socios': {
        'cnpj_basico': {'sdtype': 'id'},
        'identificador_socio': CATEGORICAL,
        'nome_socio': {'sdtype': 'name', 'pii': True},
        'cpf_cnpj_socio': UNKNOWN,
        'qualificacao_socio': CATEGORICAL,
        'data_entrada_sociedade': DATETIME,
        'pais': CATEGORICAL,
        'representante_legal': UNKNOWN,
        'nome_representante': {'sdtype': 'name', 'pii': True},
        'qualificacao_representante_legal': CATEGORICAL,
        'faixa_etaria': CATEGORICAL,
    },
    'simples': {
        'cnpj_basico': {'sdtype': 'id'},
        'opcao_simples': CATEGORICAL,
        'data_opcao_simples': DATETIME,
        'data_exclusao_simples': DATETIME,
        'opcao_mei': CATEGORICAL,
        'data_opcao_mei': DATETIME,
        'data_exclusao_mei': DATETIME,
    },
}
PRIMARY_KEYS = {
    'empresas': 'cnpj_basico',
    'estabelecimentos': 'cnpj',
    'socios': None,
    'simples': 'cnpj_basico',
}
CORE_RELATIONSHIPS = [
    ('empresas', 'cnpj_basico', 'estabelecimentos', 'cnpj_basico'),
    ('empresas', 'cnpj_basico', 'socios', 'cnpj_basico'),
    ('empresas', 'cnpj_basico', 'simples', 'cnpj_basico'),
]
# (domain table, child table, child column)
DOMAIN_RELATIONSHIPS = [
    ('naturezas', 'empresas', 'natureza_juridica'),
    ('qualificacoes', 'empresas', 'qualificacao_responsavel'),
    ('motivos', 'estabelecimentos', 'motivo_situacao_cadastral'),
    ('paises', 'estabelecimentos', 'pais'),
    ('cnaes', 'estabelecimentos', 'cnae_fiscal_principal'),
    ('municipios', 'estabelecimentos', 'municipio'),
    ('qualificacoes', 'socios', 'qualificacao_socio'),
    ('qualificacoes', 'socios', 'qualificacao_representante_legal'),
    ('paises', 'socios', 'pais'),
]


def _read(sample_dir, table_name, max_bucket=None):
    path = os.path.join(sample_dir, f'{table_name}.parquet')
    if max_bucket is None:
        return pq.read_table(path).to_pandas()

    # read in small batches so the memory peak is one batch, not the whole prepared sample
    batches = []
    for batch in pq.ParquetFile(path).iter_batches(batch_size=50_000):
        batch = batch.filter(pc.less(batch.column('bucket'), max_bucket))
        if batch.num_rows:
            batches.append(batch)

    return pa.Table.from_batches(batches).drop(['bucket']).to_pandas()


def _clean(table_name, data):
    for column, column_metadata in COLUMNS[table_name].items():
        if column_metadata is DATETIME:
            data[column] = pd.to_datetime(data[column], format='%Y%m%d', errors='coerce')

    if table_name == 'empresas':
        data['capital_social'] = pd.to_numeric(
            data['capital_social'].str.replace(',', '.', regex=False), errors='coerce'
        )
    elif table_name == 'estabelecimentos':
        data['cnpj'] = data['cnpj_basico'] + data['cnpj_ordem'] + data['cnpj_dv']
        data = data.drop(columns=['cnpj_ordem', 'cnpj_dv'])

    return data[list(COLUMNS[table_name])]


def load(sample_dir, fraction, variant='core'):
    """Load the sample for ``fraction`` of the companies.

    Args:
        sample_dir (str):
            Folder with the parquet files written by ``prepare.py``.
        fraction (float):
            Fraction of the companies to keep. Must not exceed the prepared fraction.
        variant (str):
            ``'core'`` or ``'full'``.

    Returns:
        tuple[dict, Metadata, dict]:
            The data, the metadata and a report of the foreign key values that were nulled
            because they do not exist in the domain table.
    """
    max_bucket = max(1, round(fraction * NUM_BUCKETS))
    data = {
        table_name: _clean(table_name, _read(sample_dir, table_name, max_bucket))
        for table_name in COLUMNS
    }
    data['estabelecimentos'] = data['estabelecimentos'].drop_duplicates('cnpj')
    data['empresas'] = data['empresas'].drop_duplicates('cnpj_basico')
    data['simples'] = data['simples'].drop_duplicates('cnpj_basico')
    companies = set(data['empresas']['cnpj_basico'])
    for table_name in ('estabelecimentos', 'socios', 'simples'):
        table = data[table_name]
        data[table_name] = table[table['cnpj_basico'].isin(companies)].reset_index(drop=True)

    tables = {
        table_name: {'columns': {name: dict(meta) for name, meta in columns.items()}}
        for table_name, columns in COLUMNS.items()
    }
    for table_name, primary_key in PRIMARY_KEYS.items():
        if primary_key:
            tables[table_name]['primary_key'] = primary_key

    relationships = [
        {
            'parent_table_name': parent,
            'parent_primary_key': parent_key,
            'child_table_name': child,
            'child_foreign_key': child_key,
        }
        for parent, parent_key, child, child_key in CORE_RELATIONSHIPS
    ]
    nulled = {}
    if variant == 'full':
        for domain_table in {domain for domain, _, _ in DOMAIN_RELATIONSHIPS}:
            data[domain_table] = _read(sample_dir, domain_table).drop_duplicates('codigo')
            tables[domain_table] = {
                'primary_key': 'codigo',
                'columns': {'codigo': {'sdtype': 'id'}, 'descricao': UNKNOWN},
            }

        for domain_table, child, column in DOMAIN_RELATIONSHIPS:
            valid = data[child][column].isin(set(data[domain_table]['codigo']))
            invalid = data[child][column].notna() & ~valid
            nulled[f'{child}.{column}'] = float(invalid.mean())
            data[child].loc[invalid, column] = np.nan
            tables[child]['columns'][column] = {'sdtype': 'id'}
            relationships.append({
                'parent_table_name': domain_table,
                'parent_primary_key': 'codigo',
                'child_table_name': child,
                'child_foreign_key': column,
            })

    metadata = Metadata.load_from_dict({'tables': tables, 'relationships': relationships})
    return data, metadata, nulled
