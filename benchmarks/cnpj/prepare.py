"""Prepare a company-level sample of the Brazilian CNPJ open data for the scale benchmark.

The Receita Federal publishes the CNPJ registry monthly as zipped, headerless, latin-1, ``;``
separated CSV files. This script streams every zip once and keeps the rows whose company
(``cnpj_basico``) falls in the first ``max_fraction`` of 10,000 buckets
(``int(cnpj_basico) % 10000``). Since the bucket only depends on the company, the sample keeps
every establishment, partner and Simples record of the sampled companies, so referential
integrity and the children-per-company distribution are preserved. Smaller fractions are subsets
of the stored sample (``bucket < fraction * 10000``).

Usage:
    python benchmarks/cnpj/prepare.py --release-dir ~/.cache/sdv-cnpj/2026-09 --max-fraction 0.05
"""

import argparse
import glob
import os
import time
import zipfile

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pv
import pyarrow.parquet as pq

NUM_BUCKETS = 10_000

LAYOUTS = {
    'empresas': [
        'cnpj_basico',
        'razao_social',
        'natureza_juridica',
        'qualificacao_responsavel',
        'capital_social',
        'porte_empresa',
        'ente_federativo_responsavel',
    ],
    'estabelecimentos': [
        'cnpj_basico',
        'cnpj_ordem',
        'cnpj_dv',
        'identificador_matriz_filial',
        'nome_fantasia',
        'situacao_cadastral',
        'data_situacao_cadastral',
        'motivo_situacao_cadastral',
        'nome_cidade_exterior',
        'pais',
        'data_inicio_atividade',
        'cnae_fiscal_principal',
        'cnae_fiscal_secundaria',
        'tipo_logradouro',
        'logradouro',
        'numero',
        'complemento',
        'bairro',
        'cep',
        'uf',
        'municipio',
        'ddd_1',
        'telefone_1',
        'ddd_2',
        'telefone_2',
        'ddd_fax',
        'fax',
        'correio_eletronico',
        'situacao_especial',
        'data_situacao_especial',
    ],
    'socios': [
        'cnpj_basico',
        'identificador_socio',
        'nome_socio',
        'cpf_cnpj_socio',
        'qualificacao_socio',
        'data_entrada_sociedade',
        'pais',
        'representante_legal',
        'nome_representante',
        'qualificacao_representante_legal',
        'faixa_etaria',
    ],
    'simples': [
        'cnpj_basico',
        'opcao_simples',
        'data_opcao_simples',
        'data_exclusao_simples',
        'opcao_mei',
        'data_opcao_mei',
        'data_exclusao_mei',
    ],
}
DOMAIN_TABLES = {
    'cnaes': 'Cnaes',
    'motivos': 'Motivos',
    'municipios': 'Municipios',
    'naturezas': 'Naturezas',
    'paises': 'Paises',
    'qualificacoes': 'Qualificacoes',
}
FILE_PREFIXES = {
    'empresas': 'Empresas',
    'estabelecimentos': 'Estabelecimentos',
    'socios': 'Socios',
    'simples': 'Simples',
}


def _open_csv(zip_path, column_names):
    archive = zipfile.ZipFile(zip_path)
    member = archive.namelist()[0]
    stream = archive.open(member)
    return pv.open_csv(
        stream,
        read_options=pv.ReadOptions(
            column_names=column_names, encoding='latin1', block_size=64 * 2**20
        ),
        parse_options=pv.ParseOptions(
            delimiter=';', quote_char='"', invalid_row_handler=lambda row: 'skip'
        ),
        convert_options=pv.ConvertOptions(
            column_types={name: pa.string() for name in column_names},
            strings_can_be_null=True,
            quoted_strings_can_be_null=True,
        ),
    )


def _bucket(cnpj_basico):
    numbers = pc.cast(cnpj_basico, pa.int64())
    return pc.subtract(numbers, pc.multiply(pc.divide(numbers, NUM_BUCKETS), NUM_BUCKETS))


def prepare_table(release_dir, output_dir, table_name, max_bucket):
    """Stream every zip of ``table_name`` and store the sampled rows as parquet."""
    columns = LAYOUTS[table_name]
    paths = sorted(glob.glob(os.path.join(release_dir, f'{FILE_PREFIXES[table_name]}*.zip')))
    output_path = os.path.join(output_dir, f'{table_name}.parquet')
    writer = None
    read_rows = kept_rows = 0
    start = time.perf_counter()
    for path in paths:
        reader = _open_csv(path, columns)
        for batch in reader:
            read_rows += batch.num_rows
            bucket = _bucket(batch.column('cnpj_basico'))
            mask = pc.less(bucket, max_bucket)
            table = pa.Table.from_batches([batch]).append_column('bucket', bucket).filter(mask)
            kept_rows += table.num_rows
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema)

            writer.write_table(table)

        print(  # noqa: T201
            f'  {os.path.basename(path)}: read {read_rows:,} kept {kept_rows:,} '
            f'({time.perf_counter() - start:.0f}s)',
            flush=True,
        )

    writer.close()
    return read_rows, kept_rows


def prepare_domain_tables(release_dir, output_dir):
    """Store the small code -> description tables as parquet."""
    for table_name, prefix in DOMAIN_TABLES.items():
        reader = _open_csv(os.path.join(release_dir, f'{prefix}.zip'), ['codigo', 'descricao'])
        pq.write_table(reader.read_all(), os.path.join(output_dir, f'{table_name}.parquet'))


def main():
    """Prepare the sample from the command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--release-dir', required=True)
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--max-fraction', type=float, default=0.05)
    args = parser.parse_args()

    release_dir = os.path.expanduser(args.release_dir)
    output_dir = args.output_dir or os.path.join(release_dir, f'sample_{args.max_fraction}')
    os.makedirs(output_dir, exist_ok=True)
    max_bucket = round(args.max_fraction * NUM_BUCKETS)

    prepare_domain_tables(release_dir, output_dir)
    for table_name in LAYOUTS:
        print(f'{table_name}:', flush=True)  # noqa: T201
        read_rows, kept_rows = prepare_table(release_dir, output_dir, table_name, max_bucket)
        print(f'  total read {read_rows:,} kept {kept_rows:,}', flush=True)  # noqa: T201


if __name__ == '__main__':
    main()
