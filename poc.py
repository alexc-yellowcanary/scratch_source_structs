import pandas as pd
import numpy as np

import pyarrow as pa

FILE_1 = "/home/data/data.xlsx"
FILE_2 = "/home/remote/client/dbdump.csv"

#### pyarrow custom struct?!
source_type = pa.struct([
    pa.field('name', pa.string()),
    pa.field('row_ids', pa.list_(pa.string())),
])


def make_source_struct_field(s: pd.Series) -> pa.Array:
    return pa.Array.from_pandas(s, type=pa.list_(source_type))

struct_df = pd.DataFrame({
    "employee_code": ['aa', 'bb', 'cc', 'dd'],
    "name": ["alfred", "barnaby", "clarice", "damon"],
    "department": ["accounting", "accounting", "maintenance", "executive"],
    "height": [12, 15, 18, 21],
    "source": [
        [(FILE_1, ["1" ,"2"])],
        [(FILE_1, ["3"])],
        [(FILE_2, ["1024"])],
        [
            (FILE_1, ["8","16"]),
            (FILE_2, ["2048"]),
        ],
    ],
}).assign(
    source=lambda df: df.source.pipe(make_source_struct_field)
)

def agg_row_ids(s: pd.Series):
    if len(s) == 1:
        return s

    return np.concatenate(s)

def struct_source_agg(s: pd.Series) -> np.ndarray:
    expl = s.explode()

    df = (
        pd.DataFrame({ 
            'name': expl.apply(lambda x: x['name']), 
            'row_ids': expl.apply(lambda x: x['row_ids'])})
        .groupby('name', as_index=False)
        .agg(row_ids=("row_ids", agg_row_ids)))

    result = np.array(pa.array([t for t in df.itertuples(index=False, name=None)], source_type))

    return result

struct_grouped = struct_df.groupby("department").agg(
    sum_height=("height", "sum"),
    source=("source", struct_source_agg)
)

struct_grouped.to_parquet('test.pqt')
struct_grouped_read = pd.read_parquet('test.pqt')

