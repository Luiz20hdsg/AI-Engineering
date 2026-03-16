# 🐼 Pandas — Manipulação de Dados

> Pandas é A ferramenta para manipulação e análise de dados em Python. Domine Pandas = domine Data Science.

---

## 1. Estruturas Fundamentais

```python
import pandas as pd
import numpy as np

# Series — array 1D com índice
s = pd.Series([10, 20, 30, 40], index=['a', 'b', 'c', 'd'], name='valores')

# DataFrame — tabela 2D
df = pd.DataFrame({
    'nome': ['Alice', 'Bob', 'Charlie', 'Diana'],
    'idade': [25, 30, 35, 28],
    'salario': [5000, 7000, 8000, 6000],
    'departamento': ['TI', 'RH', 'TI', 'Marketing']
})
```

## 2. Leitura de Dados

```python
# CSV
df = pd.read_csv('dados.csv', sep=',', encoding='utf-8')
df = pd.read_csv('dados.csv', usecols=['col1', 'col2'], nrows=1000)  # eficiente

# Excel
df = pd.read_excel('dados.xlsx', sheet_name='Sheet1')

# Parquet (formato eficiente para Big Data)
df = pd.read_parquet('dados.parquet')

# JSON
df = pd.read_json('dados.json')

# SQL
import sqlite3
conn = sqlite3.connect('database.db')
df = pd.read_sql('SELECT * FROM tabela', conn)

# Salvar
df.to_csv('output.csv', index=False)
df.to_parquet('output.parquet')
```

## 3. Exploração Inicial

```python
df.head(10)          # primeiras 10 linhas
df.tail(5)           # últimas 5 linhas
df.shape             # (linhas, colunas)
df.dtypes            # tipos de cada coluna
df.info()            # resumo completo
df.describe()        # estatísticas descritivas
df.describe(include='object')  # para colunas categóricas
df.columns.tolist()  # lista de colunas
df.nunique()         # valores únicos por coluna
df.isnull().sum()    # contagem de nulos por coluna
df.duplicated().sum() # contagem de duplicatas
df.memory_usage(deep=True)  # uso de memória
```

## 4. Seleção e Filtragem

```python
# Seleção de colunas
df['nome']                    # Series
df[['nome', 'idade']]        # DataFrame

# .loc — por label
df.loc[0, 'nome']            # 'Alice'
df.loc[0:2, ['nome', 'idade']]  # linhas 0-2, colunas específicas

# .iloc — por posição
df.iloc[0, 0]                # primeiro elemento
df.iloc[0:3, 0:2]            # primeiras 3 linhas, 2 colunas

# Filtragem com condições
ti = df[df['departamento'] == 'TI']
altos_salarios = df[df['salario'] > 6000]
multiplas = df[(df['idade'] > 25) & (df['salario'] > 5000)]

# .query() — mais legível
df.query('idade > 25 and salario > 5000')
df.query('departamento in ["TI", "RH"]')

# .isin()
df[df['departamento'].isin(['TI', 'Marketing'])]
```

## 5. Transformações

```python
# Criar novas colunas
df['salario_anual'] = df['salario'] * 12
df['faixa_etaria'] = pd.cut(df['idade'], bins=[20, 30, 40, 50], labels=['jovem', 'adulto', 'senior'])

# Apply — aplicar função
df['nome_upper'] = df['nome'].apply(str.upper)
df['salario_categoria'] = df['salario'].apply(lambda x: 'alto' if x > 6000 else 'baixo')

# Map — mapear valores
dept_map = {'TI': 'Tecnologia', 'RH': 'Pessoas', 'Marketing': 'Marketing'}
df['dept_nome'] = df['departamento'].map(dept_map)

# Replace
df['departamento'].replace({'TI': 'Technology', 'RH': 'HR'})

# Renomear colunas
df.rename(columns={'nome': 'name', 'idade': 'age'}, inplace=True)

# Ordenação
df.sort_values('salario', ascending=False)
df.sort_values(['departamento', 'salario'], ascending=[True, False])
```

## 6. GroupBy e Agregação

```python
# GroupBy básico
df.groupby('departamento')['salario'].mean()
df.groupby('departamento').agg({
    'salario': ['mean', 'median', 'std', 'count'],
    'idade': ['min', 'max']
})

# Agregações customizadas
df.groupby('departamento').agg(
    salario_medio=('salario', 'mean'),
    num_funcionarios=('nome', 'count'),
    idade_media=('idade', 'mean')
).reset_index()

# Transform — mantém o shape original
df['salario_normalizado'] = df.groupby('departamento')['salario'].transform(
    lambda x: (x - x.mean()) / x.std()
)

# Pivot Table
pd.pivot_table(df, values='salario', index='departamento', 
               columns='faixa_etaria', aggfunc='mean', fill_value=0)
```

## 7. Merge e Join

```python
# Dados de exemplo
funcionarios = pd.DataFrame({
    'id': [1, 2, 3, 4],
    'nome': ['Alice', 'Bob', 'Charlie', 'Diana'],
    'dept_id': [101, 102, 101, 103]
})

departamentos = pd.DataFrame({
    'dept_id': [101, 102, 104],
    'dept_nome': ['Engenharia', 'Marketing', 'Vendas']
})

# Merge (SQL-style joins)
inner = pd.merge(funcionarios, departamentos, on='dept_id', how='inner')   # apenas matches
left = pd.merge(funcionarios, departamentos, on='dept_id', how='left')     # todos da esquerda
right = pd.merge(funcionarios, departamentos, on='dept_id', how='right')   # todos da direita
outer = pd.merge(funcionarios, departamentos, on='dept_id', how='outer')   # todos

# Concat — empilhar DataFrames
df_2023 = pd.DataFrame({'mes': [1, 2], 'vendas': [100, 200]})
df_2024 = pd.DataFrame({'mes': [1, 2], 'vendas': [150, 250]})
df_total = pd.concat([df_2023, df_2024], ignore_index=True)
```

## 8. Tratamento de Dados Faltantes

```python
# Detectar
df.isnull().sum()
df.isnull().mean() * 100  # percentual de nulos

# Remover
df.dropna()                          # remove qualquer linha com nulo
df.dropna(subset=['salario'])        # remove apenas se salario for nulo
df.dropna(thresh=3)                  # mantém linhas com pelo menos 3 não-nulos

# Preencher
df['salario'].fillna(df['salario'].median())     # com mediana
df['departamento'].fillna('Desconhecido')         # com valor fixo
df['salario'].fillna(method='ffill')              # forward fill
df['salario'].interpolate(method='linear')        # interpolação

# Preencher por grupo
df['salario'] = df.groupby('departamento')['salario'].transform(
    lambda x: x.fillna(x.median())
)
```

## 9. Dados Temporais

```python
# Converter para datetime
df['data'] = pd.to_datetime(df['data_str'], format='%Y-%m-%d')

# Extrair componentes
df['ano'] = df['data'].dt.year
df['mes'] = df['data'].dt.month
df['dia_semana'] = df['data'].dt.day_name()
df['trimestre'] = df['data'].dt.quarter

# Date range
datas = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
datas_mensais = pd.date_range(start='2024-01-01', periods=12, freq='ME')

# Resample (para séries temporais)
df_ts = df.set_index('data')
df_mensal = df_ts.resample('ME').agg({'vendas': 'sum', 'clientes': 'mean'})

# Rolling window
df['media_movel_7d'] = df['vendas'].rolling(window=7).mean()
df['media_movel_30d'] = df['vendas'].rolling(window=30).mean()
```

## 10. Method Chaining (Estilo Moderno)

```python
# Em vez de múltiplas linhas separadas, encadeie métodos:
resultado = (
    df
    .query('idade > 25')
    .assign(
        salario_anual=lambda x: x['salario'] * 12,
        bonus=lambda x: x['salario'] * 0.1
    )
    .groupby('departamento')
    .agg(
        salario_medio=('salario_anual', 'mean'),
        total_funcionarios=('nome', 'count')
    )
    .sort_values('salario_medio', ascending=False)
    .reset_index()
)
```

---

## 🏋️ Exercícios

1. Baixe um dataset do Kaggle e faça uma EDA completa
2. Implemente um pipeline de limpeza de dados com method chaining
3. Use groupby + transform para criar features baseadas em grupo
4. Pratique Window Functions equivalentes usando rolling e shift
5. Otimize um código Pandas — substitua loops por operações vetorizadas
