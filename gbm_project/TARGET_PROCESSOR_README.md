# GBM靶点结构信息提取与处理流程

## 概述

本流程从mmCIF格式的蛋白质结构文件中提取详细的靶点信息，生成可供分子生成模型使用的结构化描述。

## 目录结构

```
gbm_project/
├── data/
│   ├── gbm_targets/              # 原始CIF文件
│   │   ├── EGFR-assembly1.cif.gz
│   │   ├── VEGF:VEGFR-assembly1.cif.gz
│   │   ├── IDH1-assembly1.cif.gz
│   │   ├── MGMT-assembly1.cif.gz
│   │   ├── PD-1:PD-L1-assembly1.cif.gz
│   │   ├── PI3K:AKT:mTOR-assembly1.cif.gz
│   │   ├── p53:MDM2.cif.gz
│   │   └── MDM2-assembly1.cif.gz
│   └── processed_targets/        # 处理后的输出
│       ├── EGFR_summary.json     # 快速查询用的摘要
│       ├── EGFR_structure.json   # 完整结构信息
│       ├── VEGFR_summary.json
│       ├── VEGFR_structure.json
│       └── ...
└── target_processor.py           # 处理主程序
└── target_query.py              # 查询工具
```

## 处理结果汇总

| 靶点 | 序列长度 | 结合位点数 | 药效团特征 |
|------|---------|-----------|-----------|
| EGFR | 330 aa | 2 | 147疏水, 101供体, 115受体 |
| VEGFR | 316 aa | 2 | 135疏水, 103供体, 108受体 |
| IDH1 | 425 aa | 1 | 164疏水, 145供体, 159受体 |
| MGMT | 207 aa | 2 | - |
| PD1_PDL1 | 129 aa | 1 | - |
| PI3K_AKT_mTOR | 960 aa | 2 | - |
| TP53_MDM2 | 114 aa | 1 | - |
| MDM2 | 85 aa | 1 | - |

## 输出文件格式

### *_summary.json (快速查询)

```json
{
  "target_name": "EGFR",
  "pdb_id": "XXXX",
  "metadata": {...},
  "sequence": {
    "sequence": "GAMGEAPNQALLRILKETEF...",
    "length": 330,
    "chains": ["A"]
  },
  "binding_sites": [
    {
      "site_id": "site_atp_binding",
      "site_name": "ATP-binding pocket (kinase domain)",
      "center": [0, 0, 0],
      "radius": 8.0,
      "key_residues": ["ASP855", "THR854", ...],
      "properties": {...}
    }
  ],
  "pharmacophore_features": {
    "hydrophobic_regions": ["ALA2", "MET3", ...],
    "hydrogen_bond_donors": ["ARG705", ...],
    "hydrogen_bond_acceptors": [...],
    "aromatic_regions": [...],
    "positive_charge": [...],
    "negative_charge": [...]
  }
}
```

### *_structure.json (完整信息)

包含所有原子的详细坐标和残基信息，适用于需要3D结构的应用场景。

## 使用方法

### 1. 运行完整处理流程

```bash
python3 target_processor.py
```

### 2. 使用查询工具

```python
from target_query import TargetQuery

# 初始化查询器
query = TargetQuery()

# 列出所有靶点
targets = query.list_targets()
print(targets)  # ['EGFR', 'VEGFR', 'IDH1', ...]

# 获取特定靶点信息
target = query.get_target("EGFR")
sequence = query.get_sequence("EGFR")

# 获取结合位点
sites = query.get_binding_sites("EGFR")

# 生成模型输入格式
model_input = query.generate_model_input("EGFR")

# 获取药效团特征
pharm = query.get_pharmacophore("EGFR")

# 比较多个靶点
comparison = query.compare_targets(["EGFR", "VEGFR", "IDH1"])
```

### 3. 命令行查询

```bash
python3 target_query.py
```

## 靶点结构信息隔离策略

每个靶点的结构信息已完全隔离，可独立使用：

1. **序列隔离**: 每个靶点有独立的氨基酸序列
2. **结合口袋隔离**: 每个靶点的结合位点独立编号和描述
3. **药效团隔离**: 每个靶点的理化性质特征独立存储

## 后续分子生成计划

基于提取的结构信息，可以：

1. **针对不同靶点分别生成**: 使用独立的`model_input`配置
2. **细化的结构提示**: 
   - 使用`binding_sites`中的关键残基作为药效团约束
   - 使用`pharmacophore_features`控制分子理化性质
   - 使用`sequence`进行序列特异性设计

## 扩展功能

### 添加新靶点

1. 将CIF文件放入`data/gbm_targets/`
2. 运行`python3 target_processor.py`
3. 新靶点信息会自动添加

### 自定义结合位点

修改`target_processor.py`中的`_calculate_binding_sites`方法，添加自定义口袋检测算法。


