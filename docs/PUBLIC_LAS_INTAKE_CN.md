# 公开 LAS / LAZ 小样本离线字段适配

本入口只核验一个明确指定的文件并读取其真实字段，不执行重建或性能评价。输入文件应由调用者按数据集许可取得；SHA256 只证明字节身份，不证明数据公开、授权范围或标签含义。工具没有网络访问和目录递归发现逻辑，也不会自动读取生产场景。

## 运行

在工具包根目录运行（已安装项目依赖）：

```powershell
.\.venv\Scripts\python.exe scripts/inspect_public_las.py --input "公开样本.laz" --expected-sha256 "从数据清单取得的64位SHA256" --output "新的输出目录/intake.json" --max-points 5000000 --chunk-size 100000
```

`--input`、`--expected-sha256`、`--output` 均必填。输出是一个新的 JSON 文件；已有文件、目录和符号链接一律拒绝覆盖。超过 `--max-points` 的输入在读取点记录前拒绝，绝不会悄悄截取前 N 点后宣称完整适配。`--chunk-size` 控制内存使用，默认 100000 点，最大 1000000 点。

工具在同一文件句柄上先校验 SHA256、再分块解码、最后重新校验 SHA256。哈希不匹配、输入中途改变、点数超限、实际点数与头部不一致、非法头部、非有限坐标或浮点字段会导致失败，成功前不创建结果文件。

## 字段契约

报告 `schema_version` 为 `railway.public-las-intake.v1`，对应完整 JSON Schema 位于 `public_las_intake.PUBLIC_LAS_INTAKE_SCHEMA`。写出前自动验证 schema。

| JSON 字段 | 含义与边界 |
| --- | --- |
| `source` | 文件名、字节数、实际 SHA256 和前后校验结果；不包含本机绝对路径 |
| `header.las_version` / `point_format_id` | 从真实头部读取；不假定所有数据均为 LAS 1.4 或同一格式 |
| `header.dimensions` | 每个标准及额外维度的原名、位宽、元素数和 dtype；位字段的 dtype 可为空 |
| `header.scales` / `offsets` | `scaled_xyz = raw_XYZ * scales + offsets` 的存储参数 |
| `decoded.minimum` / `maximum` | 全部已声明点记录的实际解码范围；空文件输出 null |
| `decoded.header_bounds_match` | 实际范围是否在半个存储步长及浮点容差内吻合头部；不吻合保留真实读数和 false |
| `decoded.classification.counts` | 按文件原始数值编码计数；语义类名需要单独核实数据集映射 |
| `decoded.point_source_id.counts` | 文件内原始值计数，包含 0；不自动解释成实例编号 |
| `field_interface` | 下一步适配可依赖的原始字段及尚未提取的候选/关系状态 |

所有浮点维度均要求有限，包括可选 GPS time 和 Extra Bytes。这样可以防止未处理的 NaN/Infinity 流入后续接口；它不是对数据集“测量质量合格”的认证。缺失的坐标单位或 CRS 不通过取值猜测，报告默认不宣称单位是米。

LAS scale 是量化存储步长。例如 scale 为 0.01，不能推出绝对测量精度或模型精度为一厘米。`classification` 和 `point_source_id` 不是自动成立的独立几何真值或连接真值，同一数值也不保证在不同文件中指向同一个对象。

## 当前候选公开样本

前一阶段已取得 SemanticRail3D-V2 的一个 train 样本。其来源及许可应继续使用独立数据清单追踪：

- 数据入口：[Zenodo record 15641832](https://zenodo.org/records/15641832)，DOI `10.5281/zenodo.15641832`。
- 本地已核查样本：`cloud180_Seg1.laz`。
- 固定 SHA256：`b81701df0f2f07dff14fe956f4b507c9ea719f416d99550da8e550b41ef36367`。
- 前一阶段记录为 907816 点、LAS 1.2、point format 3；本入口每次仍重新读取实际头部与所有点记录。

本适配器不从该数据集的标签构造“正确候选”，然后将同一标签当作独立真值报告性能。下一步应先明确候选提取使用哪些输入、留出划分和评价任务，标签能验证的范围与几何/连接真值必须分开。

## 测试

`tests/test_public_las_intake.py` 使用 `unittest` 和临时独立生成的微型 LAS / LAZ 文件，覆盖哈希失败、分块计数、真实格式与额外维度、点数限制、非有限值、旧输出保护、空文件和 LAS/LAZ 等价读数。不会读取固定客户目录、生产文件或联网下载数据。
